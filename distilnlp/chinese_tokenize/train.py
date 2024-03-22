import logging
import time
import argparse
import sys
import os.path
from typing import Optional, Sequence, Literal, Callable
from datetime import datetime
from functools import partial

import tqdm
import torch
import torch.utils.data
import torchtext
from accelerate import Accelerator

from distilnlp.utils.data import ConcatLMDBDataSet, BucketSampler, concat_random_split
from distilnlp.utils.profile import profile_trace

from .feature import label_pad
from .model import AttentionTCN, Codec
from .vocab import load_vocab

logger = logging.getLogger(__name__)
log = logger.info


def train(accelerator: Accelerator,
          model:AttentionTCN, 
          loader:torch.utils.data.DataLoader, 
          optimizer:Optional[torch.optim.Optimizer],
          ):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=label_pad, label_smoothing=0.1)

    total, total_acc = 0, 0
    losses = []

    model, loader, optimizer = accelerator.prepare(model, loader, optimizer)
    device = accelerator.device
    model.train()

    for features_seqs, targets_seqs, lengths in tqdm.tqdm(loader, disable=not accelerator.is_main_process):

        optimizer.zero_grad()
        logits_seqs = model(features_seqs) # -> (batch_size, max_length, num_labels)

        # loss
        weights = torch.transpose(logits_seqs, 1, 2) # -> (batch_size, num_labels, max_length)
        loss = loss_fn(weights, targets_seqs)
        losses.append(loss.item())

        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()

        # accuracy
        indices_seqs = torch.argmax(logits_seqs, dim=2)
        mask = targets_seqs != label_pad
        acc = torch.sum((indices_seqs == targets_seqs) & mask)
        total_acc += acc

        total += torch.sum(lengths)

    return total_acc/total, sum(losses)/len(losses)


def valid(accelerator: Accelerator,
          model:AttentionTCN, 
          loader:torch.utils.data.DataLoader, 
          ):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=label_pad, label_smoothing=0.1)

    total, total_acc = 0, 0
    losses = []

    model, loader = accelerator.prepare(model, loader)
    device = accelerator.device
    model.eval()
    with torch.no_grad():
        for features_seqs, targets_seqs, lengths in tqdm.tqdm(loader, disable=not accelerator.is_main_process):

            logits_seqs = model(features_seqs) # -> (batch_size, max_length, num_labels)

            # loss
            weights = torch.transpose(logits_seqs, 1, 2) # -> (batch_size, num_labels, max_length)
            loss = loss_fn(weights, targets_seqs)
            losses.append(loss.item())

            # accuracy
            indices_seqs = torch.argmax(logits_seqs, dim=2)
            mask = targets_seqs != label_pad
            acc = torch.sum((indices_seqs == targets_seqs) & mask)
            total_acc += acc

            total += torch.sum(lengths)
    
    return total_acc/total, sum(losses)/len(losses)


train_with_profile = profile_trace(train, print_fn=log)
valid_with_profile = profile_trace(valid, print_fn=log)


def save_checkpoint(model, epoch:int, checkpoint:int, save_filedir:str, additional_text:str):
    if not os.path.exists(save_filedir):
        os.mkdir(save_filedir)
    version = f'{time.strftime("%Y%m%d%H%M%S", time.localtime())}_{epoch+1}_{checkpoint+1}'
    torch.save(model.state_dict(), os.path.join(save_filedir, f'''chinese_tokenize_state_dict_{version}.pt'''))
    txt_filepath = os.path.join(save_filedir, f'chinese_tokenize_state_dict_{version}.txt')
    with open(txt_filepath, 'w') as outfile:
        outfile.write(str(model)+'\n')
        if additional_text:
            outfile.write(additional_text)


def cross_train_valid(accelerator: Accelerator,
                      model:AttentionTCN,
                      collate_batch:Callable, 
                      worker_init_fn:Callable,
                      train_valid_set: torch.utils.data.ConcatDataset, 
                      batch_size:int, 
                      learning_rate:float, 
                      num_epochs:int, 
                      checkpoint_interval:int,
                      num_workers: int,
                      save_filedir:str,
                      ):
    valid_ratio = min(0.1, 100000/len(train_valid_set))
    train_ratio = 1 - valid_ratio

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)

    for epoch in range(num_epochs):
        train_set, valid_set = concat_random_split(train_valid_set, [train_ratio, valid_ratio])

        train_length =len(train_set)
        if checkpoint_interval == 0 or checkpoint_interval > train_length:
            checkpoint_interval = train_length
        checkpoint_lengths = [checkpoint_interval] * (train_length // checkpoint_interval)
        remain = train_length - sum(checkpoint_lengths)
        if remain:
            checkpoint_lengths.append(remain)
        assert sum(checkpoint_lengths) == train_length
        checkpoint_sets = concat_random_split(train_set, checkpoint_lengths)

        # train
        for checkpoint, checkpoint_set in enumerate(checkpoint_sets):
            batch_sampler = BucketSampler(checkpoint_set, batch_size, drop_last=False)
            checkpoint_loader = torch.utils.data.DataLoader(checkpoint_set, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_batch, worker_init_fn=worker_init_fn)
            checkpoint_loader = accelerator.prepare(checkpoint_loader)
            if checkpoint % 10 == 0:
                train_acc, train_loss = train_with_profile(accelerator, model, checkpoint_loader, optimizer)
            else:
                train_acc, train_loss= train(accelerator, model, checkpoint_loader, optimizer)

            message = f'Epoch {epoch+1} checkpoint {checkpoint+1}/{len(checkpoint_sets)} train accuracy: {train_acc:.8f}, train loss: {train_loss:.8f}'
            log(message)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_checkpoint(model, epoch, checkpoint, save_filedir, message)

            lr_scheduler.step(train_loss)
            if learning_rate != lr_scheduler.get_last_lr()[0]:
                learning_rate = lr_scheduler.get_last_lr()[0]
                log(f'Update learning rate: {learning_rate}')
        
        # valid
        batch_sampler = BucketSampler(valid_set, batch_size, drop_last=False)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_batch, worker_init_fn=worker_init_fn)
        valid_loader = accelerator.prepare(valid_loader)
        if epoch == 0:
            valid_acc , valid_loss= valid_with_profile(accelerator, model, valid_loader)
        else:
            valid_acc , valid_loss= valid(accelerator, model, valid_loader)
        log(f'Epoch {epoch+1} valid accuracy: {valid_acc:.8f}, valid loss: {valid_loss:.8f}')

        del batch_sampler
        del valid_loader
        del train_set
        del valid_set


def collate_batch(codec:Codec, device, batch):
    text_seqs = [item[0] for item in batch]
    labels_seqs =[item[1] for item in batch]
    features_seqs, target_seqs, lengths = codec.encode(text_seqs, labels_seqs, return_tensor=True, device=device)
    return features_seqs, target_seqs, lengths


def worker_init_fn(worker_id):
    pass

if __name__ == '__main__':
    logging.basicConfig(handlers=[
        logging.FileHandler(f'logs/{datetime.now().isoformat()}.log', mode='w'),
        logging.StreamHandler(sys.stdout),
    ], format='%(asctime)s %(process)d %(filename)s:%(lineno)d %(message)s')
    logger = logging.getLogger()
    if len(logger.handlers) == 1 and type(logger.handlers[0]) is logging.FileHandler: # Kaggle
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s %(process)d %(filename)s:%(lineno)d %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    torch.multiprocessing.set_start_method('spawn')

    arg_parser = argparse.ArgumentParser(description='train chinese tokenize model.')
    arg_parser.add_argument('--attention_implementation', default='local-attention', help='Implementation of Attention Mechanisms. Options: local-attention and natten.')
    arg_parser.add_argument('--attention_window_size', type=int, default=3, help='Size of attention window.')
    arg_parser.add_argument('--preprocessed_path', required=True, help='Path to the preprocessed file.')
    arg_parser.add_argument('--state_dict_filepath', default='', help='The file path of the state dict of the pre-trained model. If not provided, the script will train a new model from scratch.')
    arg_parser.add_argument('--vocab_filepath', help='Path to vocab dict. If training a new model from scratch, this is required.')
    arg_parser.add_argument('--embedding_dim', default=32, help='the size of each embedding vector.')
    arg_parser.add_argument('--add_specials_to_vocab', type=bool, default=True, help='Whether to insert special symbols into the vocabulary.')
    arg_parser.add_argument('--padding_index', type=int, default=0, help='Index of padding token.')
    arg_parser.add_argument('--default_index', type=int, default=1, help='Index of unknown token.')
    arg_parser.add_argument('--min_freq', type=int, default=1000, help='The minimum frequency needed to include a token in the vocabulary.')
    arg_parser.add_argument('--embedding_filepath', default='', help='Path to embedding weight file. The script will prioritize searching for embedding weight from the state dict. If training a new model from scratch, this is required.')
    arg_parser.add_argument('--save_filedir', required=True, help='File directory to save the trained model.')
    arg_parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate.')
    arg_parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs.')
    arg_parser.add_argument('--batch_size', type=int, default=64, help='Number of samples per batch.')
    arg_parser.add_argument('--checkpoint_interval', type=int, default=0, help='Save a checkpoint every how many samples. Default is all train samples.')
    arg_parser.add_argument('--num_workers', type=int, default=0, help='The number of worker subprocesses. 0 indicates the use of only the main process.')
    args = arg_parser.parse_args()

    attention_implementation: Literal['local-attention', 'natten'] = args.attention_implementation
    attention_window_size = args.attention_window_size

    preprocessed_path = args.preprocessed_path
    state_dict_filepath = args.state_dict_filepath
    vocab_filepath = args.vocab_filepath
    add_specials_to_vocab = args.add_specials_to_vocab
    padding_index = args.padding_index
    default_index = args.default_index
    min_freq = args.min_freq
    embedding_filepath = args.embedding_filepath
    embedding_dim = args.embedding_dim
    save_filedir = args.save_filedir
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    checkpoint_interval = args.checkpoint_interval
    num_workers = args.num_workers

    accelerator = Accelerator()
    device = accelerator.device

    log(f'PyTorch Version: {torch.__version__}')
    log(f'CUDA Version: {torch.version.cuda}')
    log(f"Using {device} device")
    log(f'Accelerator state: {accelerator.state}')

    # load vocab
    ordered_dict = load_vocab(vocab_filepath)
    specials=None
    if add_specials_to_vocab:
        specials = ['']*2
        specials[padding_index] = '<pad>'
        specials[default_index] = '<unk>'
    vocab = torchtext.vocab.vocab(ordered_dict, min_freq, specials)
    vocab.set_default_index(default_index)
    log(f'vocab size: {len(vocab)}')

    # load state dict
    state_dict = None
    if state_dict_filepath:
        state_dict = torch.load(state_dict_filepath, map_location=device)
    # load embedding
    embedding_weight = None
    if embedding_filepath:
        log(f'load Pretrained embedding weight.')
        with open(embedding_filepath, 'rb') as infile:
            embedding_weight = torch.load(infile, map_location=device)

    # prepare data set
    dataset = ConcatLMDBDataSet(preprocessed_path)
    test_ratio = min(0.1, 100000/len(dataset))
    train_valid_ratio = 1 - test_ratio

    train_valid_set, test_set = concat_random_split(dataset, [train_valid_ratio, test_ratio])
    log(f'train and valid: {len(train_valid_set)}, test: {len(test_set)}')

    # load model
    codec = Codec(vocab, attention_window_size, padding_index)
    if embedding_weight is not None:
        model = AttentionTCN.from_pretrained_embedding(attention_implementation, attention_window_size, embedding_weight, padding_index, freeze_embedding=not state_dict)
    else:
        model = AttentionTCN.from_new_embedding(attention_implementation, attention_window_size, len(vocab), embedding_dim, padding_index)
    if not state_dict is None:
        model.load_state_dict(state_dict)
    else:
        log('train a new model from scratch.')
    log(model)
    log(f'Total number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad and not isinstance(p, torch.nn.parameter.UninitializedParameter))}')
    
    def test():
        test_batch_sampler = BucketSampler(test_set, batch_size, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_sampler=test_batch_sampler, num_workers=num_workers, collate_fn=partial(collate_batch, codec, device), worker_init_fn=worker_init_fn)
        test_loader = accelerator.prepare(test_loader)
        test_acc, test_loss = valid(accelerator, model, test_loader)
        log(f'test accuracy: {test_acc:.6f} test loss: {test_loss:.6f}')

    if state_dict is not None:
        test()

    cross_train_valid(accelerator, model, partial(collate_batch, codec, device), worker_init_fn, train_valid_set, batch_size, learning_rate, num_epochs, checkpoint_interval, num_workers, save_filedir)

    test()