import logging
import time
import argparse
import sys
import os.path
from typing import Optional, Sequence, Literal, Callable

import tqdm
import torch
import torch.utils.data
import torchtext

from distilnlp.utils.data import ConcatLMDBDataSet
from distilnlp.utils.profile import profile_trace

from .feature import label_pad
from .model import AttentionTCN, Codec
from .vocab import load_vocab

logger = logging.getLogger(__name__)
log = logger.info

DEVICE = (
    "cuda" if torch.cuda.is_available()
    # else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def train(model:AttentionTCN, 
          loader:torch.utils.data.DataLoader, 
          optimizer:Optional[torch.optim.Optimizer]=None,
          ):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=label_pad, label_smoothing=0.1)

    total, total_acc = 0, 0
    losses = []

    model.train()

    for features_seqs, targets_seqs, lengths in tqdm.tqdm(loader):
        features_seqs = torch.tensor(features_seqs, device=DEVICE)
        targets_seqs = torch.tensor(targets_seqs, device=DEVICE)

        optimizer.zero_grad()
        logits_seqs = model(features_seqs) # -> (batch_size, max_length, num_labels)

        # loss
        weights = torch.transpose(logits_seqs, 1, 2) # -> (batch_size, num_labels, max_length)
        loss = loss_fn(weights, targets_seqs)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        # accuracy
        indices_seqs = torch.argmax(logits_seqs, dim=2)
        mask = targets_seqs != label_pad
        acc = torch.sum((indices_seqs == targets_seqs) & mask)
        total_acc += acc

        total += torch.sum(lengths)

    return total_acc/total, sum(losses)/len(losses)


def valid(model:AttentionTCN, 
          loader:torch.utils.data.DataLoader, 
          ):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=label_pad, label_smoothing=0.1)

    total, total_acc = 0, 0
    losses = []

    model.eval()
    with torch.no_grad():
        for features_seqs, targets_seqs, lengths in tqdm.tqdm(loader):
            features_seqs = torch.tensor(features_seqs, device=DEVICE)
            targets_seqs = torch.tensor(targets_seqs, device=DEVICE)

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


def cross_train_valid(model:AttentionTCN,
                      collate_batch:Callable, 
                      train_valid_set: torch.utils.data.Dataset, 
                      batch_size:int, 
                      learning_rate:float, 
                      num_epochs:int, 
                      checkpoint_interval:int,
                      num_workers: int,
                      lowest_lr:float = 0.000001,
                      ):
    valid_ratio = min(0.1, 100000/len(train_valid_set))
    train_ratio = 1 - valid_ratio

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_model_state = None
    lowest_train_loss = None

    for epoch in range(num_epochs):
        train_set, valid_set = torch.utils.data.random_split(train_valid_set, [train_ratio, valid_ratio])

        train_length =len(train_set)
        if checkpoint_interval == 0 or checkpoint_interval > train_length:
            checkpoint_interval = train_length
        checkpoint_lengths = [checkpoint_interval] * (train_length // checkpoint_interval)
        remain = train_length - sum(checkpoint_lengths)
        if remain:
            checkpoint_lengths.append(remain)
        assert sum(checkpoint_lengths) == train_length
        checkpoint_sets = torch.utils.data.random_split(train_set, checkpoint_lengths)

        # train
        for checkpoint, checkpoint_set in enumerate(checkpoint_sets):
            checkpoint_loader = torch.utils.data.DataLoader(checkpoint_set, batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_batch)
            if epoch == 0 and checkpoint == 0:
                train_acc, train_loss = train_with_profile(model, checkpoint_loader, optimizer)
            else:
                train_acc, train_loss= train(model, checkpoint_loader, optimizer)

            message = f'Epoch {epoch+1} checkpoint {checkpoint+1}/{len(checkpoint_sets)} train accuracy: {train_acc:.8f}, train loss: {train_loss:.8f}'
            log(message)
            save_checkpoint(model, epoch, checkpoint, save_filedir, message)

            if lowest_train_loss is None or lowest_train_loss > train_loss:
                # best state
                lowest_train_loss = train_loss
                best_model_state = model.state_dict()
            if lowest_train_loss is not None and lowest_train_loss < train_loss:
                # reset model
                model.load_state_dict(best_model_state)
                # reduce learning rate
                learning_rate = learning_rate * 0.5
                if learning_rate < lowest_lr:
                    log(f'Finish early.')
                    return
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                log(f'Update learning rate: {learning_rate}')
        
        # valid
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size, shuffle=True, collate_fn=collate_batch)
        if epoch == 0:
            valid_acc , valid_loss= valid_with_profile(model, valid_loader)
        else:
            valid_acc , valid_loss= valid(model, valid_loader)
        log(f'Epoch {epoch+1} valid accuracy: {valid_acc:.8f}, valid loss: {valid_loss:.8f}')


if __name__ == '__main__':
    logging.basicConfig(handlers=[
        logging.StreamHandler(sys.stdout),
    ], format='%(asctime)s %(process)d %(filename)s:%(lineno)d %(message)s')
    logger = logging.getLogger()
    if len(logger.handlers) == 1 and type(logger.handlers[0]) is logging.FileHandler: # Kaggle
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s %(process)d %(filename)s:%(lineno)d %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    arg_parser = argparse.ArgumentParser(description='train chinese tokenize model.')
    arg_parser.add_argument('--attention_implementation', default='local-attention', help='Implementation of Attention Mechanisms. Options: local-attention and natten.')
    arg_parser.add_argument('--attention_window_size', type=int, default=3, help='Size of attention window.')
    arg_parser.add_argument('--preprocessed_path', required=True, help='Path to the preprocessed file.')
    arg_parser.add_argument('--state_dict_filepath', default='', help='The file path of the state dict of the pre-trained model. If not provided, the script will train a new model from scratch.')
    arg_parser.add_argument('--vocab_filepath', help='Path to vocab dict. If training a new model from scratch, this is required.')
    arg_parser.add_argument('--padding_index', type=int, default=0, help='Index of padding token.')
    arg_parser.add_argument('--default_index', type=int, default=1, help='Index of unknown token.')
    arg_parser.add_argument('--min_freq', type=int, default=100, help='The minimum frequency needed to include a token in the vocabulary.')
    arg_parser.add_argument('--embedding_filepath', help='Path to embedding weight file. The script will prioritize searching for embedding weight from the state dict. If training a new model from scratch, this is required.')
    arg_parser.add_argument('--save_filedir', required=True, help='File directory to save the trained model.')
    arg_parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate.')
    arg_parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs.')
    arg_parser.add_argument('--batch_size', type=int, default=1024, help='Number of samples per batch.')
    arg_parser.add_argument('--checkpoint_interval', type=int, default=0, help='Save a checkpoint every how many samples. Default is all train samples.')
    arg_parser.add_argument('--num_workers', type=int, default=0, help='The number of worker subprocesses. 0 indicates the use of only the main process.')
    args = arg_parser.parse_args()

    attention_implementation: Literal['local-attention', 'natten'] = args.attention_implementation
    attention_window_size = args.attention_window_size

    preprocessed_path = args.preprocessed_path
    state_dict_filepath = args.state_dict_filepath
    vocab_filepath = args.vocab_filepath
    padding_index = args.padding_index
    default_index = args.default_index
    min_freq = args.min_freq
    embedding_filepath = args.embedding_filepath
    save_filedir = args.save_filedir
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    checkpoint_interval = args.checkpoint_interval
    num_workers = args.num_workers

    log(f'PyTorch Version: {torch.__version__}')
    log(f'CUDA Version: {torch.version.cuda}')
    log(f"Using {DEVICE} device")

    # load vocab
    ordered_dict = load_vocab(vocab_filepath)
    vocab = torchtext.vocab.vocab(ordered_dict, min_freq)
    vocab.set_default_index(default_index)
    print(f'vocab size: {len(vocab)}')

    # load state dict
    state_dict = None
    if state_dict_filepath:
        state_dict = torch.load(state_dict_filepath, map_location=torch.device(DEVICE))
    # load embedding
    embedding_weight = None
    if state_dict:
        embedding_weight = state_dict['embedding.weight']
    if embedding_weight is None and embedding_filepath:
        print(f'load Pretrained embedding weight.')
        with open(embedding_filepath, 'rb') as infile:
            embedding_weight = torch.load(infile, map_location=torch.device(DEVICE))
    if embedding_weight is None:
        raise ValueError('no embedding!')

    # prepare data set
    dataset = ConcatLMDBDataSet(preprocessed_path)
    test_ratio = min(0.1, 100000/len(dataset))
    train_valid_ratio = 1 - test_ratio

    train_valid_set, test_set = torch.utils.data.random_split(dataset, [train_valid_ratio, test_ratio])
    log(f'train and valid: {len(train_valid_set)}, test: {len(test_set)}')

    # load model
    codec = Codec(vocab, attention_window_size, padding_index)
    model = AttentionTCN(attention_implementation, attention_window_size, embedding_weight, padding_index)
    if not state_dict is None:
        model.load_state_dict(state_dict)
    else:
        log('train a new model from scratch.')
    model.to(DEVICE)
    print(model)
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    def collate_batch(batch):
        text_seqs = [item[0] for item in batch]
        labels_seqs =[item[1] for item in batch]
        features_seqs, target_seqs, lengths = codec.encode(text_seqs, labels_seqs, return_tensor=False)
        return features_seqs, target_seqs, lengths

    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True, collate_fn=collate_batch)
    if not state_dict is None:
        test_acc, test_loss = valid(model, test_loader)
        log(f'test accuracy: {test_acc:.6f} test loss: {test_loss:.6f}')

    cross_train_valid(model, collate_batch, train_valid_set, batch_size, learning_rate, num_epochs, checkpoint_interval, num_workers)

    test_acc, test_loss = valid(model, collate_batch, test_loader)
    log(f'test accuracy: {test_acc:.6f} test loss: {test_loss:.6f}')