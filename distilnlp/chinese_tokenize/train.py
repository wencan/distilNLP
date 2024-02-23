import logging
import time
import argparse
import sys
import os.path
from typing import Optional, Sequence, Literal

import tqdm
import torch
import torch.utils.data
import torchtext

from distilnlp._utils.data import LMDBDataSet
from distilnlp._utils.profile import profile_trace

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

def collate_batch(batch):
    features_seqs = [item[0] for item in batch]
    labels_seqs =[item[1] for item in batch]

    return features_seqs, labels_seqs


def train(model:AttentionTCN, 
          codec:Codec,
          loader:torch.utils.data.DataLoader, 
          optimizer:Optional[torch.optim.Optimizer]=None,
          ):
    assert codec.label_pad_value == label_pad
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=label_pad)

    total, total_acc = 0, 0
    losses = []

    model.train()

    for texts, labels_seqs in tqdm.tqdm(loader):
        features_seqs, targets_seqs, lengths = codec.Encode(texts, labels_seqs, device=DEVICE)

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
          codec:Codec,
          loader:torch.utils.data.DataLoader, 
          ):
    assert codec.label_pad_value == label_pad
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=label_pad)

    total, total_acc = 0, 0
    losses = []

    model.eval()
    with torch.no_grad():
        for texts, labels_seqs in tqdm.tqdm(loader):
            features_seqs, targets_seqs, lengths = codec.Encode(texts, labels_seqs, device=DEVICE)

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


def cross_train_valid(model:AttentionTCN,
                      codec:Codec, 
                      preprocessed_path:str, 
                      data_indexes:Sequence[int], 
                      batch_size:int, 
                      learning_rate:float, 
                      num_epochs:int, 
                      lowest_lr:float = 0.000001,
                      ):
    valid_ratio = min(0.1, 100000/len(data_indexes))
    train_ratio = 1 - valid_ratio

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_model_state = None
    lowest_loss_train = None

    for epoch in range(num_epochs):
        train_indexes, valid_indexes = torch.utils.data.random_split(data_indexes, [train_ratio, valid_ratio])
        train_set = LMDBDataSet(preprocessed_path, train_indexes)
        valid_set = LMDBDataSet(preprocessed_path, valid_indexes)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate_batch)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size, shuffle=True, collate_fn=collate_batch)

        if epoch == 0:
            train_acc, train_loss = train_with_profile(model, codec, train_loader, optimizer)
        else:
            train_acc, train_loss= train(model, codec, train_loader, optimizer)
        log(f'Epoch {epoch+1} train accuracy: {train_acc:.8f}, train loss: {train_loss:.8f}')
        
        if epoch == 0:
            valid_acc , valid_loss= valid_with_profile(model, codec, valid_loader)
        else:
            valid_acc , valid_loss= valid(model, codec, valid_loader)
        log(f'Epoch {epoch+1} valid accuracy: {valid_acc:.8f}, valid loss: {valid_loss:.8f}')

        latest_model_state = model.state_dict()
        
        # save model state
        if not os.path.exists(save_filedir):
            os.mkdir(save_filedir)
        version = f'{time.strftime("%Y%m%d%H%M%S", time.localtime())}_{epoch+1}'
        torch.save(latest_model_state, os.path.join(save_filedir, f'''chinese_tokenize_state_dict_{version}.pt'''))
        txt_filepath = os.path.join(save_filedir, f'chinese_tokenize_state_dict_{version}.txt')
        with open(txt_filepath, 'w') as outfile:
            outfile.write(str(model)+'\n')
            outfile.write(f'valid accuracy: {valid_acc:.6f} valid loss: {valid_loss:.6f}')

        if lowest_loss_train is None or lowest_loss_train > train_loss:
            # best state
            lowest_loss_train = train_loss
            best_model_state = latest_model_state
        if lowest_loss_train is not None and lowest_loss_train < train_loss:
            # reset model
            model.load_state_dict(best_model_state)
            # reduce learning rate
            learning_rate = learning_rate * 0.5
            if learning_rate < lowest_lr:
                log(f'Finish early.')
                break
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            log(f'Update learning rate: {learning_rate}')
        
        del latest_model_state


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
    arg_parser.add_argument('--preprocessed_total', type=int, default=0, help='The total number of preprocessed data. If not provided, the script will attempt to search from the directory specified by preprocessed_path.')
    arg_parser.add_argument('--state_dict_filepath', default='', help='The file path of the state dict of the pre-trained model. If not provided, the script will train a new model from scratch.')
    arg_parser.add_argument('--vocab_filepath', help='Path to vocab dict. If training a new model from scratch, this is required.')
    arg_parser.add_argument('--padding_index', type=int, default=0, help='Index of padding token.')
    arg_parser.add_argument('--default_index', type=int, default=1, help='Index of unknown token.')
    arg_parser.add_argument('--min_freq', type=int, default=100, help='The minimum frequency needed to include a token in the vocabulary.')
    arg_parser.add_argument('--embedding_filepath', help='Path to embedding weight file. The script will also attempt to look up embedding weights from the state dict. If training a new model from scratch, this is required.')
    arg_parser.add_argument('--model_filepath', default='', help='Path to the pre-trained model file. If not provided, the script will train a new model from scratch.')
    arg_parser.add_argument('--save_filedir', required=True, help='File directory to save the trained model.')
    arg_parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate.')
    arg_parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs.')
    arg_parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    args = arg_parser.parse_args()

    attention_implementation: Literal['local-attention', 'natten'] = args.attention_implementation
    attention_window_size = args.attention_window_size

    preprocessed_path = args.preprocessed_path
    preprocessed_total = args.preprocessed_total
    state_dict_filepath = args.state_dict_filepath
    vocab_filepath = args.vocab_filepath
    padding_index = args.padding_index
    default_index = args.default_index
    min_freq = args.min_freq
    embedding_filepath = args.embedding_filepath
    model_filepath = args.model_filepath
    save_filedir = args.save_filedir
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size

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
    if embedding_filepath:
        with open(embedding_filepath, 'rb') as infile:
            embedding_weight = torch.load(infile, map_location=torch.device(DEVICE))
    if embedding_weight is None and state_dict:
        embedding_weight = state_dict['embedding.weight']
    if embedding_weight is None:
        raise ValueError('no embedding!')
    embedding_weight = embedding_weight.type(torch.get_default_dtype())

    # prepare data set
    if preprocessed_total == 0:
        with open(os.path.join(preprocessed_path, 'total.txt')) as infile:
            preprocessed_total = int(infile.readline())
    data_indexes = range(preprocessed_total)
    test_ratio = min(0.1, 100000/preprocessed_total)
    train_valid_ratio = 1 - test_ratio

    train_valid_indexes, test_indexes = torch.utils.data.random_split(data_indexes, [train_valid_ratio, test_ratio])
    log(f'train and valid: {len(train_valid_indexes)}, test: {len(test_indexes)}')

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

    test_set = LMDBDataSet(preprocessed_path, test_indexes)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True, collate_fn=collate_batch)
    if not state_dict is None:
        test_acc, test_loss = valid(model, codec, test_loader)
        log(f'test accuracy: {test_acc:.6f} test loss: {test_loss:.6f}')

    cross_train_valid(model, codec, preprocessed_path, train_valid_indexes, batch_size, learning_rate, num_epochs)

    test_acc, test_loss = valid(model, codec, test_loader)
    log(f'test accuracy: {test_acc:.6f} test loss: {test_loss:.6f}')