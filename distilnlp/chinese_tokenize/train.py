import logging
import time
import argparse
import sys
import os.path
from typing import Tuple, Optional, Sequence

import tqdm
import torch
import torch.utils.data

from distilnlp._utils.data import LMDBDataSet

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
          optimizer:torch.optim.Optimizer
          ):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=codec.label_pad_value)

    model.train()
    total, total_loss = 0, 0
    forward_seconds, check_seconds, backward_seconds = 0, 0, 0

    for texts, labels_seqs in tqdm.tqdm(loader):
        features_seqs, targets_seqs, lengths = codec.Encode(texts, labels_seqs, device=DEVICE)

        t1 = time.time()

        optimizer.zero_grad()
        outputs = model(features_seqs) # -> (batch_size, max_length, num_labels)

        t2 = time.time()
        forward_seconds += t2 - t1

        outputs = torch.transpose(outputs, 1, 2) # -> (batch_size, num_labels, max_length)
        loss = loss_fn(outputs, targets_seqs)
        total_loss += loss.item()
        total += 1
        
        t3 = time.time()
        check_seconds += t3 -t2

        loss.backward()
        optimizer.step()

        t4 = time.time()
        backward_seconds += t4 - t3
    
    return total_loss/total, forward_seconds, check_seconds, backward_seconds


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

        loss_train, forward_seconds, check_seconds, backward_seconds = train(model, codec, train_loader, optimizer)
        log(f'Epoch {epoch+1} train loss: {loss_train:.8f}, '\
            f'forward propagation: {forward_seconds}s, check loss: {int(check_seconds)}s, backward propagation: {int(backward_seconds)}s')
        
        # acc_valid, loss_valid = valid(model, valid_loader)
        # log(f'Epoch {epoch+1} valid accuracy: {acc_valid:.6f} valid loss: {loss_valid:.6f}')

        latest_model_state = model.state_dict()
        
        # save model state
        if not os.path.exists(save_filedir):
            os.mkdir(save_filedir)
        version = f'{time.strftime("%Y%m%d%H%M%S", time.localtime())}_{epoch+1}'
        torch.save(latest_model_state, os.path.join(save_filedir, f'''punctuation_normalize_gru_state_dict_{version}.pt'''))
        txt_filepath = os.path.join(save_filedir, f'punctuation_normalize_gru_state_dict_{version}.txt')
        with open(txt_filepath, 'w') as outfile:
            outfile.write(str(model)+'\n')
            # outfile.write(f'valid accuracy: {acc_valid:.6f} valid loss: {loss_valid:.6f}')

        if lowest_loss_train is None or lowest_loss_train > loss_train:
            # best state
            lowest_loss_train = loss_train
            best_model_state = latest_model_state
        if lowest_loss_train is not None and lowest_loss_train < loss_train:
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
    arg_parser.add_argument('--preprocessed_path', required=True, help='Path to the preprocessed file.')
    arg_parser.add_argument('--preprocessed_total', type=int, default=0, help='The total number of preprocessed data.')
    arg_parser.add_argument('--vocab_filepath', help='Path to vocab dict.')
    arg_parser.add_argument('--padding_index', type=int, default=0, help='Index of padding token.')
    arg_parser.add_argument('--unknown_index', type=int, default=1, help='Index of unknown token.')
    arg_parser.add_argument('--embedding_filepath', help='Path to embedding weight file.')
    arg_parser.add_argument('--model_filepath', default='', help='Path to the pre-trained model file. If not provided, the script will train a new model from scratch.')
    arg_parser.add_argument('--save_filedir', required=True, help='File directory to save the trained model.')
    arg_parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate.')
    arg_parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs.')
    arg_parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    args = arg_parser.parse_args()

    attention_window_size = 3

    preprocessed_path = args.preprocessed_path
    preprocessed_total = args.preprocessed_total
    vocab_filepath = args.vocab_filepath
    padding_index = args.padding_index
    unknown_index = args.unknown_index
    embedding_filepath = args.embedding_filepath
    model_filepath = args.model_filepath
    save_filedir = args.save_filedir
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    log(f'PyTorch Version: {torch.__version__}')
    log(f'CUDA Version: {torch.version.cuda}')
    log(f"Using {DEVICE} device")

    vocab = load_vocab(vocab_filepath, unknown_index)
    print(f'vocab size: {len(vocab)}')

    with open(embedding_filepath, 'rb') as infile:
        embedding_weight = torch.load(infile)
    embedding_weight = embedding_weight.type(torch.get_default_dtype())

    codec = Codec(vocab, attention_window_size, 0, padding_index)
    model = AttentionTCN('local-attention', attention_window_size, embedding_weight, padding_index)
    model.to(DEVICE)
    print(model)
    for name, p in model.named_parameters():
        print(f'{name} parameters: {p.numel()}')
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # prepare data set
    if preprocessed_total == 0:
        with open(os.path.join(preprocessed_path, 'total.txt')) as infile:
            preprocessed_total = int(infile.readline())
    data_indexes = range(preprocessed_total)
    test_ratio = min(0.1, 1000/preprocessed_total)
    train_valid_ratio = 1 - test_ratio

    train_valid_indexes, test_indexes = torch.utils.data.random_split(data_indexes, [train_valid_ratio, test_ratio])
    log(f'train and valid: {len(train_valid_indexes)}, test: {len(test_indexes)}')

    cross_train_valid(model, codec, preprocessed_path, train_valid_indexes, batch_size, learning_rate, num_epochs)
