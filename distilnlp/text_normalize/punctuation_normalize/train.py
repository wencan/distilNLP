import logging
import sys
import time
import os
import os.path
import argparse

import torch
import torch.utils.data
import tqdm

from .feature import default_label
from .model import BiGRUModule
from distilnlp._utils.data import LMDBDataSet


logging.basicConfig(handlers=[
    logging.StreamHandler(sys.stdout),
], format='%(asctime)s %(process)d %(filename)s:%(lineno)d %(message)s')
logger = logging.getLogger()
if len(logger.handlers) == 1 and type(logger.handlers[0]) is logging.FileHandler: # Kaggle
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s %(process)d %(filename)s:%(lineno)d %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
log = logger.info

DEVICE = (
    "cuda" if torch.cuda.is_available()
    # else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def collate_batch(batch):
    lengths = torch.as_tensor([len(item[0]) for item in batch]) # cpu
    features_seqs = torch.nn.utils.rnn.pad_sequence([torch.tensor(item[0], device=DEVICE) for item in batch], batch_first=True)
    labels_seqs = torch.nn.utils.rnn.pad_sequence([torch.tensor(item[1], device=DEVICE) for item in batch], batch_first=True)

    return features_seqs, labels_seqs, lengths


def train(model, loader, optimizer):
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    total, total_loss = 0, 0
    forward_seconds, check_seconds, backward_seconds = 0, 0, 0

    for text_seqs, labels_seqs, lengths in tqdm.tqdm(loader):
        t1 = time.time()

        optimizer.zero_grad()
        outputs = model(text_seqs, lengths)

        t2 = time.time()
        forward_seconds += t2 - t1

        masks = torch.sum(labels_seqs, dim=2) == 1

        loss = 0
        for idx, labels in enumerate(labels_seqs): # label is one hot format. default: [0, 0, 0]
            mask = masks[idx]
            labels = labels[mask]
            if labels.size(0) == 0: # no punctuation
                continue

            output = outputs[idx]
            output = output[mask]
            loss_one = loss_fn(output, labels.float())

            loss += loss_one

            total += len(labels)
            total_loss += loss_one.item()

            # for saving time
            # acc = torch.sum(torch.argmax(labels, dim=1) == torch.argmax(output, dim=1)).float()
            # total_acc += acc.item()
        
        t3 = time.time()
        check_seconds += t3 -t2
        
        if loss == 0:
            continue
        loss.backward()
        optimizer.step()

        t4 = time.time()
        backward_seconds += t4 - t3
    
    return total_loss/total, forward_seconds, check_seconds, backward_seconds


def valid(model, loader):
    loss_fn = torch.nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        total, total_acc, total_loss = 0, 0, 0
        for features_seqs, labels_seqs, lengths in tqdm.tqdm(loader):
            outputs = model(features_seqs, lengths)

            masks = torch.sum(labels_seqs, dim=2) == 1

            loss = 0
            for idx, labels in enumerate(labels_seqs): # label is one hot format. default: [0, 0, 0]
                mask = masks[idx]
                labels = labels[mask]
                if labels.size(0) == 0: # no punctuation
                    continue

                output = outputs[idx]
                output = output[mask]
                loss_one = loss_fn(output, labels.float())

                loss += loss_one

                total += len(labels)
                total_loss += loss_one.item()

                acc = torch.sum(torch.argmax(labels, dim=1) == torch.argmax(output, dim=1)).float()
                total_acc += acc.item()
        
        return total_acc/total, total_loss/total


def cross_train_valid(model, preprocessed_path, data_indexes, learning_rate, num_epochs):
    valid_ratio = min(0.1, 100000/len(data_indexes))
    train_ratio = 1 - valid_ratio

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_indexes, valid_indexes = torch.utils.data.random_split(data_indexes, [train_ratio, valid_ratio])
        train_set = LMDBDataSet(preprocessed_path, train_indexes)
        valid_set = LMDBDataSet(preprocessed_path, valid_indexes)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate_batch)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size, shuffle=True, collate_fn=collate_batch)

        loss_train, forward_seconds, check_seconds, backward_seconds = train(model, train_loader, optimizer)
        log(f'Epoch {epoch+1} train loss: {loss_train:.6f}, '\
            f'forward propagation: {int(forward_seconds)}s, check loss: {int(check_seconds)}s, backward propagation: {int(backward_seconds)}s')

        acc_valid, loss_valid = valid(model, valid_loader)
        log(f'Epoch {epoch+1} valid accuracy: {acc_valid:.6f} valid loss: {loss_valid:.6f}')

        if not os.path.exists(save_filedir):
            os.mkdir(save_filedir)
        torch.save(model.state_dict(), os.path.join(save_filedir, f'''punctuation_normalize_gru_state_dict_{time.strftime("%Y%m%d%H%M%S", time.localtime())}_{epoch+1}.pt'''))



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='train punctuation normalize model.')
    arg_parser.add_argument('--preprocessed_path', required=True, help='Path to the preprocessed file.')
    arg_parser.add_argument('--preprocessed_total', type=int, required=True, help='The total number of preprocessed data.')
    arg_parser.add_argument('--model_filepath', default='', help='Path to the pre-trained model file. If not provided, the script will train a new model from scratch.')
    arg_parser.add_argument('--save_filedir', required=True, help='File directory to save the trained model.')
    arg_parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate.')
    arg_parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs.')
    arg_parser.add_argument('--batch_size', type=int, default=4096, help='Batch size.')
    args = arg_parser.parse_args()

    preprocessed_path = args.preprocessed_path
    preprocessed_total = args.preprocessed_total
    model_filepath = args.model_filepath
    save_filedir = args.save_filedir
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    log(f"Using {DEVICE} device")
    log(f'CUDA Version: {torch.version.cuda}')

    # prepare data set
    data_indexes = range(preprocessed_total)
    test_ratio = min(0.1, 100000/preprocessed_total)
    train_valid_ratio = 1 - test_ratio

    train_valid_indexes, test_indexes = torch.utils.data.random_split(data_indexes, [train_valid_ratio, test_ratio])
    log(f'train and valid: {len(train_valid_indexes)}, test: {len(test_indexes)}')

    # train
    model = BiGRUModule()
    if model_filepath:
        model.load_state_dict(torch.load(model_filepath))
    else:
        log('train a new model from scratch.')
    model = model.to(DEVICE)
    log(model)

    test_set = LMDBDataSet(preprocessed_path, test_indexes)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True, collate_fn=collate_batch)
    if model_filepath:
        acc_test, loss_test = valid(model, test_loader)
        log(f'test accuracy: {acc_test:.6f} test loss: {loss_test:.6f}')

    cross_train_valid(model, preprocessed_path, train_valid_indexes, learning_rate, num_epochs)

    acc_test, loss_test = valid(model, test_loader)
    log(f'test accuracy: {acc_test:.6f} test loss: {loss_test:.6f}')