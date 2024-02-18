import os
import argparse
import multiprocessing
import queue
import logging
import time


from .feature import text_to_features_labels
from distilnlp._utils.data import LMDBWriter

def handle_worker(max_text_length:int, in_queue: multiprocessing.Queue, out_queue: multiprocessing.Queue, eof_event: multiprocessing.Event):
    get_count = 0
    put_count = 0
    while True:
        try:
            text = in_queue.get_nowait()
            get_count+=1
        except queue.Empty:
            if eof_event.is_set():
                break
            continue

        text = text.strip()
        if not text:
            continue
        if max_text_length and len(text) > max_text_length:
            continue

        features, labels, _ = text_to_features_labels(text)
        out_queue.put((features, labels))
        put_count+=1
    logging.info(f'get count: {get_count}, put count: {put_count}')


def write_worker(save_path: str, in_queue: multiprocessing.Queue, handled_event: multiprocessing.Event, total: multiprocessing.Value):
    with LMDBWriter(save_path, sync=False) as writer:
        count = 0
        while True:
            try:
                item = in_queue.get_nowait()
                writer.add(item) # 瓶颈在这里
                count += 1
            except queue.Empty:
                if handled_event.is_set():
                    break

        total.value = count
        logging.info(f'total: {count}')


def extra_file(save_path, filepaths, max_lines_per_file=0, max_text_length=1024):
    num_workers = min(os.cpu_count(), 3)
    in_queue = multiprocessing.Queue(maxsize=100000)
    out_queue = multiprocessing.Queue(maxsize=100000)
    eof_event = multiprocessing.Event()
    handled_event = multiprocessing.Event()
    total_value = multiprocessing.Value('i', 0)

    handler_processes = []
    for _ in range(num_workers):
        process = multiprocessing.Process(target=handle_worker, args=(max_text_length, in_queue, out_queue, eof_event))
        handler_processes.append(process)
        process.start()
    writer_process = multiprocessing.Process(target=write_worker, args=(save_path, out_queue, handled_event, total_value))
    writer_process.start()

    for filepath in filepaths:
        with open(filepath) as infile:
            start_time = time.time()
            count = 0
            while True:
                try:
                    text = next(infile)
                except StopIteration:
                    logging.info(f'{filepath} lines: {count}, elapsed time: {int(time.time() - start_time)}s')
                    break

                in_queue.put(text)

                count +=1
                if max_lines_per_file and count >= max_lines_per_file:
                    logging.info(f'{filepath} lines: {count}, elapsed time: {int(time.time() - start_time)}s')
                    break
    
    eof_event.set()
    for i, process in enumerate(handler_processes):
        if process.is_alive():
            process.join()

    handled_event.set()
    writer_process.join()
    total = total_value.value

    with open(os.path.join(save_path, 'total.txt'), 'w') as outfile:
        outfile.write(str(total))

    return total


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(processName)s %(filename)s:%(lineno)d %(message)s', level=logging.INFO)
    
    arg_parser = argparse.ArgumentParser(description='train punctuation normalize model.')
    arg_parser.add_argument('--file_paths', required=True, help='Paths to the preprocess corpus file. Multiple path are separated by commas.')
    arg_parser.add_argument('--save_path', required=True, help='The save path of preprocessed results.')
    arg_parser.add_argument('--max_lines_per_file', type=int, default=0, help='Maximum number of lines read per corpus file.')
    args = arg_parser.parse_args()

    file_paths = args.file_paths.split(',')
    save_path = args.save_path
    max_lines_per_file = args.max_lines_per_file

    total = extra_file(save_path, file_paths, max_lines_per_file)
    logging.info(f'total: {total}')