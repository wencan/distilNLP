import os
import argparse
import queue
import logging
import time
import json

from distilnlp._utils.data import LMDBWriter
from .feature import text_to_features_labels

logger = logging.getLogger(__name__)
log = logger.info

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(processName)s %(filename)s:%(lineno)d %(message)s', level=logging.INFO)
    
    arg_parser = argparse.ArgumentParser(description='train punctuation normalize model.')
    arg_parser.add_argument('--text_file_paths', required=True, help='Paths to the original corpus file. Multiple path are separated by commas.')
    arg_parser.add_argument('--segmented_file_paths', required=True, help='Paths to the segmented corpus file. Multiple path are separated by commas.')
    arg_parser.add_argument('--save_path', required=True, help='The save path of preprocessed results.')
    args = arg_parser.parse_args()

    text_filepaths = args.text_file_paths.split(',')
    segmented_file_paths = args.segmented_file_paths.split(',')
    save_path = args.save_path

    with LMDBWriter(save_path, sync=True) as writer:
        for text_filepath in text_filepaths:
            short_set = set()

            text_filename = os.path.basename(text_filepath)
            segmented_filepath = ''
            for filepath in segmented_file_paths:
                filename = os.path.basename(filepath)
                if filename == text_filename:
                    segmented_filepath = filepath
                    break
            if segmented_filepath == '':
                log(f'No segmented file found for text file {text_filename}')
                continue

            with open(text_filepath) as textfile:
                with open(segmented_filepath) as segmentedfile:
                    for text in textfile:
                        line = segmentedfile.readline()

                        text = text.strip()
                        if len(text) > 512:
                            continue
                        if len(text) < 30:
                            if text in short_set:
                                continue
                            short_set.add(text)

                        segments = json.loads(line)
                        features, labels = text_to_features_labels(text, segments)
                        writer.add((features, labels))
        total = len(writer)
    
    with open(os.path.join(save_path, 'total.txt'), 'w') as outfile:
        outfile.write(str(total))

    logging.info(f'total: {total}')