import os
import argparse
import queue
import logging
import time
import json

from distilnlp.utils.data import LMDBBucketWriter
from .feature import text_to_features_labels

logger = logging.getLogger(__name__)
log = logger.info

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(processName)s %(filename)s:%(lineno)d %(message)s', level=logging.INFO)
    
    arg_parser = argparse.ArgumentParser(description='train punctuation normalize model.')
    arg_parser.add_argument('--text_file_paths', required=True, help='Paths to the original corpus file. Multiple path are separated by commas.')
    arg_parser.add_argument('--segmented_file_paths', required=True, help='Paths to the segmented corpus file. Multiple path are separated by commas.')
    arg_parser.add_argument('--save_path', required=True, help='The save path of preprocessed results.')
    arg_parser.add_argument('--max_length', type=int, default=512, help='The maximum length of the text. Text exceeding this length will be discarded.')
    arg_parser.add_argument('--min_length', type=int, default=0, help='The minimum length of the text. Text exceeding this length will be discarded.')
    args = arg_parser.parse_args()

    text_filepaths = args.text_file_paths.split(',')
    segmented_file_paths = args.segmented_file_paths.split(',')
    max_length = args.max_length
    min_length = args.min_length
    save_path = args.save_path

    def bucket_fn(item):
        features = item[0]
        length = len(features)
        bucket_idx = length//32
        bucket = f'{bucket_idx*32}_{(bucket_idx+1)*32}'
        return bucket

    with LMDBBucketWriter(save_path, bucket_fn, sync=False) as writer:
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

            log(f'{text_filepath} -- {segmented_filepath}')

            line_no =0
            with open(text_filepath) as textfile:
                with open(segmented_filepath) as segmentedfile:
                    for idx, text in enumerate(textfile):
                        text = text.strip()
                        line_no +=1
                        line = segmentedfile.readline()
                        length = len(text)
                        if length == 0:
                            continue

                        if max_length and length >= max_length:
                            continue
                        if min_length and length < min_length:
                            continue

                        if length < 30:
                            if text in short_set:
                                continue
                            short_set.add(text)

                        try:
                            segments = json.loads(line)
                        except json.decoder.JSONDecodeError as e:
                            raise ValueError(f'cannot load json [{line}], line no: {line_no}, text: [{text}]') from e

                        if len(segments) == 0:  # ignored
                            continue
                        if len(segments) > length*9//10:
                            # low quality
                            continue

                        try:
                            features, labels = text_to_features_labels(text, segments)
                        except (ValueError, IndexError) as e:
                            log(f'skip line no: [{line_no}], text: [{text}], segments: [{segments}], exception: {e}')
                            continue
                        writer.add((features, labels))
        total = len(writer)
    
    with open(os.path.join(save_path, 'total.txt'), 'w') as outfile:
        outfile.write(str(total))

    log(f'total: {total}')