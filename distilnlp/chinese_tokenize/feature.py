from typing import Sequence

from distilnlp._utils.unicode import is_printable_symbol, space_symbol, is_exceptional_symbol

num_labels = 2
label_pad = -1
label_head = 0
label_single = label_head
label_middle = 1
label_tail = label_middle
label_ignore = label_head

def text_to_features_labels(text:str, segments:Sequence[str]):
    '''Principle: Retain all characters from the original text and only generate segmentation markers.'''
    features = []
    labels = []

    text_idx = 0
    segments_idx = 0
    while text_idx < len(text) and segments_idx < len(segments):
        if segments[segments_idx][0] != text[text_idx]:
            if text[text_idx] in space_symbol:
                # ignore/skip space
                features.append(text[text_idx])
                labels.append(label_ignore)
                text_idx += 1
                continue

        segment = segments[segments_idx]
        segment_length = len(segment)
        for segment_idx, segment_ch in enumerate(segment):
            while True:
                if text[text_idx] != segment_ch:
                    if not is_printable_symbol(text[text_idx]) or is_exceptional_symbol(text[text_idx]): # skip unprintable symbols
                        features.append(text[text_idx])
                        labels.append(label_ignore)
                        text_idx+=1
                        continue

                    raise ValueError(f'char mismatch, "{text[text_idx]}" and "{segment_ch}", text index: {text_idx}')
                break
            
            feature = text[text_idx]
            label = label_head
            if segment_idx == 0:
                if segment_length == 1:
                    label = label_single
                else:
                    label = label_head
            elif segment_idx == segment_length-1:
                label = label_tail
            else:
                label = label_middle

            features.append(feature)
            labels.append(label)
            text_idx += 1

        segments_idx+=1
    
    while text_idx < len(text):
        if not (text[text_idx] in space_symbol or not is_printable_symbol(text[text_idx]) or is_exceptional_symbol(text[text_idx])):
            raise ValueError(f'Ignored non-whitespace characters: {text[text_idx]}. text index: {text_idx}')
        features.append(text[text_idx])
        labels.append(label_ignore)
        text_idx +=1

    return features, labels