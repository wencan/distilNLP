
from distilnlp._utils.unicode import is_printable_symbol

num_labels = 3

label_default = 0   # pad
label_head = 1
label_single = label_head
label_middle = 2
label_tail = label_middle
label_ignore = label_head

def text_to_features_labels(text, segments):
    features = []
    labels = []

    text_idx = 0
    segments_idx = 0
    while text_idx < len(text) and segments_idx < len(segments):
        if text[text_idx] == ' ' and segments[segments_idx] != ' ':
            # ignore/skip
            features.append(' ')
            labels.append(label_ignore)
            text_idx += 1
            continue

        segment = segments[segments_idx].strip()
        segment_length = len(segment)
        for idx, ch in enumerate(segment):
            while True:
                if text[text_idx] != ch:
                    if not is_printable_symbol(text[text_idx]): # skip unprintable symbols
                        features.append(text[text_idx])
                        labels.append(label_ignore)
                        text_idx+=1
                        continue

                    raise ValueError(f'char misstalk, "{text[text_idx]}" and "{ch}", text: "{text}", segments: "{segments}"')
                break
            
            feature = text[text_idx]
            label = label_default
            if idx == 0:
                if segment_length == 1:
                    label = label_single
                else:
                    label = label_head
            elif idx == segment_length-1:
                label = label_tail
            else:
                label = label_middle

            features.append(feature)
            labels.append(label)
            text_idx += 1

        segments_idx+=1
    return features, labels