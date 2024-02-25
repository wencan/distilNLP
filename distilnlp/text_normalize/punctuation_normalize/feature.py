import collections


__all__ = ['num_labels', 
           'locate_punctuations', 
           'default_label', 
           'default_label_index',
           'text_to_features_labels'
           ]

def feature_scale(feature):
    # Printable ASCII characters: https://en.wikipedia.org/wiki/ASCII#Printable_characters
    max = 126
    min = 32

    return (feature - min) / (max - min)

# feature as a feature value of punctuation, is also the classification of punctuation

punctuation_to_feature = collections.OrderedDict({
    '.': feature_scale(ord('.')), # period, half width
    '。': feature_scale(ord('.')), # period, full width
    '!': feature_scale(ord('!')), # exclamation, half width
    '！': feature_scale(ord('!')), # exclamation, full width
    "'": feature_scale(ord("'")), # apostrophe, half width
    '’':feature_scale(ord("'")), # apostrophe, full width
    '(': feature_scale(ord('(')), # left_round_bracket, half width
    ')': feature_scale(ord(')')), # right_round_bracket, half width
    '（': feature_scale(ord('(')), # left_round_bracket, full width
    '）': feature_scale(ord(')')), # right_round_bracket, full width
    '⟪': feature_scale(ord('⟪')), # left_book_title, half width
    '⟫': feature_scale(ord('⟫')), # right_book_title, right width
    '《': feature_scale(ord('⟪')), # left_book_title, full width
    '》': feature_scale(ord('⟫')), # right_book_title, full width
    ':': feature_scale(ord(':')), # colon, half width
    '：': feature_scale(ord(':')), # colon, full width
    ',': feature_scale(ord(',')), # comma, half width
    '，': feature_scale(ord(',')), # comma, full width
    ';': feature_scale(ord(';')), # semicolon, half width
    '；': feature_scale(ord(';')), # semicolon, full width
    '"': feature_scale(ord('"')), # quotes, half width
    '“': feature_scale(ord('"')), # quotes, full width
    '”': feature_scale(ord('"')), # quotes, full width
})

default_label = -1
num_labels = 3 # half with、fullwith(left)、fullwith right

punctuation_to_label = {}
feature_to_punctuations = collections.defaultdict(lambda:['']*num_labels)
cateory_index = collections.defaultdict(int)
for punctuation, feature in punctuation_to_feature.items():
    label = cateory_index[feature]
    punctuation_to_label[punctuation] = label

    feature_to_punctuations[feature][label] = punctuation

    cateory_index[feature]+=1


def locate_punctuations(feature, label):
    '''Locate the corresponding punctuation based on the feature value and index. 
    If there is no punctuation at the indexed position, raise an IndexError exception.
    '''
    category = feature_to_punctuations[feature]
    return category[label]


english_feature = feature_scale(ord('E'))
digit_feature = feature_scale(ord('D'))
chinese_feature = feature_scale(ord('C'))
# space_feature = feature_scale(ord('S'))
punctuation_feature = feature_scale(ord('P'))
other_feature = feature_scale(ord('O'))


def text_to_features_labels(text):
    features = []
    labels = []
    indexes = []
    pre_feature = ''
    for idx, ch in enumerate(text):
        feature = ''
        label = default_label

        code = ord(ch)
        if code == 32:
            continue
        elif (65 <= code <= 90) or (97 <= code <= 122):
            feature = english_feature
        elif 48 <= code <= 57:
            feature = digit_feature
        elif 19968 <= code <= 40959: # '\u4e00' <= ch <= '\u9fff'
            feature = chinese_feature
        else:
            try:
                feature = punctuation_to_feature[ch]
                label = punctuation_to_label[ch]
            except KeyError:
                feature = other_feature
        
        if feature == pre_feature:  # reduce feature dim
            continue

        features.append(feature)
        labels.append(label)
        indexes.append(idx)
        pre_feature = feature
    
    return features, labels, indexes
