# import string
import collections

# import jieba

__all__ = ['label_dim', 
           'locate_punctuations', 
           'default_label', 
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

label_dim = 3

punctuation_to_label = {}
feature_to_punctuations = collections.defaultdict(lambda:['']*label_dim)
cateory_index = collections.defaultdict(int)
for punctuation, feature in punctuation_to_feature.items():
    idx = cateory_index[feature]

    one_hot = [0] * label_dim
    one_hot[idx] = 1
    punctuation_to_label[punctuation] = one_hot

    feature_to_punctuations[feature][idx] = punctuation

    cateory_index[feature]+=1


def locate_punctuations(feature, index):
    '''Locate the corresponding punctuation based on the feature value and index. 
    If there is no punctuation at the indexed position, raise an IndexError exception.
    '''
    category = feature_to_punctuations[feature]
    return category[index]


# def is_english(word):
#     for ch in word:
#         if not ch in string.ascii_letters:
#             return False
#     return True

english_feature = feature_scale(ord('E'))
digit_feature = feature_scale(ord('D'))
chinese_feature = feature_scale(ord('C'))
# space_feature = feature_scale(ord('S'))
punctuation_feature = feature_scale(ord('P'))
other_feature = feature_scale(ord('O'))

# def word_kind(word):
#     if not word:
#         return ''
    
#     tag = ''
#     if is_english(word):
#         tag = english_feature # English
#     elif word[0] in string.digits:
#         tag = digit_feature # digit
#     elif '\u4e00' <= word[0] <= '\u9fff':
#         tag = chinese_feature # Chinese
#     elif word == ' ':
#         tag = space_feature # space
#     elif word in punctuation_to_feature:
#         tag = punctuation_feature # punctuation
#     else:
#         tag = other_feature # Other
#     return tag

# def word_feature(words, idx):
#     word = words[idx]

#     kind = word_kind(word)
#     if kind == punctuation_feature:
#         return punctuation_to_feature[word]
#     else:
#         return kind

default_label = [0]*label_dim

# def text_features_labels(text):
#     features = []
#     labels = []

#     words = jieba.lcut(text)
#     for idx, word in enumerate(words):
#         feature = word_feature(words, idx)
#         features.append(feature)

#         if word in punctuation_to_feature:
#             label = punctuation_to_label[word]
#             labels.append(label)
#         else:
#             labels.append(default_label)    # no care

#     return features, labels


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
