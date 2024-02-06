import os
import io

import torch

from .feature import text_to_features_labels, locate_punctuations, default_label
from .model import BiGRUModule

DEVICE = (
    "cuda" if torch.cuda.is_available()
    # else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def punctuation_normalize(model_or_modelfile, texts):
    features_seqs, labels_seqs, indexes_seqs = zip(*[text_to_features_labels(text) for text in texts])
    lengths = torch.as_tensor([len(features) for features in features_seqs])
    pad_features_seqs = torch.nn.utils.rnn.pad_sequence([torch.tensor(features, device=DEVICE) for features in features_seqs], batch_first=True)

    if isinstance(model_or_modelfile, str) or \
        isinstance(model_or_modelfile, os.PathLike) or \
        isinstance(model_or_modelfile, io.BytesIO):
        model = BiGRUModule()
        model.load_state_dict(torch.load(model_or_modelfile, map_location=torch.device(DEVICE)))
    else:
        model = model_or_modelfile
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits_seqs = model(pad_features_seqs, lengths)
    
    normalized = []

    for i, text in enumerate(texts):
        features = features_seqs[i]
        labels = labels_seqs[i]
        
        indexes = indexes_seqs[i]
        logits = logits_seqs[i]

        chs = list(text)
        for feature_idx, label in enumerate(labels):
            if label == default_label:
                continue
            feature = features[feature_idx]
            text_index = indexes[feature_idx]

            onehot_idx = torch.argmax(logits[feature_idx])
            punctuation = locate_punctuations(feature, onehot_idx)
            chs[text_index] = punctuation

        normalized.append(''.join(chs))
    
    return normalized


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description='punctuation normalize.')
    arg_parser.add_argument('--model_filepath', default='', help='Path to the pre-trained model file.')
    args = arg_parser.parse_args()

    model_filepath = args.model_filepath

    texts = ['11。 区分自然现象与自然灾害的关键因素是生命和财产的损失.',
             'The Workshop held plenary and working group meetings.',
             '2. Welcomes the report of the Secretary-General dated 13 May 1994 (S/1994/565);',
             'II. Working paper on data supply (Group A) 17',
             '导致COVID-19的病毒传播模式:对感染预防和控制方面的预防建议的影响',
             '零跑汽车则选择了电池底盘一体化(CTC)技术.',
             '走进浙江衢州极电新能源科技有限公司三电智能制造工厂（以下简称"衢州极电工厂"）电芯车间，仿佛置身于一间庞大的医院手术室。',
             '1。 几个代表团回顾了战略计划执行进展情况并展望未来',
             '54. 妇女署推出了增强妇女经济权能知识网关（参阅http：//www.empowerwomen。org），帮助各利益攸关方建立联系并分享经验和专长。',
             '请注意 float。hex（） 是实例方法，而 float。fromhex() 是类方法。',
             'Lorber （2008）审查了美国的多溴二苯醚接触情况，审查显示就BDE-209而言，食入104.8 纳克/天的土壤/灰尘是占最大比例的接触情况，其次为通过皮肤接触土壤/灰尘（25.2 纳克/天）。',
             'BDE-209的潜在生殖毒性也在稀有鮈鲫中体现（Li，2011年）。',
             'if concatenating bytes objects, you can similarly use bytes.join（） or io.BytesIO, or you can do in-place concatenation with a bytearray object。 bytearray objects are mutable and have an efficient overallocation mechanism',
            ]
    texts = punctuation_normalize(model_filepath, texts)
    
    for text in texts:
        print(text)