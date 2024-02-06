# distilNLP
distilNLP is a low-dependency, GPU-free natural language processing toolkit designed for building natural language processing applications and preprocessing for deep learning.

# Install

```
pip install -U distilNLP
```

# Usage

## Text Normalize
Text normalization processing. Correct common character errors in text from the internet.
```python
from distilnlp import text_normalize

norm = text_normalize('Proposed programme budget for the biennium 2006-2007： Section 29， Office of Internal Oversight Services。 ', lang='en')
# got: 'Proposed programme budget for the biennium 2006-2007: Section 29, Office of Internal Oversight Services。'

norm = text_normalize(' 2006-2007两年期拟议方案预算:😇第29款,内部监督事务厅.', lang='zh')
# got: '2006-2007两年期拟议方案预算：第29款，内部监督事务厅。'

norm = text_normalize('百度的网址是:  http：//baidu.com', lang='zh')
# got: '百度的网址是： http://baidu.com'
```