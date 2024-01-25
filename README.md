# distilNLP
distilNLP is a low-dependency, GPU-free natural language processing toolkit designed for building natural language processing applications and preprocessing for deep learning.

# Install

```
pip install -U distilNLP
```

# Usage

## normalize
Normalize punctuation, remove unnecessary characters and invisible characters.
```python
from distilnlp import normalize

norm = normalize('Proposed programme budget for the biennium 2006-2007： Section 29， Office of Internal Oversight Services。 ')
# got: 'Proposed programme budget for the biennium 2006-2007: Section 29, Office of Internal Oversight Services。'

norm = normalize(' 2006-2007两年期拟议方案预算:😇第29款,内部监督事务厅.')
# got: '2006-2007两年期拟议方案预算：第29款，内部监督事务厅。'

norm = normalize('百度的网址是:  http：//baidu.com')
# got: '百度的网址是： http://baidu.com'
```