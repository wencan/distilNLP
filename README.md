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

norm_en = normalize('en', 'Proposed programme budget for the biennium 2006-2007： Section 29， Office of Internal Oversight Services。 ')
# got: 'Proposed programme budget for the biennium 2006-2007: Section 29, Office of Internal Oversight Services。'

norm_zh = normalize('zh', ' 2006-2007两年期拟议方案预算:第29款,内部监督事务厅.')
# got: '2006-2007两年期拟议方案预算：第29款，内部监督事务厅。'

norm_batch = normalize('en', ['Proposed programme budget for the biennium 2006-2007： Section 29， Office of Internal Oversight Services。 ', 
                         '（b）🥇 Also decides to amend rule 22 of the rules of procedure accordingly, with effect from its third session, to read as follows：'
                        ], n_job=-1)
# got: ['Proposed programme budget for the biennium 2006-2007: Section 29, Office of Internal Oversight Services。', 
#       '(b) Also decides to amend rule 22 of the rules of procedure accordingly, with effect from its third session, to read as follows:'
#      ]
```