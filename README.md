# distilNLP
The purpose of this project is to provide a set of natural language processing toolkits that are out-of-the-box, low-dependency, high-performance, and can achieve good results in solving real-world problems. At this stage, this project is mainly for learning and research use only.

# Install

```
pip install -U distilNLP
```

# Usage

## Text Normalize
Text normalization processing removes redundant characters and corrects incorrect punctuation.
```python
from distilnlp import text_normalize

text_normalize('The project was started in 2007 by David Cournapeau as a Google Summer of Code project， \nand since then many volunteers have contributed.\nSee the About us page for a list of core contributors. ')
# got: 'The project was started in 2007 by David Cournapeau as a Google Summer of Code project, and since then many volunteers have contributed. See the About us page for a list of core contributors.'

text_normalize('人权是所有人与生俱来的权利,不分国籍、性别、宗教或任何其他身份.')
# got: '人权是所有人与生俱来的权利，不分国籍、性别、宗教或任何其他身份。'

text_normalize([
    'This is an English 😇sentence。',
    '百度的网址是 http：//www.baidu。com'
])
# got: [
#   'This is an English sentence.',
#   '百度的网址是 http://www.baidu.com'
# ]
```
