# distilNLP
本项目的目的是提供一套开箱即用、低依赖、高性能、解决实际问题能取得良好效果的自然语言处理工具箱。现阶段，本项目仅限学习和研究使用。

The purpose of this project is to provide a set of natural language processing (NLP) tools that are ready-to-use, have low dependencies, high performance, and can achieve good results in solving real-world problems. At this stage, this project is only for learning and research purposes.

# Install

```
pip install -U distilNLP
```

# Usage

## Text Normalize
文本正规化移除多余的字符，矫正错误的标点符号。
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
