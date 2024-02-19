# distilNLP
本项目的目的是提供一套开箱即用、低依赖、高性能、解决实际问题能取得良好效果的自然语言处理工具箱。现阶段，本项目仅限学习和研究使用。

The purpose of this project is to provide a set of out of the box, low-dependency, high-performance natural language processing toolkits that can achieve good results in solving real-world problems. At this stage, this project is only for learning and research.

# Usage

## Text Normalize
文本正规化。  
移除多余的字符，矫正错误的标点符号。  
文本正规化包含标点符号正规化模型。该模型在测试语料的准确率为98.8%。但有些文本的标点符号准确率更高。请考虑将enable_punctuation_normalize参数置为False，以禁用标点符号正规化。

Text normalization processing removes redundant characters and corrects incorrect punctuation.  
Text normalization includes a punctuation normalization model. This model has an accuracy of 98.8% on the test corpus. However, some texts may have higher punctuation accuracy. Consider setting the enable_punctuation_normalize parameter to False to disable punctuation normalization.
```python
from distilnlp import text_normalize

text_normalize('The project was started in 2007 by David Cournapeau as a Google Summer of Code project， \nand since then many volunteers have contributed.\nSee the About us page for a list of core contributors. ')
# got: 'The project was started in 2007 by David Cournapeau as a Google Summer of Code project, and since then many volunteers have contributed. See the About us page for a list of core contributors.'

text_normalize('😇54。 妇女署推出了增强妇女经济权能知识网关(参阅www.empowerwomen。org)，帮助各利益攸关方建立联系并分享经验和专长。')
# got: '54. 妇女署推出了增强妇女经济权能知识网关（参阅www.empowerwomen.org），帮助各利益攸关方建立联系并分享经验和专长。'

text_normalize([
    'This is an English sentence。',
    '百度的网址是 http：//www.baidu。com'
])
# got: [
#   'This is an English sentence.',
#   '百度的网址是 http://www.baidu.com'
# ]
```
