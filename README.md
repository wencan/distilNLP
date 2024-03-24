# distilNLP
本项目的目的是提供一套开箱即用、低依赖、高性能、解决实际问题能取得良好效果的自然语言处理工具箱。  
不同于其它NLP项目采用经典机器学习技术，或者基于大规模预训练模型去微调，本项目采用深度学习技术，从头设计小型语言模型，或基于高度压缩的预训练语言模型，以期取得较高推理准确率和较低资源消耗。  
现阶段，本项目仅限学习和研究使用。

The purpose of this project is to provide a set of out of the box, low-dependency, high-performance natural language processing toolkits that can achieve good results in solving real-world problems.  
Unlike other NLP projects that use classical machine learning techniques or fine-tune based on large-scale pre-trained models, this project uses deep learning techniques to design a series of small language models from scratch, or based on highly compressed pre-trained language models, in order to achieve higher inference accuracy and lower resource consumption.
At this stage, this project is only for learning and research.

# Usage
程序自动下载所需的模型文件。国内用户需要翻墙才能正常下载。  
The program automatically downloads the required model files.

## Text Normalize
文本正规化。  
移除多余的字符，矫正错误的标点符号。  
文本正规化包含标点符号正规化模型。但是没有足够的高质量的正规化文本语料用于训练。请考虑将enable_punctuation_normalize参数置为False，以禁用标点符号正规化。

Text normalization processing removes redundant characters and corrects incorrect punctuation.  
Text normalization includes a punctuation normalization model. But there is not enough high-quality normalized text corpus available for training. Please consider setting the enable_punctuation_normalize parameter to False to disable punctuation normalization.
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

## Chinese Tokenize
中文分词
```python
from distilnlp import distilnlp
chinese_tokenize([
    '6. 讲习班的参加者是在国家和区域应急机构和服务部门的管理岗位上工作了若干年的专业人员。', 
    '人权是所有人与生俱来的权利，不分国籍、性别、宗教或任何其他身份。'
])
# got: [
#   ['6', '.', ' ', '讲习班', '的', '参加者', '是', '在', '国家', '和', '区域', '应急', '机构', '和', '服务', '部门', '的', '管理', '岗位', '上', '工作', '了', '若干年', '的', '专业', '人员', '。'], 
#   ['人权', '是', '所有', '人', '与生', '俱来', '的', '权利', '，', '不', '分', '国籍', '、', '性别', '、', '宗教', '或', '任何', '其他', '身份', '。'],
# ]
```