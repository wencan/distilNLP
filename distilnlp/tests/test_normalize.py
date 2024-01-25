from unittest import TestCase, main
from distilnlp import normalize

class TestNormalize(TestCase):
    def test_zh(self):
        self.assertEqual(normalize('zh', '人权是所有人与生俱来的权利,不分国籍、性别、宗教或任何其他身份.'), 
                                         '人权是所有人与生俱来的权利，不分国籍、性别、宗教或任何其他身份。')
        self.assertEqual(normalize('zh', '他说："你好吗？"'), 
                                         '他说：“你好吗？”')
        self.assertEqual(normalize('zh', '“他说："你好吗？"'), 
                                         '他说：“你好吗？”')
        self.assertEqual(normalize('zh', '他说："你好吗？"”'), 
                                         '他说：“你好吗？”')
        self.assertEqual(normalize('zh', '“他说：“你好吗？””'), 
                                         '“他说：“你好吗？””')
        self.assertEqual(normalize('zh', '你知道吗？'), 
                                         '你知道吗？')

    def test_en(self):
        self.assertEqual(normalize('en', 'The project was started in 2007 by David Cournapeau as a Google Summer of Code project， \nand since then many volunteers have contributed.\nSee the About us page for a list of core contributors. '), 
                                         'The project was started in 2007 by David Cournapeau as a Google Summer of Code project, and since then many volunteers have contributed. See the About us page for a list of core contributors.')

        self.assertEqual(normalize('en', 'How are you?”'), 
                                         'How are you?')
        self.assertEqual(normalize('en', '"How are you?'), 
                                         'How are you?')
        self.assertEqual(normalize('en', 'He said, "How are you?"'), 
                                         'He said, "How are you?"')

    def test_zh_en(self):
        self.assertEqual(normalize('zh', '这是一句夹杂着英文的中文文本。He said： "Who speaks English?"。结束.'), 
                                         '这是一句夹杂着英文的中文文本。He said: "Who speaks English?"。结束。')
        self.assertEqual(normalize('zh', '请注意 float.hex() 是实例方法，而 float.fromhex() 是类方法。'), 
                                         '请注意 float.hex() 是实例方法，而 float.fromhex() 是类方法。')
        self.assertEqual(normalize('zh', '请注意 float.hex() 是实例方法，而 float.fromhex() 是类方法。'), 
                                         '请注意 float.hex() 是实例方法，而 float.fromhex() 是类方法。')
        self.assertEqual(normalize('zh', '例如 var x = 42。'),
                                         '例如 var x = 42。')
        self.assertEqual(normalize('zh', '就像这样 let { bar } = foo。'),
                                         '就像这样 let { bar } = foo。')

    def test_zh_with_url(self):
        self.assertEqual(normalize('zh', '百度的网址是：  http：//baidu.com'),
                                         '百度的网址是： http://baidu.com')
    def test_remove_emoji(self):
        self.assertEqual(normalize('en', 'This is an English sent😇ence.'),
                                         'This is an English sentence.')
        self.assertEqual(normalize('zh', '这是中文⚓句子.'),
                                         '这是中文句子。')
    
    def test_remove_invisible_symbols(self):
        self.assertEqual(normalize('en', 'This \u202Cis an\u202D English\f sentence.'), 
                                         'This is an English sentence.')
    
    def test_remove_excess_symbols(self):
        self.assertEqual(normalize('zh', '“《联合国纪事》不是官方记录。'), 
                                         '《联合国纪事》不是官方记录。')
        self.assertEqual(normalize('zh', '《联合国纪事》不是官方记录。”'), 
                                         '《联合国纪事》不是官方记录。')
        self.assertEqual(normalize('en', '"The UN Chronicle  is not an official record. '), 
                                         'The UN Chronicle is not an official record.')
        self.assertEqual(normalize('en', 'The UN Chronicle  is not an official record."'), 
                                         'The UN Chronicle is not an official record.')

if __name__ == '__main__':
    main()