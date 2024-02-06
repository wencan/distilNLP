from unittest import TestCase, main
from distilnlp import text_normalize

class TextNormalize(TestCase):
    def test_zh(self):
        self.assertEqual(text_normalize('人权是所有人与生俱来的权利,不分国籍、性别、宗教或任何其他身份.'), 
                                        '人权是所有人与生俱来的权利，不分国籍、性别、宗教或任何其他身份。')
        self.assertEqual(text_normalize('你知道吗？'), 
                                        '你知道吗？')
        self.assertEqual(text_normalize('测试(一下)'),
                                        '测试（一下）')

    def test_en(self):
        self.assertEqual(text_normalize('The project was started in 2007 by David Cournapeau as a Google Summer of Code project， \nand since then many volunteers have contributed.\nSee the About us page for a list of core contributors. '), 
                                        'The project was started in 2007 by David Cournapeau as a Google Summer of Code project, and since then many volunteers have contributed. See the About us page for a list of core contributors.')

        self.assertEqual(text_normalize('How are you?”'), 
                                        'How are you?')
        self.assertEqual(text_normalize('"How are you?'), 
                                        'How are you?')
        self.assertEqual(text_normalize('He said, "How are you?"'), 
                                        'He said, "How are you?"')

    def test_it_mix_text(self):
        self.assertEqual(text_normalize('请注意 float。hex() 是实例方法，而 float.fromhex（） 是类方法。'), 
                                        '请注意 float.hex() 是实例方法，而 float.fromhex() 是类方法。')
        self.assertEqual(text_normalize('例如 var x = 42。'),
                                        '例如 var x = 42。')
        self.assertEqual(text_normalize('就像这样 let { bar } = foo。'),
                                        '就像这样 let { bar } = foo。')
        self.assertEqual(text_normalize('百度的网址是 http：//www.baidu。com'),
                                        '百度的网址是 http://www.baidu.com')
    
    def test_un_mix_text(self):
        # self.assertEqual(text_normalize('关键业绩指标2：人力资源'),
        #                                 '关键业绩指标2：人力资源')
        self.assertEqual(text_normalize('1。 几个代表团回顾了战略计划执行进展情况并展望未来'),
                                        '1. 几个代表团回顾了战略计划执行进展情况并展望未来')
        # self.assertEqual(text_normalize('在本议程项目下，审计委员会将向执行局提交2013年12月31日终了财政年度财政报告和已审计财务报表以及审计委员会的报告(A/69/5/Add.12)，供执行局参考。'),
                                        # '在本议程项目下，审计委员会将向执行局提交2013年12月31日终了财政年度财政报告和已审计财务报表以及审计委员会的报告（A/69/5/Add.12），供执行局参考。')
        # self.assertEqual(text_normalize('54. 妇女署推出了增强妇女经济权能知识网关(参阅www.empowerwomen。org)，帮助各利益攸关方建立联系并分享经验和专长。'),
                                        # '54. 妇女署推出了增强妇女经济权能知识网关（参阅www.empowerwomen.org），帮助各利益攸关方建立联系并分享经验和专长。')
        # self.assertEqual(text_normalize('Lorber （2008）审查了美国的多溴二苯醚接触情况，审查显示就BDE-209而言，食入104.8 纳克/天的土壤/灰尘是占最大比例的接触情况，其次为通过皮肤接触土壤/灰尘（25.2 纳克/天）。'),
                                        # 'Lorber （2008）审查了美国的多溴二苯醚接触情况，审查显示就BDE-209而言，食入104.8 纳克/天的土壤/灰尘是占最大比例的接触情况，其次为通过皮肤接触土壤/灰尘（25.2 纳克/天）。')

    def test_remove_emoji(self):
        self.assertEqual(text_normalize('This is an English sent😇ence.'),
                                        'This is an English sentence.')
        self.assertEqual(text_normalize('这是中文⚓句子.'),
                                        '这是中文句子。')
    
    def test_remove_invisible_symbols(self):
        self.assertEqual(text_normalize('This \u202Cis an\u202D English\f sentence.'), 
                                        'This is an English sentence.')
        # self.assertEqual(text_normalize('🅰️ 插层剥离制备原子薄层材料的机理'), 
        #                                 '插层剥离制备原子薄层材料的机理')
    
    def test_remove_excess_symbols(self):
        self.assertEqual(text_normalize('“《联合国纪事》不是官方记录。'), 
                                        '《联合国纪事》不是官方记录。')
        self.assertEqual(text_normalize('《联合国纪事》不是官方记录。”'), 
                                        '《联合国纪事》不是官方记录。')
        self.assertEqual(text_normalize('"The UN Chronicle  is not an official record. '), 
                                        'The UN Chronicle is not an official record.')
        self.assertEqual(text_normalize('The UN Chronicle  is not an official record."'), 
                                        'The UN Chronicle is not an official record.')

if __name__ == '__main__':
    main()