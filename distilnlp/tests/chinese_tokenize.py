from unittest import TestCase, main

from distilnlp.chinese_tokenize.feature import label_single, label_head, label_middle, label_tail, label_ignore, text_to_features_labels

class TestFeature(TestCase):
    def test_segmented_mix_to_features_labels(self):
        text = '测试运行器 （test runner）'
        segments = ['测试', '运行器', '（', 'test', 'runner', '）']
        expect_features = list(text)
        expect_labels = [label_head, label_tail, label_head, label_middle, label_tail, label_ignore, label_single, label_head, label_middle, label_middle, label_tail, label_ignore, label_head, label_middle, label_middle, label_middle, label_middle, label_tail, label_single]
        self.assertEqual(text_to_features_labels(text, segments), (expect_features, expect_labels))

    def test_segmented_mystery_symbol_to_features_labels_0(self):
        text = '雷蒙•勒努瓦'
        segments = ['雷蒙', '•', '勒努瓦']
        expect_features = list(text)
        expect_labels = [label_ignore, label_head, label_tail, label_single, label_head, label_middle, label_tail]
        self.assertEqual(text_to_features_labels(text, segments), (expect_features, expect_labels))

    def test_segmented_mystery_symbol_to_features_labels_1(self):
        text = '''
'''
        segments = ['\ue003\ue003\n']
        expect_features = list(text)
        expect_labels = [label_head, label_tail, label_tail]
        self.assertEqual(text_to_features_labels(text, segments), (expect_features, expect_labels))

    def test_segmented_mystery_symbol_to_features_labels_2(self):
        text = '第二條　   中華民國之主權屬於國民全體。'
        segments = ['第二', '條', '中華', '民國', '之', '主權', '屬於', '國民', '全體', '。']
        expect_features = list(text)
        expect_labels = [label_head, label_tail, label_head, label_ignore, label_ignore, label_ignore, label_ignore, label_head, label_tail, label_head, label_tail, label_single, label_head, label_tail, label_head, label_tail, label_head, label_tail, label_head, label_tail, label_single]
        self.assertEqual(text_to_features_labels(text, segments), (expect_features, expect_labels))

    def test_segmented_mystery_symbol_to_features_labels_3(self):
        text = '''3.課程名稱：論文寫作    	    	2學分
'''
        segments = ['3', '.', '課程', '名稱', '：', '論文', '寫作', '2', '學分']
        expect_features = list(text)
        expect_labels = [label_single, label_single, label_head, label_tail, label_head, label_tail, label_single, label_head, label_tail, label_head, label_tail, 
                         label_ignore, label_ignore, label_ignore, label_ignore, label_ignore, label_ignore, label_ignore, label_ignore, label_ignore, label_ignore,
                         label_single, label_head, label_tail, label_ignore]
        self.assertEqual(text_to_features_labels(text, segments), (expect_features, expect_labels))

    def test_segmented_mystery_symbol_to_features_labels_4(self):
        # contain IDEOGRAPHIC SPACE and ZERO WIDTH NO-BREAK SPACE
        text = '''二一如二　二二如四　　二九一十八﻿
'''
        segments = ['二一', '如', '二\u3000二二', '如', '四\u3000\u3000', '二', '九', '一十八']
        expect_features = list(text)
        expect_labels = [label_head, label_tail, label_single, label_head, label_middle, label_middle, label_tail, label_single, label_head, label_middle, label_tail, 
                         label_single, label_single, label_head, label_middle, label_tail, label_ignore, label_ignore]
        self.assertEqual(text_to_features_labels(text, segments), (expect_features, expect_labels))

    def test_segmented_mystery_symbol_to_features_labels_5(self):
        # contain unprintable symbol
        text = '''a ­见卷四第八章。——译者注
'''
        segments = ['a', ' \xad', '见', '卷', '四', '第八', '章', '。', '——', '译者', '注']
        expect_features = list(text)
        expect_labels = [label_single, label_head, label_tail, label_single, label_single, label_single, label_head, label_tail, label_single, label_single, label_head, label_tail, label_head, label_tail, label_single, label_ignore]
        self.assertEqual(text_to_features_labels(text, segments), (expect_features, expect_labels))

    def test_segmented_mystery_symbol_to_features_labels_6(self):
        # contain Replacement Character
        text = '''過此以往,未之或知也已.�
'''
        segments = ['過', '此', '以往', ',', '未', '之', '或', '知', '也', '已', '.']
        expect_features = list(text)
        expect_labels = [label_single, label_single, label_head, label_tail, label_single, label_single, label_single, label_single, label_single, label_single, label_single, label_single, label_ignore, label_ignore]
        self.assertEqual(text_to_features_labels(text, segments), (expect_features, expect_labels))

if __name__ == '__main__':
    main()