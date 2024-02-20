from unittest import TestCase, main

from distilnlp.chinese_tokenize.feature import label_default, label_single, label_head, label_middle, label_tail, label_ignore, text_to_features_labels

class TestFeature(TestCase):
    def test_segmented_mix_to_features_labels(self):
        text = '测试运行器 （test runner）'
        segments = ['测试', '运行器', '（', 'test', 'runner', '）']
        expect_features = ['测', '试', '运', '行', '器', ' ', '（', 't', 'e', 's', 't', ' ', 'r', 'u', 'n', 'n', 'e', 'r', '）']
        expect_labels = [label_head, label_tail, label_head, label_middle, label_tail, label_ignore, label_single, label_head, label_middle, label_middle, label_tail, label_ignore, label_head, label_middle, label_middle, label_middle, label_middle, label_tail, label_single]
        self.assertEqual(text_to_features_labels(text, segments), (expect_features, expect_labels))

    def test_segmented_mystery_symbol_to_features_labels(self):
        text = '雷蒙•勒努瓦'
        segments = ['雷蒙', '•', '勒努瓦']
        expect_features = ['', '雷', '蒙', '•', '勒', '努', '瓦']
        expect_labels = [label_ignore, label_head, label_tail, label_single, label_head, label_middle, label_tail]
        self.assertEqual(text_to_features_labels(text, segments), (expect_features, expect_labels))

        text = ''
        segments = ['\ue003\ue003\n']
        expect_features = ['', '']
        expect_labels = [label_head, label_tail]
        self.assertEqual(text_to_features_labels(text, segments), (expect_features, expect_labels))

if __name__ == '__main__':
    main()