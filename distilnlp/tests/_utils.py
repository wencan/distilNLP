from unittest import TestCase, main

from distilnlp._utils.unicode import is_mystery_symbol

class TestUnicode(TestCase):
    def test_is_mystery_symbol(self):
        s = '雷蒙•勒努瓦'
        ch = s[len(s)-1]
        self.assertEqual(is_mystery_symbol(ch), True)

        s = '🅰️插层剥离制备原子薄层材料的机理'
        ch = s[1]
        self.assertEqual(is_mystery_symbol(ch), True)

if __name__ == '__main__':
    main()