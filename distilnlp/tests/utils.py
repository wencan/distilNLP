import random
from unittest import TestCase, main

import torch.utils.data

from distilnlp.utils.unicode import is_printable_symbol
from distilnlp.utils.data import BucketSampler

class TestUnicode(TestCase):
    def test_is_printable_symbol(self):
        s = '雷蒙•勒努瓦'
        ch = s[len(s)-1]
        self.assertEqual(is_printable_symbol(ch), False)

        s = '🅰️插层剥离制备原子薄层材料的机理'
        ch = s[1]
        self.assertEqual(is_printable_symbol(ch), False)

class TestData(TestCase):
    def test_bucket_sampler(self):
        total = 10000
        numbers = list(range(total))
        bucket_lengths = [100, 3, 1300, 2400, 0, 200, 10, 3000, 2900, 87]
        assert sum(bucket_lengths) == total

        buckets = []
        start_pos = 0
        for length in bucket_lengths:
            end_pos = start_pos + length
            buckets.append(numbers[start_pos:end_pos])
            start_pos = end_pos
        assert total == sum([len(bucket) for bucket in buckets])

        dataset = torch.utils.data.ConcatDataset(buckets)
        batch_size = 7
        batch_sampler = BucketSampler(dataset, batch_size, drop_last=False)
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)

        loaded = [num for nums in loader for num in nums]
        self.assertEqual(total, len(loaded))
        self.assertEqual(numbers, sorted(loaded))

if __name__ == '__main__':
    main()