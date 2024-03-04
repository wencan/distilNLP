import tempfile
import os
from unittest import TestCase, main

import torch.utils.data

from distilnlp.utils.unicode import is_printable_symbol
from distilnlp.utils.data import BucketSampler, LMDBBucketWriter, LMDBDataSet, LMDBNamedWriter, LMDBNamedDataSet

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
        bucket_lengths = [100, 3, 1300, 2400, 0, 196, 14, 3000, 2900, 87]
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
    
    def test_lmdb_bucket_writer(self):
        with tempfile.TemporaryDirectory(prefix='.test_lmdb_bucket_writer_', dir='.') as tmpdir:
            with LMDBBucketWriter(tmpdir) as writer:
                for num in range(100):
                    bucket = str(num//10)
                    writer.add(bucket, num)
                assert len(writer) == 100, len(writer)
            for idx in range(10):
                bucket = str(idx)
                path = os.path.join(tmpdir, bucket)
                dataset = LMDBDataSet(path)
                numbers = list(dataset)
                self.assertEqual(numbers, [idx*10+i for i in range(10)])

    def test_lmdb_named(self):
        with tempfile.TemporaryDirectory(prefix='.test_lmdb_named_', dir='.') as tmpdir:
            with LMDBNamedWriter(tmpdir) as writer:
                for num in range(100):
                    dbname = str(num//10)
                    writer.add(dbname, num)
                dbs_stat = writer.dbs_stat()
                self.assertEqual(len(dbs_stat), 10)
                self.assertEqual(len(writer), 100)
            dataset = LMDBNamedDataSet(tmpdir)
            self.assertEqual(len(dataset), 100)
            self.assertEqual(list(dataset), list(range(100)))
                

if __name__ == '__main__':
    main()