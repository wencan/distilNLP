import os
import shutil
from typing import List, Sized, Iterator

import torch
import torch.utils.data
import pickle
import lmdb

class LMDBWriter(torch.utils.data.Dataset):
    def __init__(self, dir_path, map_size=1024*1024*1024*1024, sync=True, dumps=pickle.dumps):
        super(LMDBWriter, self).__init__()
        self._dumps = dumps

        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        self._db = lmdb.open(dir_path, map_size=map_size, sync=sync)

        self._max_index = 0

    def __len__(self):
        return self._max_index
    
    def __enter__(self):
        return self
    
    def __exit__(self, *exc):
        self.close()
        return False
    
    def add(self, item):
        self.add_batch((item, ))

    def add_batch(self, items):
        # If sync=False, don’t flush system buffers to disk when committing a transaction.
        with self._db.begin(write=True) as tx_x:
            for item in items:
                key = pickle.dumps(self._max_index)
                if self._dumps:
                    item = self._dumps(item)
                tx_x.put(key, item)

                self._max_index +=1
    
    def get_indexes(self):
        indexes = list(range(self._max_index))
        return indexes 

    def close(self):
        if self._db:
            self._db.sync()
            self._db.close()
            self._db = None


class LMDBDataSet(torch.utils.data.Dataset):
    def __init__(self, filepath, map_size=1024*1024*1024*1024, loads=pickle.loads):
        super(LMDBDataSet, self).__init__()
        self._loads = loads

        self._filepath = filepath
        self._map_size = map_size

        self._db = None

        db = lmdb.open(filepath)
        try:
            self._length = db.stat()['entries']
            self._indexes = range(self._length)
        finally:
            db.close()

        self._tx_r = None
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        if self._db is None:
            self._db = lmdb.open(self._filepath, map_size=self._map_size, readonly=True)
        if self._tx_r is None:
            self._tx_r = self._db.begin()

        index = self._indexes[idx]
        key = pickle.dumps(index)
        value = self._tx_r.get(key)

        if self._loads:
            value = self._loads(value)
        return value
    
    def close(self):
        if self._tx_r:
            self._tx_r.commit()
        if self._db:
            self._db.close()


class BucketSampler(torch.utils.data.Sampler[List[int]]):
    '''
    Implementation of batch_sampler.
    BucketSampler ensures that each batch of data comes from the same bucket.
    '''
    def __init__(self, buckets: torch.utils.data.ConcatDataset, batch_size: int, drop_last: bool):
        super().__init__()

        self._batch_size = batch_size
        self._drop_last = drop_last

        self._cumulative_sizes = buckets.cumulative_sizes
        self._lengths = []
        self._cumulative_offsets = []
        for idx, cumulative_size in enumerate(self._cumulative_sizes):
            if idx == 0:
                self._cumulative_offsets.append(0)
                self._lengths.append(cumulative_size)
            else:
                self._cumulative_offsets.append(self._cumulative_sizes[idx-1])
                self._lengths.append(cumulative_size - self._cumulative_sizes[idx-1])
        self._total = sum(self._lengths)

    def __len__(self) -> int:
        if self._drop_last:
            return sum([length//self._batch_size for length in self._lengths])
        else:
            return sum([(length+self._batch_size-1)//self._batch_size for length in self._lengths])
    
    def __iter__(self) -> Iterator[int]:
        probabilities = torch.tensor(self._lengths, dtype=torch.float)
        samplers = [iter(torch.randperm(length)) for length in self._lengths]
        remain = self._total

        while remain > 0:
            bucket_indice = torch.multinomial(probabilities, 
                                              num_samples=1, 
                                              replacement=False
                                              )
            offset = self._cumulative_offsets[bucket_indice]
            sampler = samplers[bucket_indice]

            indices = [0] * self._batch_size
            idx = 0
            while True:
                try:
                    indices[idx] = offset + next(sampler)
                    idx+=1
                    remain -= 1
                except StopIteration:
                    if idx > 0 and not self._drop_last:
                        yield indices[:idx]

                    # samplers = samplers[:bucket_indice] + samplers[bucket_indice+1:]
                    # probabilities = torch.cat((probabilities[:bucket_indice], probabilities[bucket_indice+1:]))
                    break

                if idx == self._batch_size:
                    yield indices
                    break