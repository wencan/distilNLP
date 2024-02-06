import os
import shutil

import torch
import pickle
import lmdb

class LMDBWriter(torch.utils.data.Dataset):
    def __init__(self, dir_path, map_size=1024*1024*1024*1024, sync=True, dumps=pickle.dumps):
        super().__init__()
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
            self._db.close()
            self._db = None


class LMDBDataSet(torch.utils.data.Dataset):
    def __init__(self, filepath, indexes, map_size=1024*1024*1024*1024, loads=pickle.loads):
        super().__init__()
        self._loads = loads

        self._filepath = filepath
        self._map_size = map_size

        self._db = None
        self._indexes = indexes

        self._tx_r = None
    
    def __len__(self):
        return len(self._indexes)
    
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