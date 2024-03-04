import os
import shutil
from typing import List, Iterator, Callable, Any, Sequence, Union
from collections import defaultdict, namedtuple

import torch
import torch.utils.data
import pickle
import lmdb

__all__ = [
    'LMDBWriter',
    'LMDBBucketWriter',
    'LMDBNamedWriter',
    'LMDBDataSet',
    'ConcatLMDBDataSet',
    'LMDBNamedDataSet',
    'BucketSampler',
]


class LMDBWriter:
    def __init__(self, path, map_size=1024*1024*1024*1024, sync=True, dumps=pickle.dumps):
        self._dumps = dumps

        if os.path.exists(path) and os.listdir(path):
            raise OSError('The directory is not empty.')

        self._db = lmdb.open(path, map_size=map_size, sync=sync)

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

    def close(self):
        if self._db:
            self._db.sync()
            self._db.close()
            self._db = None


class LMDBDataSet(torch.utils.data.Dataset):
    def __init__(self, path, loads=pickle.loads):
        super(LMDBDataSet, self).__init__()
        self._loads = loads

        self._path = path

        self._db = None

        db = lmdb.open(path, readonly=True)
        try:
            self._length = db.stat()['entries']
            self._map_size = db.info()['map_size']
        finally:
            db.close()
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        if self._db is None:
            self._db = lmdb.open(self._path, map_size=self._map_size, readonly=True)

        key = pickle.dumps(index)
        with self._db.begin() as tx_r:
            value = tx_r.get(key)
        if value is None:
            raise IndexError()

        if self._loads:
            value = self._loads(value)
        return value
    
    def close(self):
        if self._db:
            self._db.close()


class LMDBBucketWriter:
    def __init__(self, path, bucket_fn:Callable[[Any], str], map_size=1024*1024*1024*1024, sync=True, dumps=pickle.dumps):
        self._path = path
        self._bucket_fn = bucket_fn
        self._map_size = map_size
        self._sync = sync
        self._dumps = dumps
        self._writers = dict() # name -> env

        if os.path.exists(path):
            if os.listdir(path):
                raise OSError('The directory is not empty.')
        else:
            os.mkdir(path)

    def __len__(self):
        return sum([len(writer) for writer in self._writers.values()])

    def __enter__(self):
        return self
    
    def __exit__(self, *exc):
        self.close()
        return False
    
    def _get_writer(self, bucket:str):
        try:
            return self._writers[bucket]
        except KeyError:
            path = os.path.join(self._path, bucket)
            writer = LMDBWriter(path, map_size=self._map_size, sync=self._sync, dumps=self._dumps)
            self._writers[bucket] = writer
            return writer

    def add(self, item):
        self.add_batch((item, ))

    def add_batch(self, items):
        groups = defaultdict(list)
        for item in items:
            bucket = self._bucket_fn(item)
            groups[bucket].append(item)
        for bucket, members in groups.items():
            writer = self._get_writer(bucket)
            writer.add_batch(members)
    
    def close(self):
        for writer in self._writers.values():
            writer.close()
        self._writers = dict()


class ConcatLMDBDataSet(torch.utils.data.ConcatDataset):
    def __init__(self, paths: Union[str, Sequence[str]], loads=pickle.loads) -> None:
        if isinstance(paths, str):
            paths = [os.path.join(paths, path) for path in os.listdir(paths)]
            paths = filter(lambda path: os.path.isdir(path), paths)
        datasets = [LMDBDataSet(path, loads) for path in paths]
        super().__init__(datasets)


dbstat = namedtuple('dbstat', ('dbname', 'entries'))
__lmdb_dbs_stat_key__ = pickle.dumps('__imdb_metadata_dbs_stat__')


class LMDBNamedWriter:
    def __init__(self, path, map_size=1024*1024*1024*1024, sync=True, max_dbs=128, dumps=pickle.dumps):
        self._dumps = dumps

        if os.path.exists(path) and os.listdir(path):
            raise OSError('The directory is not empty.')

        self._db = lmdb.open(path, map_size=map_size, sync=sync, max_dbs=max_dbs)
        self._dbs = dict()
        self._max_index = defaultdict(int) # dbname -> max_index

    def __len__(self):
        return sum(self._max_index.values())
    
    def __enter__(self):
        return self
    
    def __exit__(self, *exc):
        self.close()
        return False
    
    def _open_named_db(self, dbname: str):
        try:
            return self._dbs[dbname]
        except KeyError:
            db = self._db.open_db(dbname.encode())
            self._dbs[dbname] = db
            return db
    
    def add(self, dbname, item):
        self.add_batch(dbname, (item, ))

    def add_batch(self, dbname:str, items):
        # If sync=False, don’t flush system buffers to disk when committing a transaction.

        db = self._open_named_db(dbname)

        with self._db.begin(db=db, write=True) as tx_x:
            for item in items:
                key = pickle.dumps(self._max_index[dbname])
                if self._dumps:
                    item = self._dumps(item)
                tx_x.put(key, item)

                self._max_index[dbname] +=1
    
    def dbs_stat(self) -> List[dbstat]:
        stats = [dbstat(dbname, max_index) for dbname, max_index in self._max_index.items()]
        stats = sorted(stats, key=lambda stat: stat.dbname)
        return stats

    def close(self):
        with self._db.begin(write=True) as tx_x:
            value = pickle.dumps(self.dbs_stat())
            tx_x.put(__lmdb_dbs_stat_key__, value)

        if self._db:
            self._db.sync()
            self._db.close()
            self._db = None


class _LMDBDNamedReader:
    def __init__(self, path, loads=pickle.loads):
        self._loads = loads

        self._path = path

        self._db = None
        self._dbs = dict()
        self._dbs_stat = []

        db = lmdb.open(path, readonly=True)
        try:
            self._map_size = db.info()['map_size']

            with db.begin() as tx_r:
                dbs_stat = tx_r.get(__lmdb_dbs_stat_key__)
                if dbs_stat is not None:
                    self._dbs_stat = pickle.loads(dbs_stat)
        finally:
            db.close()

    def __len__(self):
        return self._length
    
    def _open_named_db(self, dbname: str):
        try:
            return self._dbs[dbname]
        except KeyError:
            db = self._db.open_db(dbname.encode())
            self._dbs[dbname] = db
            return db
    
    def get(self, dbname:str, index:int):
        if self._db is None:
            self._db = lmdb.open(self._path, map_size=self._map_size, readonly=True, max_dbs=len(self._dbs_stat))

        db = self._open_named_db(dbname)
        key = pickle.dumps(index)
        with self._db.begin(db=db) as tx_r:
            value = tx_r.get(key, db=db)
        if value is None:
            raise IndexError()

        if self._loads:
            value = self._loads(value)
        return value

    def dbs_stat(self) -> List[dbstat]:
        return self._dbs_stat

    def close(self):
        if self._tx_r:
            self._tx_r.commit()
        if self._db:
            self._db.close()


class _LMDBNamedDBReader(torch.utils.data.Dataset):
    def __init__(self, base: _LMDBDNamedReader, dbstat: dbstat):
        self._base = base
        self._dbstat = dbstat
    
    def __len__(self):
        return self._dbstat.entries
    
    def __getitem__(self, index):
        return self._base.get(self._dbstat.dbname, index)


class LMDBNamedDataSet(torch.utils.data.ConcatDataset):
    def __init__(self, path: str):
        base = _LMDBDNamedReader(path)
        dbs = [_LMDBNamedDBReader(base, dbstat) for dbstat in base.dbs_stat()]
        super().__init__(dbs)


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
                    break

                if idx == self._batch_size:
                    yield indices
                    break