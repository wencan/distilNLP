import cProfile
import pstats
import io
from typing import Callable

def profile_trace(f:Callable, print_fn:Callable=print, limit:int=10):
    def wrap(*args, **kwargs):
        with cProfile.Profile() as profile:
            try:
                return f(*args, **kwargs)
            finally:
                buff = io.StringIO()
                stats = pstats.Stats(profile, stream=buff)
                stats = stats.strip_dirs()
                stats = stats.sort_stats(pstats.SortKey.CUMULATIVE)
                stats.print_stats(limit)
                print_fn(buff.getvalue())
    return wrap