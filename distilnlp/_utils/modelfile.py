import shutil
import os
import platform
import urllib.request
from typing import Literal

__all__ = ['downloaded_model_filepath']

app_name = 'distilNLP'

def cache_dir_path():
    '''Local cache file path.'''
    if platform.system() == "Windows":
        cache_dir = os.path.join(os.environ["LOCALAPPDATA"], "Cache", app_name)
    elif platform.system() == "Linux":
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", app_name)
    elif platform.system() == "Darwin":
        cache_dir = os.path.join(os.path.expanduser("~"), "Library", "Caches", app_name)
    else:
        raise NotImplementedError(f"Unsupported platform: {platform.system()}")
    return cache_dir


def downloaded_model_filepath(name: str, version: str, url: str, serialize_type:Literal['state_dict']='state_dict', postfix='pt'):
    '''Return the path of the already downloaded model file. 
    If the file does not exist, download it from the specified url.'''
    cache_dir = cache_dir_path()
    model_dir = os.path.join(cache_dir, 'model', name)
    filename = f'{serialize_type}_{version}.{postfix}'
    file_path = os.path.join(model_dir, filename)

    if os.path.exists(file_path):
        return file_path
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
       
    # other versions
    exists_filepaths = []
    for dir, _, filepaths in os.walk(model_dir):
        filepaths = [os.path.join(dir, filepath) for filepath in filepaths]
        exists_filepaths.extend(filepaths)
    
    # download
    filename, _ = urllib.request.urlretrieve(url)
    shutil.move(filename, file_path)

    # delete other versions
    for filepath in exists_filepaths:
        os.remove(filepath)

    return file_path