import os, pickle
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    for dir_ in dirs:
        create_dir(dir_)

def create_dir(dir_):
    """
    dir - a directories to create if it is not found
    :param dir:
    :return exit_code: 0:success -1:failed
    """
    try:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('[Warning] Path [%s] already exists. Rename it to [%s]' % (path, new_name))
        os.rename(path, new_name)
    os.makedirs(path)
    
def mkdir_and_check(path):
    if os.path.exists(path):
        print('[Error] Path [%s] already exists.' % path)
        exit(-1)
    os.makedirs(path)
    
def pickle_data(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=2)

def unpickle_data(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

