import sys
import os.path as osp


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
add_path(osp.join(this_dir, '..'))
# print(this_dir)
# print(osp.join(this_dir, '..'))

