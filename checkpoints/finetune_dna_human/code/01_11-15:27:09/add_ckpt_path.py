import sys
import os
import os.path as path


def add_path_to_dust3r(ckpt):
    HERE_PATH = os.path.dirname(os.path.abspath(ckpt))
    # workaround for sibling import
    sys.path.insert(0, HERE_PATH)
