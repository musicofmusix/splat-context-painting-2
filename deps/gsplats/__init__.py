import os
import sys

thirdparty_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if thirdparty_root not in sys.path:
    sys.path.insert(0, thirdparty_root)
