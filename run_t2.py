import sys,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import t2_adapter
t2_adapter.install()
runpy.run_module('cutamp.scripts.run_cutamp',run_name='__main__')
