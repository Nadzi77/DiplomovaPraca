
from pathlib import Path
from utils.general import increment_path

FPS = 25
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
path = '..\..\datasets\DatasetsJur\ByteTrack\SGAN-%s\positions.txt' % FPS
incpath = Path(increment_path(Path(path), exist_ok=False))
print (incpath)

with open(incpath, 'a') as f:
    f.write('Hello world!')