import os
import shutil
import random

path = './CAMUS/training'
files = os.listdir(path)
random.shuffle(files)
num_files = len(files)
for file in files[int(num_files*0.8):]:
	dir = os.path.join(path, file)
	shutil.move(dir, "./CAMUS/val")