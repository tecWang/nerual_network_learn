import os

def mkdir(fpath):
	if not os.path.exists(fpath):
		os.makedirs(fpath)