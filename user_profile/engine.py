# coding:  utf-8

import os
import time

basic_cmd = "python "
# generate the indices
indices = range(0, 200, 50)

for i in xrange(len(indices)-1):
	time.sleep(5)
	from_idx = str(indices[i])
	to_idx = str(indices[i + 1])
	cmd = basic_cmd + "/home/mjoys/user_profile2/projects/user_label.py " + from_idx + " " + to_idx
	print(cmd)
	os.system(cmd)

