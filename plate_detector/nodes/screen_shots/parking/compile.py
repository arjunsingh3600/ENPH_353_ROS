#!/usr/bin/env python
import cv2
import os

for number in range(1,9):
	path =os.getcwd() + "/P{}".format(number)
	files =os.listdir(path)

	for index, file in enumerate(files):
		os.rename(os.path.join(path, file), os.path.join(os.getcwd(), ''.join([ 'P',str(number),'_',str(index), '.png'])))


# for file in os.listdir(os.path.join(os.getcwd(),'parking_all')):
# 	image = cv2.imread('./parking_all/'+file,0)

# 	if '.png' in file:
# 		image = cv2.resize(image, (0,0))

# 		image = cv2.imwrite(file,image)
