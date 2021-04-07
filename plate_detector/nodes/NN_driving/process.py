#!/usr/bin/env python

import string
import os
import matplotlib.pyplot as plt
import gzip, pickle, pickletools
import cv2

import numpy as np


def plot_dist():
	dist ={}
	dist['left'] =0
	dist['right'] =0
	dist['straight'] =0



	for filename in os.listdir(os.getcwd()+'/train_data'):

		if 'left' in filename:
		 	dist['left'] = dist['left'] + 1  

		if 'right' in filename :
		 	dist['right'] = dist['right'] + 1 

		if 'straight' in filename :
		 	dist['straight'] = dist['straight'] + 1 
			
	
	plt.bar(list(dist.keys()), dist.values(), color='g')
  	plt.show()


def compress_save():
	x_data =[]

	y_data = []


	
	for filename in os.listdir(os.getcwd() + '/train_data'):
		
		

		if 'left' in filename:
		 	 sv_dir = np.asarray([1,0,0])

		elif 'right' in filename :
		 	sv_dir = np.asarray([0,0,1])
		 
		elif 'straight' in filename :
		 	sv_dir = np.asarray([0,1,0])
		else: 
		 	continue

		
		image = cv2.imread('./train_data/'+filename,0)

		image = cv2.resize(image, (0,0), fx=0.4,fy=0.4)
	


		x_data.append(image[:,:,np.newaxis])
		y_data.append(sv_dir)

		 

		
	x_data = np.array(x_data)
	y_data =np.array(y_data)

	print(x_data.shape)
	print(y_data.shape)
	

	with gzip.open('./driving_data_x.pbz2', "wb") as f:
	    pickled = pickle.dumps(x_data)
	    optimized_pickle = pickletools.optimize(pickled)
	    f.write(optimized_pickle)

	with gzip.open('./driving_data_y.pbz2', "wb") as f:
	    pickled = pickle.dumps(y_data)
	    optimized_pickle = pickletools.optimize(pickled)
	    f.write(optimized_pickle)




if __name__ == "__main__":
	
	compress_save()



	#plot_dist()

