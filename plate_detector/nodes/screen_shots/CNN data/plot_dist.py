#!/usr/bin/env python

import string
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
	
	char_list =[]

	dist ={}
	
	for num in range(0,10):
			char_list.append(str(num))
	for letter in string.ascii_uppercase:
			char_list.append(letter)


	for char in char_list:
		dist[char] = 0


	for filename in os.listdir(os.getcwd()):
		 if filename.endswith(".png"):
		 	dist[filename[-5]] = dist[filename[-5]] +1 
	plt.bar(list(dist.keys()), dist.values(), color='g')
  	plt.show()




