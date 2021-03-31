#!/usr/bin/env python

import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt




def car_mask(image):
	""" take raw frame and return a mask of the car. Using color thresholding

	 """


	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


	# reduce size
	
	#get blues
	lower_blue_bound =(120,240, 0)
	upper_blue_bound =(130,255,255)
	#get whites
	lower_white_bound =(0,0,85)
	upper_white_bound =(360,40,150)


	mask2 = cv2.inRange(image,lower_blue_bound,upper_blue_bound) 

	
	mask = cv2.inRange(image,lower_white_bound,upper_white_bound)


	# remove noise 
	erode_kernel = np.ones((5,5), np.uint8)
	dilate_kernel = np.ones((25,25), np.uint8)


	mask = cv2.erode(mask, erode_kernel, iterations=2) 
	mask = cv2.dilate(mask, dilate_kernel, iterations=1) 
	dilate_kernel = np.ones((30,30), np.uint8)
	mask2  =cv2.dilate(mask2, dilate_kernel, iterations=1) 


	

	return  cv2.bitwise_and(mask,mask,mask= mask2)



	


	



def car_contours(image,max_only = True):

	mask = car_mask(image)

	im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	

	if max_only:
		
		try:
			c = max(contours, key = cv2.contourArea)
		except:
			print('empty')
			return [],False

		return c,(area< 30000 and area>15000)
	else:
		return contours,True






def show_image(show_mask = True):
	img = cv2.imread("./plate.png",1)

	img2= img.copy()




	#masks

	if show_mask:

		masked_img = cv2.bitwise_and(img,img,mask= car_mask(img))
		cv2.imshow("images",np.hstack([cv2.resize(img, (0,0), fx=0.5, fy=0.5) ,cv2.resize(masked_img, (0,0), fx=0.5, fy=0.5) ]))

	else:
		contours,_ =car_contours(img,False)
		contours =np.asarray(contours)

		# print(contours[])
		filter_area = [cv2.contourArea(c) >1000 for c in contours]

		print(contours[1])


		cv2.drawContours(img2,contours, -1, (255,0,0), 3)
		cv2.imshow("images",np.hstack([cv2.resize(img, (0,0), fx=0.5, fy=0.5) ,cv2.resize(img2, (0,0), fx=0.5, fy=0.5) ]))


	cv2.waitKey(0)

def mod_contour(img):
	mask = car_mask(img)

	im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	try:
		c = max(contours, key = cv2.contourArea)
	except:
		print('empty')
		return [],False

	return c,cv2.contourArea(c) > 0






def show_video():

#Check if camera opened successfully 

	cap = cv2.VideoCapture('./sample.mp4')
	if (cap.isOpened()== False): 
	  print("Error opening video stream or file")

	# Read until video is completed
	while(cap.isOpened()):
	  # Capture frame-by-frame
	  ret, frame = cap.read()
	  if ret == True:

		# Display the resulting frame

		frame2 = frame.copy()

		contour,found = mod_contour(frame)
		if found:
			x,y,w,h = cv2.boundingRect(contour)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


		# mask = car_mask(frame)


		# frame = cv2.bitwise_and(frame,frame,mask= mask)

		cv2.imshow("images",frame)


		# Press Q on keyboard to  exit
		if cv2.waitKey(25) & 0xFF == ord('q'):
		  break

	  # Break the loop
	  else: 
		break

	# When everything done, release the video capture object
	cap.release()

	# Closes all the frames
	cv2.destroyAllWindows()



show_image(False)
#show_video()