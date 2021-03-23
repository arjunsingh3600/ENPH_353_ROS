#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np






"""
License Plate detection

1) get blue mask - set threshold + erosion to remove noise
2) establish ROI base on corners of bluemask
3) Apply liberal white mask
4) $$?

"""

class Node:

	def __init__(self):

		self.sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.show_image,queue_size=1)
		self.bridge = CvBridge()
		print("subscribed to node");

		self.getsize = True

		# for recording video


		self.duration = 1 #in s
		self.no_vid_frames = 0
		self.fps =60


	
	def car_mask(self,image):
		""" take raw frame and return a mask of the car. Using color thresholding

		 """
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		image = cv2.medianBlur(image,5)

		# reduce size
		
		#get blues
		lower_blue_bound =(120,109, 0)
		upper_blue_bound =(130,255,255)
		#get whites
		# lower_white_bound =(0,0,86)
		# upper_white_bound =(0,0,203)
		lower_white_bound =(104,0,0)
		upper_white_bound =(127,56,201)


		mask2 = cv2.inRange(image,lower_blue_bound,upper_blue_bound) 

		
		mask = cv2.inRange(image,lower_white_bound,upper_white_bound)


		# remove noise 
		erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(10	, 1	 ))
		dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(6	, 1 ))

		mask = cv2.dilate(mask, dilate_kernel, iterations=1) 
		mask = cv2.erode(mask, erode_kernel, iterations=1) 


		
		erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(10	, 2 ))
		dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(15	, 1 ))
		#mask2  =cv2.dilate(mask2, dilate_kernel, iterations=1)
		mask2  =cv2.dilate(mask2, dilate_kernel, iterations=5) 

		dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(10	, 20 ))
		 
		

		
		
		#mask2  =cv2.erode(mask2, erode_kernel, iterations=1) 
		erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3, 3 ))
		dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(10	, 10))
		

		mask_final = cv2.bitwise_and(mask,mask,mask= mask2)

		
		
		mask_final  = cv2.erode(mask_final, erode_kernel, iterations=1) 
		mask_final = cv2.dilate(mask_final, dilate_kernel, iterations=1) 


	
		#	cv2.imshow("images",np.hstack([cv2.resize(cv2.bitwise_and(image2,image2,mask= mask), (0,0), fx=0.5, fy=0.5) ,cv2.resize(cv2.bitwise_and(image2,image2,mask= mask2), (0,0), fx=0.5, fy=0.5) ]))

		#return mask2
		return mask_final

	# def detect_corners(self,mask):

	# 	# rows and cols are as defined when array is printed in shell
	# 	# cols -> 
	# 	#	0	0	0	0
	# 	#	0	0	0	0
	# 	#	1	1	1	1	row || 
	# 	#	0	0	0	0		\/
		

	# 	diff_rows = np.diff(mask,n=1,axis=0,prepend=0,append=0) # subtract two adjacent rows. subtract the 0 index  row with a row of 0
	# 	max_rows =np.ndarray.flatten( np.argwhere(np.any(diff_rows != 0, axis=1))) # get row no where difference was not 0

	# 	# get the topmost and bottomost  index

	# 	bottom = max(max_rows)
	# 	top = min(max_rows)

		
	# 	diff_cols = np.diff(mask,n=1,axis=1,prepend=0,append=0) # subtract two adjacent cols. subtract 0th col with 0s
	# 	max_cols = np.ndarray.flatten(np.argwhere(np.any(diff_cols != 0, axis=0)))  # get col no where difference was not 0
		
	# 	# get the rightmost and leftmost  index

	# 	right = max(max_cols)
	# 	left = min(max_cols)

		
	# 	#rows,cols = np.where(diff_y !=0 )

	# 	co_ord =[ (left,top),(right,bottom)]

	# 	return co_ord


	# def find_plate(self,frame):

	# 	white_mask,blue_mask =  car_mask(frame)

	# 	roi_corners = detect_corners(blue_mask)

	# 	cv2.rectangle(frame,corners[0],corners[1],(0,255,0),3)


	# 	return frame



	def get_contours(self,frame):

		mask = self.car_mask(frame)

		im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	

		
		try:
			c = max(contours, key = cv2.contourArea)
			area = cv2.contourArea(c)
		except:
			print('empty')
			return [],False

		return c,(area>0)




	def show_image(self,image_message):
		cv_image = self.bridge.imgmsg_to_cv2(image_message,"bgr8")



		contour,found = self.get_contours(cv_image)

		if found:
			x,y,w,h = cv2.boundingRect(contour)
			cv2.rectangle(cv_image,(x,y),(x+w,y+h),(0,255,0),2)


		
		#mask = self.car_mask(cv_image)
		# im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# cv2.drawContours(cv_image,contours,0,(0,0,255),2)

		# cv2.imshow('cv_img', cv2.bitwise_and(cv_image,cv_image,mask = mask))

		cv2.imshow('img',cv_image)
		cv2.waitKey(1)


	def save_video(self,image):

		if self.getsize:
			height,width,layers = image.shape
			self.size = (width,height)

			self.vidwriter = cv2.VideoWriter("~/Videos/sample_footage.avi",cv2.VideoWriter_fourcc(*'DIVX'), self.fps, self.size)
			self.getsize = False

			print("started recording")

		
		if self.no_vid_frames <= self.duration*self.fps:

			self.vidwriter.write(image)
			self.no_vid_frames = self.no_vid_frames +1

			if self.no_vid_frames == self.duration*30:
				self.vidwriter.release()
				self.no_vid_frames = self.no_vid_frames +1
				print("finished recording")








if __name__ == "__main__":

	img = Node()
	rospy.init_node('image_feature', anonymous=True)

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print "Bye Bye lads"
	cv2.destroyAllWindows()

