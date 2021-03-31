#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os 
from scipy.spatial import distance as dist




class Node:

	def __init__(self):


		rospy.init_node('image_feature', anonymous=True)

		self.sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.show_image,queue_size=1)
		self.bridge = CvBridge()
		print("subscribed to node");

		self.getsize = True

		self.rate = rospy.Rate(1);

		self.img =0;


		self.prev_img =  np.zeros((100,100,3));

		


	
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
		mask2  =cv2.dilate(mask2, dilate_kernel, iterations=9) 

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


	def process_plate(self,plate):


		# apply mask  isolate plate		upper_grey = (152,146,90)

		# lower_grey = (0,0,0)
		# upper_grey = (360,255,255)

		lower_grey = (0,3,0)
		upper_grey = (190,150,100)
		mask_grey = cv2.inRange(plate,lower_grey,upper_grey) 
		#mask_grey = cv2.bitwise_not(mask_grey)


		mask_grey = cv2.dilate(mask_grey, cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(2	, 2)), iterations=1) 



		return cv2.bitwise_and(plate,plate,mask = mask_grey)

			# points =  np.array([ [x,y],[x+w,y],[x,y+h],[x+w,y+h] ],np.float32)
			# dst = np.array([ [0,0],[w-1,0],[0,h-1],[w-1,h-1] ],np.float32)
			# M = cv2.getPerspectiveTransform(points, dst)
			# plate = cv2.warpPerspective(cv_image, M, (w, h))
		



	def order_points(self,pts):
		# sort the points based on their x-coordinates
		xSorted = pts[np.argsort(pts[:, 0]), :]
		# grab the left-most and right-most points from the sorted
		# x-roodinate points
		leftMost = xSorted[:2, :]
		rightMost = xSorted[2:, :]
		# now, sort the left-most coordinates according to their
		# y-coordinates so we can grab the top-left and bottom-left
		# points, respectively
		leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
		(tl, bl) = leftMost
		# now that we have the top-left coordinate, use it as an
		# anchor to calculate the Euclidean distance between the
		# top-left and right-most points; by the Pythagorean
		# theorem, the point with the largest distance will be
		# our bottom-right point
		D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
		(br, tr) = rightMost[np.argsort(D)[::-1], :]
		# return the coordinates in top-left, top-right,
		# bottom-right, and bottom-left order
		return np.array([tl, tr, br, bl], dtype="float32")


	def show_image(self,image_message):
		cv_image = self.bridge.imgmsg_to_cv2(image_message,"bgr8")


		
		

		contour,found = self.get_contours(cv_image)
		plate = np.zeros((1,1,3), np.uint8)
		
		if found:
			#x,y,w,h = cv2.boundingRect(contour)

			rect = cv2.minAreaRect(contour)
			

			box = cv2.boxPoints(rect)
			box = np.int0(box)
			width = int(rect[1][0])
			height = int(rect[1][1])


			#cv2.imshow('warped',warped)
			#cv2.drawContours(cv_image,[box],0,(0,0,255),2)

		
			

			if(width*height > 1000 and width >0):
				#top left, top right, bottom left, bottom right
				font = cv2.FONT_HERSHEY_SIMPLEX
				text = "width {} height {} Area {}".format(width,height,width*height)
				cv2.putText(cv_image,text,(50,50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
				# plate = cv_image[y:y+h, x:x+w]

				# if  not (self.prev_img == plate):
				# 	name  = str(os.path.dirname(os.path.realpath(__file__))) + '/screen_shots/' + str(self.img)+'.jpeg'
				# 	cv2.imwrite( name, plate)
				# 	self.rate.sleep()


				src_pts = self.order_points(box).astype("float32")
				dst_pts = np.array([[0, 0],
								[width-1, 0],
								[width-1, height-1],
								[0, height-1]
							], dtype="float32")
				# dst_pts = np.array([[width-1, 0],
				# 			[width-1, height-1],
				# 			[0, height-1],
				# 		[0, 0]	
				# 			], dtype="float32")
				M = cv2.getPerspectiveTransform(src_pts, dst_pts)
				plate = cv2.warpPerspective(cv_image, M, (width, height))


				# self.prev_img = plate
				# self.img = self.img +1;


	
		

				

				



				#plate = self.process_plate(plate)

				#cv2.rectangle(cv_image,(x,y),(x+w,y+h),(0,255,0),2)

				cv2.imshow("plate",plate)
			else:
				cv2.imshow('hello',cv_image)

		else:
			cv2.imshow('hello',cv_image)



		# scaled = cv2.resize(cv_image, (0,0), fx=0.5, fy=0.5)
		# height,width,chanel = scaled.shape


		# padding_top = int(np.floor((height-plate.shape[0])/2))
		# padding_top_2 = int(np.ceil((height-plate.shape[0])/2)) +1
		# padding_right = int(np.ceil((width-plate.shape[1])/2)) +1
		# padding_right_2 = int(np.floor((width-plate.shape[1])/2)) 
		
		# plate = cv2.copyMakeBorder( plate, padding_top , padding_top_2 , padding_right, padding_right_2, cv2.BORDER_CONSTANT,0)

		
		
		#print(plate.shape)
		# cv2.imshow("images",np.hstack([ scaled ,plate]))

		
	

		#cv2.imshow("images",cv_image)
		
		#mask = self.car_mask(cv_image)
		# im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# cv2.drawContours(cv_image,contours,0,(0,0,255),2)

		# cv2.imshow('cv_img', cv2.bitwise_and(cv_image,cv_image,mask = mask))

		
		cv2.waitKey(1)






if __name__ == "__main__":

	img = Node()
	

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print "Bye Bye lads"
	cv2.destroyAllWindows()























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

