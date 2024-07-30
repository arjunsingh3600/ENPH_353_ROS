#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os 
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Twist

from  random import randint




class move_capture():
	

	def __init__(self):
		
		rospy.init_node('movement_tracker', anonymous=True)
		cmd_sub = Subscriber("/R1/cmd_vel",Twist)
		image_sub = Subscriber("/R1/pi_camera/image_raw",Image)

		ats = ApproximateTimeSynchronizer([image_sub, cmd_sub], queue_size=5, slop=0.1,allow_headerless=True)
		ats.registerCallback(self.save_command)


		
		sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.debug_window,queue_size=1)

		self.toggle_flag_counter =0
		self.toggle_flag = True
		self.bridge = CvBridge()
		self.i=0




	def mask_image(self,image):
		hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		#grey road thresholds:
		lGrey = (0, 0, 74)
		uGrey = (10, 10, 94)

		#white road thresholds:
		lWhite = (0, 0, 240)
		uWhite = (10, 10, 255)

		lower_blue_bound =(110,180, 90)
		upper_blue_bound =(140,255,255)

		carMask = cv2.inRange(hsvImg,lower_blue_bound,upper_blue_bound) 
		roadMask = cv2.inRange(hsvImg,lGrey,uGrey)
		rlineMask = cv2.inRange(hsvImg,lWhite,uWhite)

		mask = roadMask | carMask

		return mask


	def debug_window(self,image_message):
		cv_image = self.bridge.imgmsg_to_cv2(image_message,"bgr8")


		mask = self.mask_image(cv_image)
		

		cv_image  =  cv2.bitwise_and(cv_image,cv_image,mask = mask)

		cv_image = cv2.resize(cv_image, (0,0), fx=0.4,fy=0.4)
		cv2.imshow('hello',cv_image)
		cv2.waitKey(1)

	def save_command(self,image,vel):

		direction =""



		found = True

		if vel.linear.x > 0:
			direction = "straight"
			self.toggle_flag_counter = 0
			print(direction)

		elif vel.linear.x ==0:
			if vel.angular.z >0 :
				direction = "left"
				self.toggle_flag_counter = 0
				print(direction)
			elif vel.angular.z <0 :
				direction = "right"
				self.toggle_flag_counter = 0
				print(direction)
			elif vel.angular.z==0: 
				self.toggle_flag_counter = self.toggle_flag_counter + 1
				found = False	

			else:
				found = False 
				self.toggle_flag_counter = 0
				
		else:
			found = False

		if self.toggle_flag_counter ==3:
			self.toggle_flag = not self.toggle_flag
			print("recording {}".format(self.toggle_flag))
			self.toggle_flag_counter =0

		if found and self.toggle_flag:

			image = self.bridge.imgmsg_to_cv2(image,"bgr8")

			# only consider mask with road + cars
			image = self.mask_image(image)
			# resize
			image = cv2.resize(image, (0,0), fx=0.4,fy=0.4)

			# font = cv2.FONT_HERSHEY_SIMPLEX
			# text = str(state)
			# cv2.putText(image,text,(50,50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

			
			
			name  = str(os.path.dirname(os.path.realpath(__file__))) + '/train_data/' + str(self.i) + " __{}__".format(randint(1,99999)) + direction +'.png'
			cv2.imwrite( name, image)
			
			self.i = self.i +1




if __name__ == "__main__":

	# start loop

	
	while True:
		print('Press k three times to toggle recording. to begin, type start')
		start_flag = raw_input()

		if start_flag == "start":
			break

	img = move_capture()
	

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print "Bye Bye lads"
	cv2.destroyAllWindows()



