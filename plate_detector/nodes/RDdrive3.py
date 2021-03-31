#!/usr/bin/env python
import sys
import rospy
import math
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist


# rospy.init_node('topic_publisher', anonymous=True)

# bridge = CvBridge()

# move = Twist()
# move.linear.x = 0.0
# move.angular.z = 0.1

# velPub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
# #image_pub = rospy.Publisher('DebugScreen', Image, queue_size=1)


# imgSub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,callMain)


def getCenterOfRoad(image):
	lineDetected = False
	(rows,cols) = image.shape
	sumOflineCoord = 0
	numOflineCoord = 0
	centerOfLine = 0
	for x in range(0,cols):
  		for y in range(rows-300,rows-200):
  			if(image[y,x]>0):
  				sumOflineCoord = sumOflineCoord + x
  				lineDetected = True
  				numOflineCoord = numOflineCoord + 1
  	if(lineDetected):
  		centerOfLine = sumOflineCoord/(numOflineCoord)

  	return centerOfLine

#searches from left to right to find location of back of car. If back of car is detected in region to xMax, return true. 
#thresh is blue pxl threshold. 
def detectCarBack(image, xMax, end, thresh, rows, cols):
	
	lineDetected = False
	numOflineCoord = 0

	if(xMax<end):
		for x in range(xMax,end):
			for y in range(0,rows):
				if(image[y,x]>0):
					
					numOflineCoord = numOflineCoord + 1

		if(numOflineCoord>thresh):
			return True

	

	return False

#returns x pixel coord of left line + true if detected, false otherwise.
def getLeftLine(image, y, center):
	runningAvg = np.zeros(5)
	(rows,cols) = image.shape
	for x in range(0, cols):
  		runningAvg[(x%5)]=image[rows-y,x]
  		if(np.mean(runningAvg)>240 and center > x):
  			return (x, True)
  	return (cols-1, False)
#returns x pixel coord of right line + true if detected, false otherwise. y =  pixels from bottom of image to search for line.
def getRightLine(image, y, center):
	runningAvg = np.zeros(5)
	(rows,cols) = image.shape
	for x in range(cols-1, -1, -1):
  		runningAvg[(x%5)]=image[rows-y,x]
  		if(np.mean(runningAvg)>240 and center < x):
  			return (x, True)
  	return (cols-1, False)


class navigation():

	def __init__(self):

		self.velPub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
		self.imgSub = rospy.Subscriber('/R1/pi_camera/image_raw',Image, self.callMain)

		self.bridge = CvBridge()




	def callMain(self, data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print("CvBridgeError")

		(rows,cols,channels) = cv_image.shape

		hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

		#grey road thresholds:
		lGrey = (0, 0, 74)
		uGrey = (10, 10, 94)

		#white road thresholds:
		lWhite = (0, 0, 240)
		uWhite = (10, 10, 255)

		lower_blue_bound =(110,180, 90)
		upper_blue_bound =(140,255,255)



		carMask = cv2.inRange(hsv,lower_blue_bound,upper_blue_bound) 

		#get masks for road(grey) and roadlines(white)
		greyMask = cv2.inRange(hsv,lGrey,uGrey)
		whiteMask = cv2.inRange(hsv,lWhite,uWhite)

		center = getCenterOfRoad(greyMask)



		upper = 150
		lower = 100

		(rightLineLo, RLExistLo) = getRightLine(whiteMask, lower, center)
		(rightLineHi, RLExistHi) = getRightLine(whiteMask, upper, center)

		(leftLineLo, LLExistLo) = getLeftLine(whiteMask, lower, center)
		(leftLineHi, LLExistHi) = getLeftLine(whiteMask, upper, center)


		if(RLExistHi):
			cv2.circle(cv_image, (rightLineHi,rows-upper), 20, 255)

		if(RLExistLo):
			cv2.circle(cv_image, (rightLineLo,rows-lower), 20, 255)

		if(LLExistHi):
			cv2.circle(cv_image, (leftLineHi,rows-upper), 20, 255)

		if(LLExistLo):
			cv2.circle(cv_image, (leftLineLo,rows-lower), 20, 255)

		if(center>0):
			cv2.circle(cv_image, (center,rows-250), 10, 255)

		(x,y) = (cols/2, 380)

		cv2.circle(cv_image, (x,y), 10, (255, 255,0))

		font = cv2.FONT_HERSHEY_SIMPLEX
		org = (50, 50) 
		fontScale = 1
		color = (255, 0, 0) 
		thickness = 2
		cv2.putText(cv_image, 'R:'+str(rows)+'  C:'+str(cols), org, font, fontScale, color, thickness, cv2.LINE_AA)
		cv2.putText(cv_image, 'Center:'+str(center), (50, 100), font, fontScale, color, thickness, cv2.LINE_AA)
		cv2.putText(cv_image, 'Grayscale Val RoadFilter:'+str(hsv[y, x, :]), (50, 150), font, fontScale, color, thickness, cv2.LINE_AA)

		#get offset from centerline in pixels: +=to the left -= to the right. this cooresponds to angular.z signs in twist.
		centerOfScreen = cols/2

		isCarCollision = detectCarBack(carMask, cols/3, centerOfScreen,  300, rows, cols)

		p = 1.1
		angle = 0
		straighOffset = 0.0
		#6 worked
		amp = 5.0
		velocity = 0.005*amp

		distanceTocenter = 0
		# and (LLExistLo or LLExistHi)
		if(isCarCollision):
			velocity = 0.005*amp
			#neg 0.2 works too.
			omega = -0.2

		elif((RLExistLo and RLExistHi)):

			distanceTocenter = centerOfScreen-center

			omega = p*distanceTocenter/cols
			if(omega>0.3):
				omega = 0.3
			elif(omega<-0.3):
				omega = -0.3



		elif(LLExistHi or LLExistLo):
			omega = -0.2
			velocity = 0.005*amp

		else:
			#turnLeft
			omega = 0.23
			velocity = 0.005*amp


		move = Twist()
		move.linear.x = velocity
		move.angular.z = omega

		# move = Twist()
		# move.linear.x = 0.0
		# move.angular.z = 0.1

		cv2.putText(cv_image, 'dToc:'+str(distanceTocenter)+' CarColImm:'+str(isCarCollision), (50, 200), font, fontScale, color, thickness, cv2.LINE_AA)
		cv2.putText(cv_image, 'Vel(m/s):'+str(velocity)+' AngVel (rad/s):'+str(omega), (50, 250), font, fontScale, color, thickness, cv2.LINE_AA)




		cv2.imshow("Image window", cv_image)
		# cv2.imshow("Image Cars", carMask)
		cv2.waitKey(1)

		
		self.velPub.publish(move)

		#image_pub.publish(data)


		#rospy.spin()




# move = Twist()
# move.linear.x = 0.0
# move.angular.z = 0.1
# velPub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
#image_pub = rospy.Publisher('DebugScreen', Image, queue_size=1)


# imgSub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,callMain)
# r = rospy.Rate(20)
# while not rospy.is_shutdown():
# 	r.sleep()
if __name__=='__main__':
	rospy.init_node('topic_publisher', anonymous=True)
	my_node = navigation()

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Closing")
		cv2.destroyAllWindows()