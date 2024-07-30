#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import os 
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import tensorflow as tf


sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)


class nnDriver:
	"""
		NN driver based on the following paper -
		https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
		
		Note : This impementation was not used in the final competition. A simpler PID control version with 
		obstactle detection was opted for instead. While the NNdriver was an interesting proof of concept,
		It did not out perform the classic PID control which was easier to integrate and more interpreatable.
	"""

	def __init__(self,linear_vel = 0.09 ,angular_vel =0.18):


		#load model
		path =os.path.dirname(os.path.realpath(__file__))
		
		
		with open(path +'/NN_driving/Model/driving_model_config.json') as json_file:
			json_config = json_file.read()
			self.model = models.model_from_json(json_config)
		self.model.load_weights(path  + '/NN_driving/Model/weights_only_driving.h5')
		print("in driving cnn")



	
		rospy.init_node('topic_publisher', anonymous=True)
		self.bridge = CvBridge()
		self.velPub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

		sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.image_callback,queue_size=1)

		self.move = Twist()
		self.angular_vel =angular_vel
		self.linear_vel = linear_vel



	def generate_vel(self,cv_image):


		
		# process
		mask = self.process(cv_image)

		cv2.imshow('a',cv2.resize(cv_image,(0,0),fx=0.4,fy=0.4))
		cv2.waitKey(1)

		# send to vel command
		return self.send_vel(mask)
		

	def process(self,image):

		
		


		#apply mask
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
		

		mask = roadMask | carMask
		
		#resize

		mask = cv2.resize(mask, (205,115))
		#normalize
		mask = np.asarray(mask)


		mean_px = mask.mean().astype(np.float32)
		std_px = mask.std().astype(np.float32)
		mask = (mask - mean_px)/(std_px)



		return mask


	def send_vel(self,image):
		#predict velocity

	
		with graph.as_default():
			set_session(sess)
			prediction = self.model.predict(np.asarray([image[:,:,np.newaxis]]))
		prediction = np.argmax(prediction)



		# mutate self.move
		if prediction == 0: #left
			self.move.linear.x = 0.
			self.move.angular.z = self.angular_vel
		elif prediction ==2:
			self.move.linear.x = 0.
			self.move.angular.z = self.angular_vel *-1
		elif prediction ==1:
			self.move.linear.x = self.linear_vel
			self.move.linear.z = 0


		return self.move

	

if __name__ == "__main__":

	driver = nnDriver(0.17,0.34)



	try:
		rospy.spin()
	except KeyboardInterrupt:
		print "Bye Bye lads"
	cv2.destroyAllWindows()
