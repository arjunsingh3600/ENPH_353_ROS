#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

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


		




	def show_image(self,image_message):
		cv_image = self.bridge.imgmsg_to_cv2(image_message,"bgr8")



		cv2.imshow('cv_img', cv_image)
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

