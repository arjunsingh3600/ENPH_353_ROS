#!/usr/bin/python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
from scipy.spatial import distance as dist

from CNN_license_plate.predict import PlateCNN

from scipy import stats
from pyzbar.pyzbar import decode

from random import randint

from std_msgs.msg import String


# Initialise Node and start rospy.spin ( or equivalent).
# Node will subscribe to the image feed and try to extract license plates from frames
# CNN_license_plate.predict is the CNN to read plates. File structure must be followed for successsful import

class Node:

    def __init__(self):

        self.cnn = PlateCNN('./plate_model.h5')  # path does not matter. NN path hardcoded in predict.py
        rospy.init_node('image_feature', anonymous=True)

        self.sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image,
                                    self.show_image, queue_size=1)
        self.score_pub = rospy.Publisher('/license_plate', String,
                queue_size=1)

        self.bridge = CvBridge()

        self.recorded_spots = []  # readings sent to score tracker
        self.recorded_plates = []  # all recorded plate readings

        self.published = []

        self.counter = 0

        # turn QR on to collect qr + plate data to

        self.QR = False

        if self.QR:

            self.cnn = PlateCNN('dummy')

            self.plate_data = {}

            self.image_buffer = []
            self.plate_buffer = []
            self.last_plate = ''

    def car_mask(self, image):
        """ take raw frame and return a mask of the car. Using color thresholding

........ """

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.medianBlur(image, 5)

        # reduce size

        # get blues

        lower_blue_bound = (120, 109, 0)
        upper_blue_bound = (130, 255, 255)

        # get whites

        lower_white_bound = (104, 0, 0)
        upper_white_bound = (127, 56, 201)

        mask2 = cv2.inRange(image, lower_blue_bound, upper_blue_bound)

        mask = cv2.inRange(image, lower_white_bound, upper_white_bound)

        # remove noise

        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                ksize=(10, 1))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                ksize=(6, 1))

        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        mask = cv2.erode(mask, erode_kernel, iterations=1)

        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                ksize=(10, 2))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                ksize=(15, 1))
        mask2 = cv2.dilate(mask2, dilate_kernel, iterations=9)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                ksize=(10, 20))

        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                ksize=(3, 3))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                ksize=(10, 10))

        mask_final = cv2.bitwise_and(mask, mask, mask=mask2)

        mask_final = cv2.erode(mask_final, erode_kernel, iterations=1)
        mask_final = cv2.dilate(mask_final, dilate_kernel, iterations=1)

        return mask_final

    def get_contours(self, frame):

        mask = self.car_mask(frame)

        (im2, contours, hierarchy) = cv2.findContours(mask,
                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        try:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
        except:

            return ([], False)

        return (c, area > 0)

    def order_points(self, pts):

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

        D = dist.cdist(tl[np.newaxis], rightMost, 'euclidean')[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order

        return np.array([tl, tr, br, bl], dtype='float32')

    def get_plate(self, contour, image):
        rect = cv2.minAreaRect(contour)

        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width = int(np.max(rect[1]))
        height = int(np.min(rect[1]))

        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

        if width * height > 3200 and width > 0:

            src_pts = self.order_points(box).astype('float32')
            dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1,
                               height - 1], [0, height - 1]],
                               dtype='float32')

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            plate = cv2.warpPerspective(image, M, (width, height))

            return (True, plate)
        else:
            return (False, None)

    def publish_readings(self):

        if not len(self.recorded_plates) == 0 \
            and not len(self.recorded_spots) == 0:

            plate_val = stats.mode(self.recorded_plates)[0]
            spot_val = stats.mode(self.recorded_spots)[0]

            plate_val = plate_val[0]
            spot_val = spot_val[0]

            if plate_val[0].isdigit() or plate_val[1].isdigit() \
                or not plate_val[2].isdigit() \
                or not plate_val[3].isdigit():
                self.recorded_plates = []
                self.recorded_spots = []
                return

            print str('TeamRed,multi21,{},{}').format(spot_val[1],
                    plate_val)
            self.score_pub.publish(str('TeamRed,multi21,{},{}'
                                   ).format(spot_val[1], plate_val))

            self.recorded_plates = []
            self.recorded_spots = []

            if spot_val not in self.published:
                self.published.append(spot_val)
                self.score_pub.publish(str('TeamRed,multi21,-1,XXXX'))
                print 'stopped timer'

    def get_parking(self, contour, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        (x, y, w, h) = cv2.boundingRect(contour)

        lower_val = np.array([0, 0, 0])
        upper_val = np.array([360, 30, 30])

        mask = cv2.inRange(hsv, lower_val, upper_val)

        if w > 2 * h:

            image = mask[y - 2 * h:y + h, x:x + w]

            # count no of digits

            vsum = np.sum(image, axis=0)

            vsum[vsum > 0] = 1

            vdif = np.diff(vsum.astype(float))

            LL = np.where(vdif > 0)[0]
            RL = np.where(vdif < 0)[0]

            if LL.shape[0] is not 2 or RL.shape[0] is not 2:
                return

            image = cv2.resize(image, (100, 100))
            parking = self.cnn.predict_parking(image)

            self.recorded_spots.append(parking)

    def show_image(self, image_message):
        """
........call back function.  prints plate digits if detected
........"""

        cv_image = self.bridge.imgmsg_to_cv2(image_message, 'bgr8')

        (contour, found) = self.get_contours(cv_image)
        plate = np.zeros((1, 1, 3), np.uint8)

        if found:

        # if any plate  was found

            (is_big, plate) = self.get_plate(contour, cv_image)
            if self.QR:
                self.get_qr(cv_image)

            if is_big:

            # if plate was big enough

                self.get_parking(contour, cv_image)

                if self.QR:
                    (success, digits) = self.cnn.pre_process(plate)

                    if success:
                        self.image_buffer.append(digits)

                digits = self.cnn.predict(plate)

                if not digits == '':

                    self.recorded_plates.append(digits)

                    cv2.imshow('Success', plate)
                    cv2.waitKey(1)
            else:

                self.publish_readings()

            if self.QR:
                if len(self.image_buffer) > 0 \
                    and len(self.plate_buffer) > 0:
                    self.save_data()

# code for QR data collection

    def get_qr(self, cv_image):

        plate = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        barcodes = decode(plate)

        if len(barcodes) > 0:

            plate_string = barcodes[0].data.decode()

            if plate_string is not self.last_plate:
                self.plate_buffer = []
                self.last_plate = plate_string

            self.plate_buffer.append(plate_string)
            print 'qr found'

    def save_data(self):

        # for QR

        digits = self.image_buffer[-1]
        plate_id = self.plate_buffer[-1]

        prefix = plate_id[:3]
        print prefix
        index = 0
        for char in plate_id[3:]:

            name = str(os.path.dirname(os.path.realpath(__file__))) \
                + '/screen_shots/' + prefix + str(randint(10000,
                    99999)) + '_' + char + '.png'
            cv2.imwrite(name, digits[index] * 255)
            index = index + 1

        self.image_buffer = []
        self.plate_buffer = []
        rospy.sleep(1)


if __name__ == '__main__':

    img = Node()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print 'Exiting'
    cv2.destroyAllWindows()

    # ....return cv2.bitwise_and(plate,plate,mask = mask_grey)

    # ........# points =  np.array([ [x,y],[x+w,y],[x,y+h],[x+w,y+h] ],np.float32)
    # ........# dst = np.array([ [0,0],[w-1,0],[0,h-1],[w-1,h-1] ],np.float32)
    # ........# M = cv2.getPerspectiveTransform(points, dst)
    # ........# plate = cv2.warpPerspective(cv_image, M, (w, h))
#!/usr/bin/python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
from scipy.spatial import distance as dist

from CNN_license_plate.predict import PlateCNN

from scipy import stats
from pyzbar.pyzbar import decode

from random import randint

from std_msgs.msg import String


#
#
# Initialise Node and start rospy.spin ( or equivalent).
# Node will subscribe to the image feed and try to extract license plates from frames
# CNN_license_plate.predict is the CNN to read plates. File structure must be followed for successsful import

class Node:

    def __init__(self):

        self.cnn = PlateCNN('./plate_model.h5')  # path does not matter. NN path hardcoded in predict.py
        rospy.init_node('image_feature', anonymous=True)

        self.sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image,
                                    self.show_image, queue_size=1)
        self.score_pub = rospy.Publisher('/license_plate', String,
                queue_size=1)

        self.bridge = CvBridge()

        self.recorded_spots = []  # readings sent to score tracker
        self.recorded_plates = []  # all recorded plate readings

        self.published = []

        self.counter = 0

        # turn QR on to collect qr + plate data to

        self.QR = False

        if self.QR:

            self.cnn = PlateCNN('dummy')

            self.plate_data = {}

            self.image_buffer = []
            self.plate_buffer = []
            self.last_plate = ''

    def car_mask(self, image):
        """ take raw frame and return a mask of the car. Using color thresholding

........ """

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.medianBlur(image, 5)

        # reduce size

        # get blues

        lower_blue_bound = (120, 109, 0)
        upper_blue_bound = (130, 255, 255)

        # get whites

        lower_white_bound = (104, 0, 0)
        upper_white_bound = (127, 56, 201)

        mask2 = cv2.inRange(image, lower_blue_bound, upper_blue_bound)

        mask = cv2.inRange(image, lower_white_bound, upper_white_bound)

        # remove noise

        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                ksize=(10, 1))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                ksize=(6, 1))

        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        mask = cv2.erode(mask, erode_kernel, iterations=1)

        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                ksize=(10, 2))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                ksize=(15, 1))
        mask2 = cv2.dilate(mask2, dilate_kernel, iterations=9)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                ksize=(10, 20))

        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                ksize=(3, 3))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                ksize=(10, 10))

        mask_final = cv2.bitwise_and(mask, mask, mask=mask2)

        mask_final = cv2.erode(mask_final, erode_kernel, iterations=1)
        mask_final = cv2.dilate(mask_final, dilate_kernel, iterations=1)

        return mask_final

    def get_contours(self, frame):

        mask = self.car_mask(frame)

        (im2, contours, hierarchy) = cv2.findContours(mask,
                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        try:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
        except:

            return ([], False)

        return (c, area > 0)

    def order_points(self, pts):

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

        D = dist.cdist(tl[np.newaxis], rightMost, 'euclidean')[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order

        return np.array([tl, tr, br, bl], dtype='float32')

    def get_plate(self, contour, image):
        rect = cv2.minAreaRect(contour)

        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width = int(np.max(rect[1]))
        height = int(np.min(rect[1]))

        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

        if width * height > 3200 and width > 0:

            src_pts = self.order_points(box).astype('float32')
            dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1,
                               height - 1], [0, height - 1]],
                               dtype='float32')

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            plate = cv2.warpPerspective(image, M, (width, height))

            return (True, plate)
        else:
            return (False, None)

    def publish_readings(self):

        if not len(self.recorded_plates) == 0 \
            and not len(self.recorded_spots) == 0:

            plate_val = stats.mode(self.recorded_plates)[0]
            spot_val = stats.mode(self.recorded_spots)[0]

            plate_val = plate_val[0]
            spot_val = spot_val[0]

            if plate_val[0].isdigit() or plate_val[1].isdigit() \
                or not plate_val[2].isdigit() \
                or not plate_val[3].isdigit():
                self.recorded_plates = []
                self.recorded_spots = []
                return

            print str('TeamRed,multi21,{},{}').format(spot_val[1],
                    plate_val)
            self.score_pub.publish(str('TeamRed,multi21,{},{}'
                                   ).format(spot_val[1], plate_val))

            self.recorded_plates = []
            self.recorded_spots = []

            if spot_val not in self.published:
                self.published.append(spot_val)
                self.score_pub.publish(str('TeamRed,multi21,-1,XXXX'))
                print 'stopped timer'

    def get_parking(self, contour, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        (x, y, w, h) = cv2.boundingRect(contour)

        lower_val = np.array([0, 0, 0])
        upper_val = np.array([360, 30, 30])

        mask = cv2.inRange(hsv, lower_val, upper_val)

        if w > 2 * h:

            image = mask[y - 2 * h:y + h, x:x + w]

            # count no of digits

            vsum = np.sum(image, axis=0)

            vsum[vsum > 0] = 1

            vdif = np.diff(vsum.astype(float))

            LL = np.where(vdif > 0)[0]
            RL = np.where(vdif < 0)[0]

            if LL.shape[0] is not 2 or RL.shape[0] is not 2:
                return

            image = cv2.resize(image, (100, 100))
            parking = self.cnn.predict_parking(image)

            self.recorded_spots.append(parking)

    def show_image(self, image_message):
        """
........call back function.  prints plate digits if detected
........"""

        cv_image = self.bridge.imgmsg_to_cv2(image_message, 'bgr8')

        (contour, found) = self.get_contours(cv_image)
        plate = np.zeros((1, 1, 3), np.uint8)

        if found:

        # if any plate  was found

            (is_big, plate) = self.get_plate(contour, cv_image)
            if self.QR:
                self.get_qr(cv_image)

            if is_big:

            # if plate was big enough

                self.get_parking(contour, cv_image)

                if self.QR:
                    (success, digits) = self.cnn.pre_process(plate)

                    if success:
                        self.image_buffer.append(digits)

                digits = self.cnn.predict(plate)

                if not digits == '':

                    self.recorded_plates.append(digits)

                    cv2.imshow('Success', plate)
                    cv2.waitKey(1)
            else:

                self.publish_readings()

            if self.QR:
                if len(self.image_buffer) > 0 \
                    and len(self.plate_buffer) > 0:
                    self.save_data()

# code for QR data collection

    def get_qr(self, cv_image):

        plate = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        barcodes = decode(plate)

        if len(barcodes) > 0:

            plate_string = barcodes[0].data.decode()

            if plate_string is not self.last_plate:
                self.plate_buffer = []
                self.last_plate = plate_string

            self.plate_buffer.append(plate_string)
            print 'qr found'

    def save_data(self):

        # for QR

        digits = self.image_buffer[-1]
        plate_id = self.plate_buffer[-1]

        prefix = plate_id[:3]
        print prefix
        index = 0
        for char in plate_id[3:]:

            name = str(os.path.dirname(os.path.realpath(__file__))) \
                + '/screen_shots/' + prefix + str(randint(10000,
                    99999)) + '_' + char + '.png'
            cv2.imwrite(name, digits[index] * 255)
            index = index + 1

        self.image_buffer = []
        self.plate_buffer = []
        rospy.sleep(1)


if __name__ == '__main__':

    img = Node()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print 'Exiting'
    cv2.destroyAllWindows()

