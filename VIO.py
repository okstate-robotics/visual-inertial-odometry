import pandas as pd
import subprocess
import rospy
import numpy as np
# ROS Image message
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PointStamped
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance as Dist
#from array import *
from sklearn.metrics.pairwise import euclidean_distances
import threading
import time

class Hansel:

    # Instantiate CvBridge
    #bridge = CvBridge()
    #df = pd.DataFrame(columns=['pointNum','x', 'y', 'patch','frameNum'])
    def __init__(self):
        self.frame = 0
        self.img = 0
        self.imgr =0
        self.image_is_comming = False
        self.IMU_is_comming = False
        self.GT_is_comming = False
        self.bridge = CvBridge()
        self.df = pd.DataFrame(columns=['pointNum','x', 'y', 'patch','rotation','translation','frameNum','groundtruth'])

        self.RTdf = pd.DataFrame(columns=['rotation','translation','frameNum','groundtruth'])
        self.startTime = 0
        self.landmarks = pd.DataFrame(columns=['pointNum','x', 'y','frameNum'])
        self.perceptionDF = pd.DataFrame(columns=['frameNum','difx', 'dify','imu angular velocity','imu linear velocity','groundtruth'])
        self.RT_IMU_GT_DF = pd.DataFrame(
            columns=['frameNum', 'rotation','translation', 'imu angular velocity', 'imu linear velocity', 'groundtruth'])

        self.patchsize = 4
        self.cfpatchs = np.array(16)
        self.lfpatchs = np.array(16)
        self.groundtruth = []
        self.groundtruth_for_frame =[]
        self.IMU_INIT = []
        self.IMU = []
        self.IMU_angular = []
        self.IMU_linear=[]
        self.camera_matrix_right = np.array([[457.587, 0.0, 379.999], [0.0, 456.134, 255.238], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.camera_matrix_left = np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375],[ 0.0, 0.0, 1.0]], dtype=np.float32)
        self.distortion_coefficient= np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.], dtype=np.float32)


        self.rate = rospy.Rate(rospy.get_param('~hz', 4))
        #self.rate =rospy.Rate(10)
        rospy.Subscriber("/cam0/image_raw", Image, self.image_callback)
        rospy.Subscriber("/cam1/image_raw", Image, self.image_callbackr)

        rospy.Subscriber("/leica/position", PointStamped, self.ground_truth_callback)
        rospy.Subscriber("/imu1", Imu, self.IMU_callback)
        #rospy.Subscriber("/cam0/image_raw", Imu, self.image_callback)
    def savetoCSV(self):
        self.RT_IMU_GT_DF.to_csv('test_set11Hz', encoding='utf-8', index=False)
        print("csv saved")
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#************************************************************************************************************
    def create_dataframe(self,currentFrameNumber,frameTocheck):
        cFrame = self.row_extractor_by_frameNumber(self.RTdf, 'frameNum', currentFrameNumber)
        current_Rotation = cFrame[['rotation']].to_numpy()
        current_translation = cFrame[['translation']].to_numpy()
        currentGT = cFrame[['groundtruth']].to_numpy()
        lFrame = self.row_extractor_by_frameNumber(self.RTdf, 'frameNum', frameTocheck)
        last_Rotation = lFrame[['rotation']].to_numpy()
        lastt_translation = lFrame[['translation']].to_numpy()
        lastGT = lFrame[['groundtruth']].to_numpy()



        ######ground truth difference
        GT_diff = [lastGT[0][0][0] - currentGT[0][0][0], lastGT[0][0][1] - currentGT[0][0][1],
                   lastGT[0][0][2] - currentGT[0][0][2]]
        if GT_diff[0] == 0 and GT_diff[1] == 0 and GT_diff[2] == 0 and self.frame > 3:
            lastFrame = self.RTdf[self.RTdf['frameNum'] == frameTocheck - 2]
            lastGT = lastFrame[['groundtruth']].to_numpy()

            GT_diff = [lastGT[0][0][0] - currentGT[0][0][0], lastGT[0][0][1] - currentGT[0][0][1],
                       lastGT[0][0][2] - currentGT[0][0][2]]

        imu_angular = [self.IMU[0].angular_velocity.x, self.IMU[0].angular_velocity.y, self.IMU[0].angular_velocity.z]
        imu_linear = [self.IMU[0].linear_acceleration.x, self.IMU[0].linear_acceleration.y,
                      self.IMU[0].linear_acceleration.z]
        self.RT_IMU_GT_DF.loc[len(self.RT_IMU_GT_DF)] = [currentFrameNumber, current_Rotation,current_translation, imu_angular,
                                                         imu_linear, GT_diff]
    #*********************************************************************************************************************
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def find_same_patches_in_both_camera(self):
        pass

##########################################find distance betweeen 2 frames
    def distance(self, vec1, vec2):
        dist = Dist.euclidean(vec1, vec2)
        dist = np.sum(dist)
        return(dist)
    ########################################find top Features
    ################################get coordinate of pointNum
    def coordinateOf(self,frameNumber,pointNumber):
        framerows = self.df[self.df['frameNum'] == frameNumber]
        exactRow = framerows[framerows['pointNum'] == pointNumber]
        X = int(exactRow['x'])
        Y = int(exactRow['y'])
        return [X,Y]
    def row_extractor_by_frameNumber(self,df,name_of_df_column,number_of_frame):
        return df[df[name_of_df_column] == number_of_frame]
#############################################get pandas rows give be similar pairs
    def similarPatchsIn(self,currentFrameNumber,frameTocheck,df):
       pass
#######################################show
    def show_Method(self,image1,image2):
        cv2.imshow('image2',image2)
        cv2.imshow('image1',image1)
        cv2.waitKey(1)
    def ground_truth_callback(self,msg):
        self.GT_is_comming = True

        self.groundtruth = [msg.point.x,msg.point.y,msg.point.z]

    def IMU_callback(self,msg):
        self.IMU_is_comming = True
        self.IMU_INIT = [msg]


    def findDecomposedEssentialMatrix(self, p1, p2):
        # fundamental matrix and inliers
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_LMEDS, 1, 0.999)
        #F, mask = cv.findFundamentalMat(p1, p2, cv.FM_RANSAC, 1.0, 0.999)
        mask = mask.astype(bool).flatten()
        E = np.dot(self.camera_matrix_left.T, np.dot(F, self.camera_matrix_right))

        _, R, t, _ = cv2.recoverPose(E, p1[mask], p2[mask], self.camera_matrix_right)

        return R, t

    def image_callback(self,msg):

        self.image_is_comming = True
        try:
            # Convert your ROS Image message to OpenCV2
            self.img = self.bridge.imgmsg_to_cv2(msg, "mono8")


        except CvBridgeError, e:
            print(e)

    def image_callbackr(self, msg):

        self.image_is_comming = True
        try:
            # Convert your ROS Image message to OpenCV2
            self.imgr = self.bridge.imgmsg_to_cv2(msg, "mono8")


        except CvBridgeError, e:
            print(e)
    def main_loop(self):
        print("inside main loop")


        while not rospy.is_shutdown():


            self.rate.sleep()

            #rospy.Subscriber(image_topic, Image, self.image_callback)
            if self.image_is_comming == True and self.IMU_is_comming == True and self.GT_is_comming == True:
                corners = cv2.goodFeaturesToTrack(self.img,20,0.01,10)
                cornersr = cv2.goodFeaturesToTrack(self.imgr, 20, 0.01, 10)
                self.IMU = self.IMU_INIT
                self.groundtruth_for_frame =self.groundtruth

                corners = np.int0(corners)
                cornersr = np.int0(cornersr)





                ############################new approach
                Rotation ,translation = self.findDecomposedEssentialMatrix(corners, cornersr)

                self.RTdf.loc[len(self.RTdf)] = [Rotation, translation, self.frame,self.groundtruth_for_frame]




                if (self.frame>0):
                    self.create_dataframe(self.frame,self.frame-1)
                    print(self.RT_IMU_GT_DF.loc[len(self.RT_IMU_GT_DF) - 1])




     ################################## Landmark Matching

                self.frame += 1














rospy.init_node('image_listener')
#bag_filename = 'MH_03_medium.bag'
#player_proc = subprocess.Popen(['rosbag', 'play', bag_filename])
hansel = Hansel()

hansel.main_loop()
hansel.savetoCSV()
