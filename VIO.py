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
        self.startTime = 0
        self.landmarks = pd.DataFrame(columns=['pointNum','x', 'y','frameNum'])
        self.perceptionDF = pd.DataFrame(columns=['frameNum','difx', 'dify','imu angular velocity','imu linear velocity','groundtruth'])
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
        rospy.Subscriber("/imu0", Imu, self.IMU_callback)
        #rospy.Subscriber("/cam0/image_raw", Imu, self.image_callback)
    def savetoCSV(self):
        self.df.to_csv('file_name2', encoding='utf-8', index=False)

        self.perceptionDF.to_csv('perception', encoding='utf-8', index=False)
        print("csv saved")

    def find_same_patches_in_both_camera(self):
        pass

##########################################find distance betweeen 2 frames
    def distance(self, vec1, vec2):

        dist = Dist.euclidean(vec1, vec2)
        dist = np.sum(dist)
        #print(dist)
        return(dist)
    ########################################find top Features
    ################################get coordinate of pointNum
    def coordinateOf(self,frameNumber,pointNumber):
        framerows = self.df[self.df['frameNum'] == frameNumber]
        exactRow = framerows[framerows['pointNum'] == pointNumber]
        X = int(exactRow['x'])
        Y = int(exactRow['y'])
        #print ("x =" ),
        #print(X),
        #print("y =" ),
        #print(Y)
        return [X,Y]
    def row_extractor_by_frameNumber(self,name_of_df_column,number_of_frame):
        return self.df[self.df[name_of_df_column] == number_of_frame]
#############################################get pandas rows give be similar pairs
    def similarPatchsIn(self,currentFrameNumber,frameTocheck,df):
       # print(frameTocheck)

        #currentFrame1 = self.df[self.df['frameNum'] == currentFrameNumber]
        currentFrame = self.row_extractor_by_frameNumber('frameNum',currentFrameNumber)
        #print (currentFrame)
        #print (currentFrame1)
        #lastFrame =self.df[self.df['frameNum'] == frameTocheck]
        lastFrame = self.row_extractor_by_frameNumber('frameNum',frameTocheck)
        currentPatchs = currentFrame[['patch']].to_numpy()
        currentGT = currentFrame[['groundtruth']].to_numpy()
        #print ("gt = "),
        #print(currentGT[0][0][0])

        lastPatchs = lastFrame[['patch']].to_numpy()
        lastGT = lastFrame[['groundtruth']].to_numpy()

        GT_diff = [lastGT[0][0][0] -currentGT[0][0][0],lastGT[0][0][1] -currentGT[0][0][1],lastGT[0][0][2] -currentGT[0][0][2]]
        if GT_diff[0]==0 and GT_diff[1]==0 and GT_diff[2]==0 and self.frame >3:
            lastFrame =self.df[self.df['frameNum'] == frameTocheck-2]
            lastGT = lastFrame[['groundtruth']].to_numpy()
            GT_diff = [lastGT[0][0][0] -currentGT[0][0][0],lastGT[0][0][1] -currentGT[0][0][1],lastGT[0][0][2] -currentGT[0][0][2]]
        #print ("gt = "),
        #print(GT_diff)
        ik=-1
        jk=-1
        #####I put distances in a list dist = [current frame patch number, last frame patch number, distance between them]
        tempdist = [1000,1000,1000]
        dist = np.array(tempdist)
        for i in currentPatchs:
            ik +=1 #currentframe
            jk = -1
            for j in lastPatchs:
                jk +=1
                temprow = [ik,jk,self.distance(i,j)]
                dist = np.vstack([dist,temprow])

        # it sort the point by their distances
        dist  = dist[dist[:,2].argsort()]
        mostMatchs = dist[:5,:]
        ########################IMU _ Perception _ Ground truth data saving
        #point number 1
        imu_angular = [self.IMU[0].angular_velocity.x,self.IMU[0].angular_velocity.y,self.IMU[0].angular_velocity.z]
        imu_linear = [self.IMU[0].linear_acceleration.x,self.IMU[0].linear_acceleration.y,self.IMU[0].linear_acceleration.z]
        #self.coordinateOf(currentFrameNumber-1,mostMatchs[0,1])
        #self.coordinateOf(currentFrameNumber,mostMatchs[0,0])
        #print(self.coordinateOf(currentFrameNumber,mostMatchs[0,0]))
        p1=[]
        p2=[]
        for i in range(5):
            [x1, y1] = self.coordinateOf(currentFrameNumber - 1, mostMatchs[i, 1])
            [x2, y2] = self.coordinateOf(currentFrameNumber, mostMatchs[i, 0])
            self.perceptionDF.loc[len(self.perceptionDF)] = [currentFrameNumber, x2 - x1, y2 - y1, imu_angular,
                                                             imu_linear, GT_diff]

        ####################put data in dataframe
       # self.perceptionDF.loc[len(self.perceptionDF)] = [self.coordinateOf(currentFrameNumber-1,mostMatchs[0,1]),]




#######################################show
    def show_Method(self,image1,image2):
        cv2.imshow('image2',image2)
        cv2.imshow('image1',image1)
        cv2.waitKey(1)
    def ground_truth_callback(self,msg):
        #print("frame = "),
        #print(self.frame)
        self.GT_is_comming = True

        self.groundtruth = [msg.point.x,msg.point.y,msg.point.z]
        #print (" groundtruth"),
        #print(self.groundtruth)

    def IMU_callback(self,msg):
        self.IMU_is_comming = True
        self.IMU_INIT = [msg]
        #print(msg)

    def findDecomposedEssentialMatrix(self, p1, p2):
        # fundamental matrix and inliers
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_LMEDS, 1, 0.999)
        #F, mask = cv.findFundamentalMat(p1, p2, cv.FM_RANSAC, 1.0, 0.999)
        mask = mask.astype(bool).flatten()
        E = np.dot(self.camera_matrix_left.T, np.dot(F, self.camera_matrix_right))

        _, R, t, _ = cv2.recoverPose(E, p1[mask], p2[mask], self.camera_matrix_right)

        return R, t

    def image_callback(self,msg):
        #print("frame = "),
        #print(self.frame)
        #print (" image_callback")
        self.image_is_comming = True
        try:
            # Convert your ROS Image message to OpenCV2
            self.img = self.bridge.imgmsg_to_cv2(msg, "mono8")


        except CvBridgeError, e:
            print(e)

    def image_callbackr(self, msg):
        # print("frame = "),
        # print(self.frame)
        # print (" image_callback")
        self.image_is_comming = True
        try:
            # Convert your ROS Image message to OpenCV2
            self.imgr = self.bridge.imgmsg_to_cv2(msg, "mono8")


        except CvBridgeError, e:
            print(e)
    def main_loop(self):
        print("inside main loop")
        #while(self.img==0) :
        #print(self.topic_is_comming)

        while not rospy.is_shutdown():

            #r.sleep()
            #print(self.topic_is_comming)
            self.rate.sleep()

            #rospy.Subscriber(image_topic, Image, self.image_callback)
            if self.image_is_comming == True and self.IMU_is_comming == True and self.GT_is_comming == True:
                #self.show_Method(self.img,self.imgr)
                #####################################333
               # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=50)
                #disparity = stereo.compute(self.img, self.imgr)
                #plt.imshow(disparity, 'gray')

                #plt.show()

                ############################################################33
                corners = cv2.goodFeaturesToTrack(self.img,20,0.01,10)
                cornersr = cv2.goodFeaturesToTrack(self.imgr, 20, 0.01, 10)
                cl = cv2.goodFeaturesToTrack(self.img, 20, 0.01, 10)
                cr = cv2.goodFeaturesToTrack(self.imgr, 20, 0.01, 10)


                self.IMU = self.IMU_INIT
                self.groundtruth_for_frame =self.groundtruth

                corners = np.int0(corners)
                cornersr = np.int0(cornersr)
                ####we dont need the mask
                #mask = np.zeros_like(self.img)
                #######################3333

                pointNum = -1



                ############################new approach
                Rotation ,translation = self.findDecomposedEssentialMatrix(corners, cornersr)
                #print(cv2.findEssentialMat(corners,cornersr,[457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1.0]))
                #print (cv2.findFundamentalMat(corners,cornersr,method=cv2.RANSAC))
                #print(cv2.findEssentialMat(corners,cornersr,[[457.587, 0.0, 379.999],[ 0.0, 456.134, 255.238],[ 0.0, 0.0, 1.0]],method=cv2.RANSAC,prob=0.99))
                #print (self.camera_matrix_left)
                #pts_l_norm = cv2.undistortPoints(np.expand_dims(corners, axis=1), cameraMatrix=self.camera_matrix_left, distCoeffs=self.distortion_coefficient)
                #pts_r_norm = cv2.undistortPoints(np.expand_dims(cornersr, axis=1), cameraMatrix=self.camera_matrix_right, distCoeffs=self.distortion_coefficient)
                #E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC,
                           #                    prob=0.999, threshold=3.0)
                #points, R, t, mask = cv2.recoverPose(E, pts_l_norm, pts_r_norm)

                print ('Rotation = '),
                print (Rotation)
                print ('Translation = '),
                print (translation)



                ###############################################################################################33
                for i in corners:

                    x,y = i.ravel()
                    if((x-4)>0 and (y-4)>0 and (x+4)<752 and (y+4)<480 ):
                        pointNum +=1
                        patch =self.img[y-self.patchsize:y+self.patchsize,x-self.patchsize:x+self.patchsize]
               ####################put data in dataframe
                        self.df.loc[len(self.df)] = [pointNum,x,y,patch.flatten(),Rotation,translation,self.frame,self.groundtruth_for_frame]




                if (self.frame>0):

                    #self.find_same_patches_in_both_camera()
                    self.similarPatchsIn(self.frame,self.frame-1,self.df)



     ################################## Landmark Matching

                self.frame += 1














rospy.init_node('image_listener')
#bag_filename = 'MH_03_medium.bag'
#player_proc = subprocess.Popen(['rosbag', 'play', bag_filename])
hansel = Hansel()

hansel.main_loop()
hansel.savetoCSV()
