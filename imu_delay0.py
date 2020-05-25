import rospy
from sensor_msgs.msg import Imu
class delay:
	def __init__(self):
		rospy.init_node("One_node_to_rule_them_all")
#pub = rospy.Publisher("echo", String,queue_size =10)
		self.rate = rospy.Rate(10)
		self.MSG=[]
		self.pub2 = rospy.Publisher("/imu1", Imu, queue_size=10)
		rospy.Subscriber("/imu0", Imu, callback=self.callback,callback_args=self.pub2)





	def callback(self,msg,pub):
		pub.publish(msg)
		#pub.publish(input.data)
        # The message has 2 parts, a header and the data. We only want to use the actual data
	def main(self):
		while not rospy.is_shutdown():
			#rospy.Subscriber("/imu0", Imu,callback=self.callback)


			#self.pub2.publish()
			self.rate.sleep()
d = delay()
d.main()