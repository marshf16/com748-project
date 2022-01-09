import cv2
from geometry_msgs.msg import Twist
from rclpy.node import Node 
from cv_bridge import CvBridge 
from sensor_msgs.msg import Image 
import rclpy
from .drive_car import Car

class car_controller(Node):
    def __init__(self):
        super().__init__('video_subscriber')

        # Create subsriber to get video feed
        self.subscriber = self.create_subscription(Image, '/camera/image_raw', self.process_data, 10)

        # Create publisher to send vehicle speed
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 40)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.send_cmd_vel)

        # Properties of car
        self.velocity = Twist()
        self.bridge = CvBridge()
        self.Car = Car()

    def send_cmd_vel(self):
        self.publisher.publish(self.velocity)
        
    def process_data(self, data): 
        # Convert frame to OpenCV format
        opencv_frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')

        # Get values required to drive car for current frame
        angle, speed, image = self.Car.drive_car(opencv_frame)

        # Apply values to drive car
        self.velocity.angular.z = angle
        self.velocity.linear.x = speed      

        cv2.imshow("FRONT CAMERA", image)
        cv2.waitKey(1)


def main(args=None):
  rclpy.init(args=args)
  cc = car_controller()
  rclpy.spin(cc)
  rclpy.shutdown()

if __name__ == '__main__':
	main()