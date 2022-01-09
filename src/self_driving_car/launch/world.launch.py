import os
from ament_index_python.packages import get_package_share_directory 
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():

  package_dir = get_package_share_directory('self_driving_car')
  # world_file = os.path.join(package_dir, 'worlds', 'self_driving_car.world')
  # world_file = os.path.join(package_dir, 'worlds', 'test_sign_detection.world')
  world_file = os.path.join(package_dir, 'worlds', 'realistic_self_driving_car_map.world')

  return LaunchDescription([

        ExecuteProcess(
          cmd=['gazebo', '--verbose', world_file, '-s', 'libgazebo_ros_factory.so'], output='screen'
        ),
  ])
