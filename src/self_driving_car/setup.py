from setuptools import setup
import os 
from glob import glob

package_name = 'self_driving_car'
config = 'config'
computer_vision = 'computer_vision'
computer_vision_lanes = 'computer_vision/lanes'
computer_vision_signs = 'computer_vision/signs'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, config, computer_vision, computer_vision_lanes, computer_vision_signs],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name,'launch'), glob('launch/*')),
        (os.path.join('share', package_name,'worlds'), glob('worlds/*')),
        (os.path.join('lib', package_name), glob('scripts/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marshf16',
    maintainer_email='ryan.marshall168@gmail.com',
    description='Self driving car inside simulation using Computer Vision and Deep Learning',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'start_car = self_driving_car.start_car:main'
        ],
    },
)
