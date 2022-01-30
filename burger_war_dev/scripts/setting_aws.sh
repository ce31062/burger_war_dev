# install packages
sudo apt-get update -y
sudo apt-get install -y ros-melodic-dwa-local-planner
sudo apt-get install -y ros-melodic-global-planner
sudo apt-get install -y ros-melodic-rviz
sudo apt-get install -y ros-melodic-costmap-converter
sudo apt-get install -y ros-melodic-mbf-costmap-core
sudo apt install -y ros-melodic-teb-local-planner
sudo apt install -y libarmadillo-dev libarmadillo8  # necessary to install obstacle_detector
sudo apt install -y wget
sudo apt install -y xdotool curl wmctrl

# library install
cd $HOME/catkin_ws/src/burger_war_dev
pip install -r requirements.txt

# catkin build
cd $HOME/catkin_ws
catkin build
source ~/catkin_ws/devel/setup.bash
