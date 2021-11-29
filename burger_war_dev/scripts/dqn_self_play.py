#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""""""""
設計方針:
    ・基本動作はLidar敵検出を行いつつ、学習済みモデル(TGS)で推論動作する。(BASIC)
    ・敵が一定距離内になった場合に自身の左右背後を取られないようにしつつ (DEFENCE)
    ・敵の左右、背後のターゲットを取るべく、敵に向かって進んでいく。(ATTACK)
    ・お見合いによる時間切れを防ぐため、一定時間間隔で前進後退する機能を実装 (escape)
将来構想(発散)：
    ・お見合い状態 and 自分がリード and 相手が動かない　条件の時はそのまま膠着逃げ切りを狙う
    ・敵との間合い(閾値)をハイパーパラメータとして最適値を探索させる
　　・敵の戦略をクラスタリング(教師なし学習)して、分類に応じた戦略選択を学習させる
    ・学習時、学習に応じて敵を徐々にレベルアップさせて学習が滞らないように、多様性を持たせる
    ・エポックがある程度進んだら、依然にsaveした学習済みモデルと対戦させて正しい学習ができているか確認する
"""""""""
debug_naoya = True
from enemy_camera_detector import EnemyCameraDetector
from enum import Enum
import math
import requests
import tf
import actionlib
import angles
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse
# added above libraries
import os
import sys
import time
import subprocess
import json
import requests
import copy
import rospy
import rosparam
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, PoseWithCovarianceStamped, Vector3
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from tf import transformations as tft
import cv2
import torch
import torchvision
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from utils.state import State
from utils.wallAvoid import punish_by_count, punish_by_min_dist, manual_avoid_wall_2
from utils.lidar_transform import lidar_transform
from agents.agent import Agent

if debug_naoya:
    print("TGS+Rulu-base operation mode")
print("dqn_self_play.pyを実行")

# config
FIELD_SCALE = 2.4
FIELD_MARKERS = [
    "Tomato_N", "Tomato_S", "Omelette_N", "Omelette_S", "Pudding_N", "Pudding_S",
    "OctopusWiener_N", "OctopusWiener_S", "FriedShrimp_N", "FriedShrimp_E", "FriedShrimp_W", "FriedShrimp_S"
]
ROBOT_MARKERS = {
    "r": ["RE_B", "RE_L", "RE_R"],
    "b": ["BL_B", "BL_L", "BL_R"]
}

JUDGE_URL = ""


# functions
def send_to_judge(url, data):
    res = requests.post(url,
                        json.dumps(data),
                        headers={'Content-Type': 'application/json'}
                        )
    return res

"""
Respect for SeigoRun2.py
"""
class ActMode(Enum):
    BASIC = 1
    ATTACK = 2
    ESCAPE = 3
    DEFENCE = 4

class SeigoBot2:
    # ここで状態決定　
    def mode_decision(self):
        exist, distance, direction_diff = self.detect_enemy()
        if exist == False:  # いなかったら巡回
            return ActMode.BASIC
        else:
            # print self.enemy_body_remain, self.all_field_score[0:6]
            self.enemy_info = [distance, direction_diff]
            if self.enemy_body_remain <= 1 and distance < 1.0 and self.Is_lowwer_score == False:
                return ActMode.DEFENCE
            if distance < self.snipe_th and self.enable_escape_approach == True and self.Is_lowwer_score == True:
                # if low score, once attack then try to escape.
                return ActMode.ESCAPE
            if distance < self.snipe_th:  # 発見して近かったら攻撃
                return ActMode.ATTACK
            # rospy.loginfo('detect enemy but so far')
            return ActMode.BASIC

    def enemy_position_callback(self, position):
        self.enemy_position = position

    def lidar_callback(self, scan):
        self.scan = scan

    def detect_enemy(self):
        exist, distance, direction_diff = self.detect_from_lidar()
        # もしカメラで確認できる範囲なら

        # if abs(direction_diff) < self.camera_angle_limit and distance > self.camera_range_limit[0] and distance < self.camera_range_limit[1]:
        #     exist = exist and self.is_camera_detect  # カメラとLidarのandをとる
        #     if exist == False:
        #         rospy.loginfo('detect enemy from LiDAR, but cannot detect from camera. So ignore')
        return exist, distance, direction_diff
    # RESPECT @koy_tak
    def detect_collision(self):
        front = False
        rear = False
        deg_90 = int((math.pi/2.0)/self.scan.angle_increment)
        front_count = len([i for i in self.scan.ranges[0:int(deg_90)] if i < self.distance_to_wall_th]) + \
            len([i for i in self.scan.ranges[int(deg_90)*3:-1]
                 if i < self.distance_to_wall_th])
        rear_count = len([i for i in self.scan.ranges[int(
            deg_90):int(deg_90)*3] if i < self.distance_to_wall_th])
        if front_count > 0 and rear_count == 0:
            front = True
            rospy.logwarn("front collision !!!")
        elif front_count == 0 and rear_count > 0:
            rear = True
            rospy.logwarn("rear collision !!!")
        elif front_count > 0 and rear_count > 0:
            front = front_count > rear_count
            rear = not front
            rospy.logwarn("both side collision !!!")
        # if (self.scan.ranges[0] != 0 and self.scan.ranges[0] < self.distance_to_wall_th) or (self.scan.ranges[10] != 0 and self.scan.ranges[10] < self.distance_to_wall_th) or (self.scan.ranges[350] != 0 and self.scan.ranges[350] < self.distance_to_wall_th):
        #     rospy.logwarn('front collision !!')
        #     front = True
        # if (self.scan.ranges[180] != 0 and self.scan.ranges[180] < self.distance_to_wall_th) or (self.scan.ranges[190] != 0 and self.scan.ranges[190] < self.distance_to_wall_th) or (self.scan.ranges[170] != 0 and self.scan.ranges[170] < self.distance_to_wall_th):
        #     rospy.logwarn('rear collision !!')
        #     rear = True
        return front, rear

    def status_transition(self):
        def get_mode_txt(n):
            if n == ActMode.BASIC:
                return 'basic'
            elif n == ActMode.ATTACK:
                return 'attack'
            elif n == ActMode.ESCAPE:
                return 'escape'
            elif n == ActMode.DEFENCE:
                return 'defence'
            else:
                return 'unknown'

        pre_act_mode = self.act_mode
        self.act_mode = self.mode_decision()
        if self.act_mode == ActMode.BASIC:
            self.basic()
        elif self.act_mode == ActMode.ATTACK:
            self.attack()
        elif self.act_mode == ActMode.DEFENCE:
            self.defence()
        elif self.act_mode == ActMode.ESCAPE:
            if pre_act_mode != self.act_mode:
                self.escape_mode_start_time = rospy.Time.now().to_sec()
            self.escape()
        else:
            rospy.logwarn('unknown actmode !!!')

        if pre_act_mode != self.act_mode:
            text = 'change to ' + get_mode_txt(self.act_mode)
            rospy.loginfo(text)





# main class
class DQNBot:
    """
    An operator to train the dqn agent.
    """
    def __init__(self, robot="r", online=False, policy_mode="epsilon", debug=True, save_path=None, load_path=None, manual_avoid=False):
        """
        Args:
            robot (str): robot namespace ("r" or "b")
            online (bool): training is done on online or not
            policy_mode (str): policy ("epsilon" or "boltzmann")
            debug (bool): debug mode
            save_path (str): model save path
            load_path (str): model load path
            manual_avoid (bool): manually avoid walls or not
        """
        # attributes
        self.robot = robot
        self.enemy = "b" if robot == "r" else "r"
        self.online = online
        self.policy_mode = policy_mode
        self.debug = debug
        self.my_markers = ROBOT_MARKERS[self.robot]
        self.op_markers = ROBOT_MARKERS[self.enemy]
        self.marker_list = FIELD_MARKERS + self.my_markers + self.op_markers
        self.score = {k: 0 for k in self.marker_list}
        self.past_score = {k: 0 for k in self.marker_list}
        self.my_score = 0
        self.op_score = 0

        if save_path is None:
            self.save_path = "../catkin_ws/src/burger_war_dev/burger_war_dev/scripts/models/tmp.pth"
        else:
            self.save_path = save_path

        self.load_path = load_path

        # state variables
        self.lidar_ranges = None
        self.my_pose = None
        self.image = None
        self.mask = None
        self.state = None
        self.past_state = None
        self.action = None

        # other variables
        self.game_state = "end"
        self.step = 0
        self.episode = 0
        self.bridge = CvBridge()

        # rostopic subscription
        self.lidar_sub = rospy.Subscriber('scan', LaserScan, self.callback_lidar)
        if debug_naoya:
            self.scan = LaserScan()
        if debug_naoya:
            self.camera_detector = EnemyCameraDetector()
            self.is_camera_detect = False
            self.camera_detect_angle = -360
        self.image_sub = rospy.Subscriber('image_raw', Image, self.callback_image)
        self.state_sub = rospy.Timer(rospy.Duration(0.5), self.callback_warstate)

        if self.debug:
            if self.robot == "r": self.odom_sub = rospy.Subscriber("red_bot/tracker", Odometry, self.callback_odom)
            if self.robot == "b": self.odom_sub = rospy.Subscriber("tracker", Odometry, self.callback_odom)
        else:
            self.amcl_sub = rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.callback_amcl)

        # rostopic publication
        self.twist_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # rostopic service
        self.state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.pause_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        # agent
        self.agent = Agent(num_actions=len(ACTION_LIST), batch_size=BATCH_SIZE, capacity=MEM_CAPACITY, gamma=GAMMA, prioritized=PRIOTIZED, lr=LR)

        if self.load_path is not None:
            self.agent.load_model(self.load_path)

        # mode
        self.punish_if_facing_wall = not manual_avoid

        self.punish_far_from_center = True
    
    def callback_lidar(self, data):
        """
        callback function of lidar subscription
        Args:
            data (LaserScan): distance data of lidar
        """
        raw_lidar = np.array(data.ranges)
        raw_lidar = lidar_transform(raw_lidar, self.debug)
        self.lidar_ranges = torch.FloatTensor(raw_lidar).view(1, 1, -1)   # (1, 1, 360)

    def callback_image(self, data):
        """
        callback function of image subscription
        Args:
            data (Image): image from from camera mounted on the robot
        """
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")  # 640x480[px]

            # preprocess image
            img = img[100:, :]   # 640x380[px]
            img = cv2.resize(img, (160, 95))
            deriv_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
            deriv_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
            grad = np.sqrt(deriv_x ** 2 + deriv_x ** 2)
            
            '''
            # visualize preprocessed image for debug
            def min_max(x, axis=None):
                min = x.min(axis=axis, keepdims=True)
                max = x.max(axis=axis, keepdims=True)
                result = (x-min)/(max-min)
                return result
            cv2.imshow('grad', min_max(grad))
            cv2.waitKey(1)
            '''

            img = torchvision.transforms.ToTensor()(grad)
            self.image = img.unsqueeze(0)                   # (1, 3, 95, 160)
        except CvBridgeError as e:
            rospy.logerr(e)
    
    def callback_odom(self, data):
        """
        callback function of tracker subscription
        Args:
            data (Odometry): robot pose
        """
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        if self.robot == "b":
            x *= -1
            y *= -1
        self.my_pose = torch.FloatTensor([x, y]).view(1, 2)

    def callback_amcl(self, data):
        """
        callback function of amcl subscription
        Args:
            data (PoseWithCovarianceStamped): robot pose
        """
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        self.my_pose = torch.FloatTensor([-y, x]).view(1, 2)

    def callback_warstate(self, event):
        """
        callback function of warstate subscription
        Notes:
            https://github.com/p-robotics-hub/burger_war_kit/blob/main/judge/README.md
        """
        # get the game state from judge server by HTTP request
        resp = requests.get(JUDGE_URL + "/warState")
        json_dict = json.loads(resp.text)
        self.game_state = json_dict['state']
        self.my_score = int(json_dict['scores'][self.robot])
        self.op_score = int(json_dict['scores'][self.enemy])
        #print("name:{}, state:{}, score:{}".format(self.robot, self.game_state, self.my_score))
        
        if self.game_state == "running":            
            for tg in json_dict["targets"]:
                if tg["player"] == self.robot:
                    self.score[tg["name"]] = int(tg["point"])
                elif tg["player"] == self.enemy:
                    self.score[tg["name"]] = -int(tg["point"])

            msk = []
            for k in self.marker_list:
                if self.score[k] > 0:    msk.append(0)
                elif self.score[k] == 0: msk.append(1)
                else:                    msk.append(1)
            
            self.mask = torch.FloatTensor(msk).view(1, 18)

    def get_reward(self, past, current):
        """
        reward function.
        
        Args:
            past (dict): score dictionary at previous step
            current (dict): score dictionary at current step
        Return:
            reward (int)
        """
        diff_my_score = {k: current[k] - past[k] for k in self.score.keys() if k not in self.my_markers}
        diff_op_score = {k: current[k] - past[k] for k in self.my_markers}

        # Check LiDAR data to punish for AMCL failure
        if self.punish_if_facing_wall:
            #bad_position = punish_by_count(self.lidar_ranges, dist_th=DIST_TO_WALL_TH, count_th=NUM_LASER_CLOSE_TO_WALL_TH)
            bad_position = punish_by_min_dist(self.lidar_ranges, dist_th=0.13)
            if self.punish_far_from_center:
                pose = self.my_pose.squeeze()
                dist_from_center = torch.sqrt(torch.pow(pose, 2).sum()).item()
                if dist_from_center > 1.0:
                    bad_position -= 1.0
        # else:
        #     if self.punish_far_from_center:
        #         pose = self.my_pose.squeeze()
        #         bad_position = punish_by_count(self.lidar_ranges, dist_th=0.2, count_th=90)
        #         if abs(pose[0].item()) > 1:
        #             bad_position -= 0.1
        #         if abs(pose[1].item()) > 1:
        #             bad_position -= 0.1
        #     else:
        #         bad_position = 0

        plus_diff = sum([v for v in diff_my_score.values() if v > 0])
        minus_diff = sum([v for v in diff_op_score.values() if v < 0])

        return plus_diff + minus_diff + bad_position

    def strategy(self):

        # past state
        self.past_state = self.state

        # current state
        self.state = State(
            self.my_pose,           # (1, 2)
            self.lidar_ranges,      # (1, 1, 360)
            self.image,             # (1, 3, 480, 640)
            self.mask,              # (1, 18)
        )

        if self.action is not None:
            current_score = copy.deepcopy(self.score)
            reward = self.get_reward(self.past_score, current_score)
            print("reward: {}".format(reward))
            self.past_score = current_score
            reward = torch.LongTensor([reward])
            self.agent.memorize(self.past_state, self.action, self.state, reward)

        # manual wall avoidance
        if not self.punish_if_facing_wall:
            avoid, linear_x, angular_z = manual_avoid_wall_2(self.lidar_ranges, dist_th=0.13, back_vel=0.2)
        else:
            avoid = False

        if avoid:
            self.action = None
        else:
            # get action from agent
            if self.robot == "b":
                if self.episode % 2 == 0:
                    policy = "boltzmann"
                else:
                    policy = "epsilon"
            else:
                policy = "epsilon"

            self.action = self.agent.get_action(self.state, self.episode, policy, self.debug)
            choice = int(self.action.item())

            linear_x = ACTION_LIST[choice][0]
            angular_z = ACTION_LIST[choice][1]

        print("step: {}, vel:{}, omega:{}".format(self.step, linear_x, angular_z))

        # update twist
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = angular_z
        self.twist_pub.publish(twist)

        self.step += 1

    def move_robot(self, model_name, position=None, orientation=None, linear=None, angular=None):
        state = ModelState()
        state.model_name = model_name
        # set pose
        pose = Pose()
        if position is not None:
            pose.position = Point(*position)
        if orientation is not None:
            tmpq = tft.quaternion_from_euler(*orientation)
            pose.orientation = Quaternion(tmpq[0], tmpq[1], tmpq[2], tmpq[3])
        state.pose = pose
        # set twist
        twist = Twist()
        if linear is not None:
            twist.linear = Vector3(*linear)
        if angular is not None:
            twist.angular = Vector3(*angular)
        state.twist = twist
        try:
            self.state_service(state)
        except rospy.ServiceException, e:
            print("Service call failed: %s".format(e))

    def init_amcl_pose(self):
        amcl_init_pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=1)
        amcl_pose = PoseWithCovarianceStamped()
        amcl_pose.header.stamp = rospy.Time.now()
        amcl_pose.header.frame_id = "map"
        amcl_pose.pose.pose.position.x = -1.3
        amcl_pose.pose.pose.position.y = 0.0
        amcl_pose.pose.pose.position.z = 0.0
        amcl_pose.pose.pose.orientation.x = 0.0
        amcl_pose.pose.pose.orientation.y = 0.0
        amcl_pose.pose.pose.orientation.z = 0.0
        amcl_pose.pose.pose.orientation.w = 1.0
        amcl_init_pub.publish(amcl_pose)

    def stop(self):
        if self.robot == "r":
            print("***** EPISODE {} DONE *****".format(self.episode))
            rospy.sleep(0.5)
            self.pause_service()

    def restart(self):
        self.episode += 1

        if self.robot == "r":
            # restart judge server
            resp = send_to_judge(JUDGE_URL + "/warState/state", {"state": "running"})

            # restart gazebo physics
            self.unpause_service()

        # reset amcl pose
        self.init_amcl_pose()

        print("restart the game by {}".format(self.robot))

    def reset(self):
        # reset parameters
        self.step = 0
        self.score = {k: 0 for k in self.marker_list}
        self.past_score = {k: 0 for k in self.marker_list}
        self.my_score = 0
        self.op_score = 0
        self.lidar_ranges = None
        self.my_pose = None
        self.image = None
        self.mask = None
        self.state = None
        self.past_state = None
        self.action = None

        if self.robot == "r":
            # reset judge server
            subprocess.call('bash ../catkin_ws/src/burger_war_dev/burger_war_dev/scripts/reset.sh', shell=True)

            # reset robot's positions
            self.move_robot("red_bot", (0.0, -1.3, 0.0), (0, 0, 1.57), (0, 0, 0), (0, 0, 0))
            self.move_robot("blue_bot", (0.0, 1.3, 0.0), (0, 0, -1.57), (0, 0, 0), (0, 0, 0))

    def train(self, n_epochs=20):
        for epoch in range(n_epochs):
            print("episode {}: epoch {}".format(self.episode, epoch))
            self.agent.update_policy_network()
    
    def run(self, rospy_rate=1):

        r = rospy.Rate(rospy_rate)

        while not rospy.is_shutdown():
            
            while not all([v is not None for v in [self.lidar_ranges, self.my_pose, self.image, self.mask]]):
                pass

            if self.game_state == "stop" and self.debug:
                
                if not self.debug:
                    break

                # stop the game
                self.stop()

                # offline learning
                if not self.online:
                    self.train(n_epochs=EPOCHS)

                # update target q function
                if self.episode % UPDATE_Q_FREQ == 0:
                    self.agent.update_target_network()

                # save model
                if self.my_score > self.op_score:
                    self.agent.save_model(self.save_path)
                    if self.episode % 100 == 0:
                        self.agent.save_model(self.save_path.split(".pth")[0] + "_ckpt_{}.pth".format(self.episode))
                    print("{} Win the Game and Save model".format(self.robot))
                else:
                    time.sleep(1.5)
                    try:
                        self.agent.load_model(self.save_path)
                        print("{} Lose the Game and Load model".format(self.robot))
                    except:
                        print("{} cannot load model".format(self.robot))

                # reset the game
                self.reset()

                time.sleep(1)

                # restart the game
                self.restart()

            elif self.game_state == "running":
                self.strategy()

                # online learning
                if self.online and self.debug:
                    self.agent.update_policy_network()

            r.sleep()

    
if __name__ == "__main__":
    print("dqn_self_play_main_.pyを実行")
    rospy.init_node('dqn_run')
    JUDGE_URL = rospy.get_param('/send_id_to_judge/judge_url')

    try:
        ROBOT_NAME = rosparam.get_param('DQNRun/side')
    except:
        try:
            ROBOT_NAME = rosparam.get_param('enemyRun/side')
        except:
            ROBOT_NAME = "b"

    print("name: {}, server: {}".format(ROBOT_NAME, JUDGE_URL))

    # parameters
    print("parametersの設定")
    ONLINE = True
    POLICY = "epsilon"
    DEBUG = False
    SAVE_PATH = None
    LOAD_PATH = "../catkin_ws/src/burger_war_dev/burger_war_dev/scripts/models/20210314.pth"
    MANUAL_AVOID = False

    # wall avoidance
    print("wall avoidanceの設定")
    DIST_TO_WALL_TH = 0.18
    NUM_LASER_CLOSE_TO_WALL_TH = 30

    # action lists
    print("action lists設定")
    VEL = 0.2
    OMEGA = 30 * 3.14/180
    ACTION_LIST = [
        [VEL, 0],
        [-VEL, 0],
        [0, 0],
        [0, OMEGA],
        [0, -OMEGA],
    ]

    # agent config
    print("agent config設定")
    UPDATE_Q_FREQ = 10
    BATCH_SIZE = 16
    MEM_CAPACITY = 2000
    GAMMA = 0.99
    PRIOTIZED = False
    LR = 0.0005
    EPOCHS = 20

    # time freq [Hz]
    RATE = 1

    try:
        bot = DQNBot(robot=ROBOT_NAME, online=ONLINE, policy_mode=POLICY, debug=DEBUG, save_path=SAVE_PATH, load_path=LOAD_PATH, manual_avoid=MANUAL_AVOID)
        bot.run(rospy_rate=RATE)

    except rospy.ROSInterruptException:
        pass
