#!python3
#-*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import random
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
#import torchvision.models as models

from utils.state import State
from utils.transition import Transition
from utils.random_replaymemory import RandomReplayMemory
from utils.permemory import PERMemory
from networks.maskNet import MaskNet
import pickle

print("brain.pyを実行する")
#------------------------------------------------

class Brain:
    TARGET_UPDATE = 10
    def __init__(self, num_actions, batch_size=32, capacity=10000, gamma=0.99, prioritized=True, lr=0.0005):
	print("Brainの初期化")
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_actions = num_actions
        self.prioritized = prioritized


        # Instantiate memory object
        if self.prioritized:
            print('* Prioritized Experience Replay Mode')
            self.memory = PERMemory(capacity)
        else:
            print('* Random Experience Replay Mode')
            self.memory = RandomReplayMemory(capacity)

        # Build network
        self.policy_net = MaskNet(self.num_actions, duel=False)
        self.target_net = MaskNet(self.num_actions, duel=False)
        self.target_net.eval()

        # Set device type; GPU or CPU (Use GPU if available)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu') #temporary setting
        self.policy_net = self.policy_net.to(self.device)

        self.target_net = self.target_net.to(self.device)

        print('using device:', self.device)
        #print(self.policy_net)  # Print network

        # Configure optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def replay(self):
        """Experience Replayでネットワークの重みを学習 """
	print("Experience Replayでネットワークの重みを学習 ")
        # Do nothing while size of memory is lower than batch size
        if len(self.memory) < self.batch_size:
            return

        # Extract datasets and their corresponding indices from memory
        transitions, indexes = self.memory.sample(self.batch_size)

        # ミニバッチの作成-----------------

        # transitionsは1stepごとの(state, action, next_state, reward)が、self.batch_size分格納されている
        # つまり、(state, action, next_state, reward)×self.batch_size
        # これをミニバッチにしたい。つまり
        # (state×self.batch_size, action×BATCH_SIZE, next_state, reward×BATCH_SIZE)にする
        batch = Transition(*zip(*transitions))
        batch_state = State(*zip(*batch.state))
        batch_next_state = State(*zip(*batch.next_state))

        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするマスクを作成
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(self.device)

        # バッチから状態、行動、報酬を格納（non_finalはdoneになっていないstate）
        # catはConcatenates（結合）のことです。
        # 例えばstateの場合、[torch.FloatTensor of size 1x4]がself.batch_size分並んでいるのですが、
        # それを size self.batch_sizex4 に変換します
        pose_batch = Variable(torch.cat(batch_state.pose)).to(self.device)
        lidar_batch = Variable(torch.cat(batch_state.lidar)).to(self.device)
        image_batch = Variable(torch.cat(batch_state.image)).to(self.device)
        mask_batch = Variable(torch.cat(batch_state.mask)).to(self.device)

        action_batch = Variable(torch.cat(batch.action)).to(self.device)
        reward_batch = Variable(torch.cat(batch.reward)).to(self.device)

        non_final_next_poses = Variable(torch.cat([s for s in batch_next_state.pose if s is not None])).to(self.device)
        non_final_next_lidars = Variable(torch.cat([s for s in batch_next_state.lidar if s is not None])).to(self.device)
        non_final_next_images = Variable(torch.cat([s for s in batch_next_state.image if s is not None])).to(self.device)
        non_final_next_masks = Variable(torch.cat([s for s in batch_next_state.mask if s is not None])).to(self.device)


        # ミニバッチの作成終了------------------

        # ネットワークを推論モードに切り替える
        self.policy_net.eval()

        # Q(s_t, a_t)を求める
        # self.policy_net(state_batch)は、[torch.FloatTensor of size self.batch_sizex2]になっており、
        # 実行したアクションに対応する[torch.FloatTensor of size self.batch_sizex1]にするために
        # gatherを使用します。
        state_action_values = self.policy_net(pose_batch, lidar_batch, image_batch, mask_batch).gather(1, action_batch)

        # max{Q(s_t+1, a)}値を求める。
        # 次の状態がない場合は0にしておく
        next_state_values = Variable(torch.zeros(self.batch_size).type(torch.FloatTensor)).to(self.device)

        # double dqn part
        a_m = Variable(torch.zeros(self.batch_size).type(torch.LongTensor)).to(self.device)
        a_m[non_final_mask] = self.policy_net(non_final_next_poses,
                                            non_final_next_lidars,
                                            non_final_next_images,
                                            non_final_next_masks).max(1)[1].detach()

        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 次の状態がある場合の値を求める
        # 出力であるdataにアクセスし、max(1)で列方向の最大値の[値、index]を求めます
        # そしてその値（index=0）を出力します
        next_state_values[non_final_mask] = self.target_net(
                                                non_final_next_poses,
                                                non_final_next_lidars,
                                                non_final_next_images,
                                                non_final_next_masks
                                            ).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 教師となるQ(s_t, a_t)値を求める
        expected_state_action_values = reward_batch + self.gamma * next_state_values
        expected_state_action_values = expected_state_action_values.unsqueeze(1)

        # ネットワークを訓練モードに切り替える
        self.policy_net.train()  # TODO: No need?

        # 損失関数を計算する。smooth_l1_lossはHuberlossです
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values)

        # ネットワークを更新します
        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # バックプロパゲーションを計算
        self.optimizer.step()  # 結合パラメータを更新

        # Update priority
        if self.prioritized and indexes != None:
            for i, val in enumerate(state_action_values):
                td_err = abs(expected_state_action_values[i].item() - val.item())
                self.memory.update(indexes[i], td_err)


    def decide_action(self, state, episode, policy_mode="epsilon", debug=True):
        """
        policy
        Args:
            state (State): state object
            episode (int): current episode
            policy_mode (str): exploration methods
                - epsilon: deterministic policy with eps-greedy
                - boltzmann: stochastic policy by softmax
            debug (bool): whether train or inference
        """

        if not debug:
            self.policy_net.eval()  # ネットワークを推論モードに切り替える
            print("推論を行う")
            # Set device type; GPU or CPU
            input_pose = Variable(state.pose).to(self.device)
            input_lidar = Variable(state.lidar).to(self.device)
            input_image = Variable(state.image).to(self.device)
            input_mask = Variable(state.mask).to(self.device)

            # Infer
            output = self.policy_net(input_pose, input_lidar, input_image, input_mask)
            action = output.data.max(1)[1].view(1, 1)

            return action

        if policy_mode == "epsilon":
            # ε-greedy法で徐々に最適行動のみを採用する
            # epsilon = 0.5 * (1 / (episode + 1))
            print("ε-greedy法を採用する")
            if episode < 50:
                epsilon = 0.25
            elif episode < 100:
                epsilon = 0.15
            else:
                epsilon = 0.05

            if epsilon <= np.random.uniform(0, 1):
                self.policy_net.eval()  # ネットワークを推論モードに切り替える
                # Set device type; GPU or CPU
                input_pose = Variable(state.pose).to(self.device)
                input_lidar = Variable(state.lidar).to(self.device)
                input_image = Variable(state.image).to(self.device)
                input_mask = Variable(state.mask).to(self.device)

                # Infer
                output = self.policy_net(input_pose, input_lidar, input_image, input_mask)
                action = output.data.max(1)[1].view(1, 1)

                print("Q-values: {}, Action: {}".format(output[0], action.item()))
            else:
                # Generate random value [0.0, 1.0)
                action = torch.LongTensor([[random.randrange(self.num_actions)]])
                action = action.to(self.device)
                print("Random action: {}".format(action.item()))

        elif policy_mode == "boltzmann":
            print("boltzmann法を採用する")
            self.policy_net.eval()  # ネットワークを推論モードに切り替える

            # Set device type; GPU or CPU
            input_pose = Variable(state.pose).to(self.device)
            input_lidar = Variable(state.lidar).to(self.device)
            input_image = Variable(state.image).to(self.device)
            input_mask = Variable(state.mask).to(self.device)

            # Infer
            output = self.policy_net(input_pose, input_lidar, input_image, input_mask)
            prob = F.softmax(output, dim=1)

            action = torch.multinomial(prob, 1)
            print("Prob: {}, Action: {}".format(prob[0], action.item()))

        else:
            print("Error: policy_mode is 'epsilon' or 'boltzmann'")
            action = torch.LongTensor([[random.randrange(self.num_actions)]])
            action = action.to(self.device)

        return action  # FloatTensor size 1x1

    def save_model(self, path):
        # Save a model checkpoint.
        print('Saving model...: {}'.format(path))
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        print('Loading model...: {}'.format(path))
        #model = torch.load(path)
        model = torch.load(path, torch.device('cpu'))        
        self.policy_net.load_state_dict(model)
        self.update_target_network()

    def save_memory(self, path):
        print('Saving memory (size={})...: {}'.format(len(self.memory) ,path))
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_memory(self, path):
        print('Loading memory...: {}'.format(path))
        with open(path, 'rb') as f:
            self.memory =  pickle.load(f)
            print('Loaded memory (size={})'.format(len(self.memory)))

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
