import copy

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import cv2
import numpy as np

import myutil


class Environment():
    """

    """
    # 行動空間 (Steering Wheel=-1から1, Gas=0から1, Break=0から1)
    action_space = [
        (-1, 1, 0),
        (0, 1, 0),
        (1, 1, 0),
        (-1, 1, 0.2),
        (0, 1, 0.2),
        (1, 1, 0.2),
        (-1, 0, 0.2),
        (0, 0, 0.2),
        (1, 0, 0.2),
        (-1, 0, 0),
        (0, 0, 0),
        (1, 0, 0)
    ]

    # no_rand_less_break
    # action_space = [
    #     (-1, 1, 0),
    #     (0, 1, 0),
    #     (1, 1, 0),
    #     (-1, 0.5, 0),
    #     (0, 0.5, 0),
    #     (1, 0.5, 0),
    #     (-1, 1, 0.2),
    #     (0, 1, 0.2),
    #     (1, 1, 0.2),
    # ]

    # 行動数
    n_action = len(action_space)

    def __init__(self):
        super().__init__()
        self.gym_env = gym.make("CarRacing-v2", domain_randomize=False, render_mode="rgb_array_list")
        self.record_env = RecordVideo(self.gym_env, "./video", video_length=0)

        self.breaking_count = 0
        self.previous_breaking = False

    def reset(self):
        state_image, _ = self.record_env.reset(options={"randomize": False})
        state_image = self._convert_image(state_image)
        return state_image

    def step(self, action_id):
        if action_id == -1:
            action = (0, 0.1, 0)
        else:
            action = self.action_space[action_id]

        # terminated: ゴールへの到達
        # truncation: 時間制限もしくは境界外へ出た
        obs, reward, terminated, truncation, info = self.record_env.step(action)

        #
        # ブレーキの回数をカウント
        #
        steering, accelerator, breaking = action
        if breaking > 0:
            if self.previous_breaking:
                self.breaking_count += 1
            self.previous_breaking = True
        else:
            self.breaking_count = 0
            self.previous_breaking = False

        negative_breaking = False
        if self.breaking_count > 20:
            if float(reward) < 0:
                reward *= 2.0
                negative_breaking = True

        done = False
        if terminated or truncation:
            done = True

        state_image = self._convert_image(obs)

        return reward, done, state_image, negative_breaking

    def render(self):
        image_list = self.record_env.render()
        return image_list[0]

    def _convert_image(self, image):
        norm_image = copy.deepcopy(image)
        norm_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
        norm_image = norm_image.astype(float)
        norm_image /= 255.0

        reshape_image = norm_image.reshape([1, 96, 96, 1])
        return reshape_image

    def get_action(self, action_id):
        if action_id == -1:
            return (0, 0.1, 0)
        else:
            return self.action_space[action_id]
