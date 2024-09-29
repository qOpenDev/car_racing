"""
agt_targetQ.py
経験再生に
ターゲットネットワークを追加したQ学習のアルゴリズム
"""
import os.path
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


class ReplayMemory:  # (B)
    """ 経験を記録するクラス """
    def __init__(self, memory_size):
        self.memory_size = memory_size  # 記憶サイズ
        self.memory = []

    def __len__(self):  # (C)
        """ len()で、memoryの長さを返す """
        return len(self.memory)

    def add(self, experience):  # (D)
        """ 経験を記憶に追加する """
        # 経験の保存
        self.memory.append(experience)

        # 容量がオーバーしたら古いものを1つ捨てる
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def sample(self, data_length):  # (E)
        """ data_length分、ランダムにサンプルする """
        out = random.sample(self.memory, data_length)
        return out


class TargetQAgent():
    """
    経験再生にターゲットネットワークを取り入れた
    Q学習エージェントクラス
    """

    def __init__(self, **kwargs):
        super().__init__()

        # 行動の種類数（ネットワークの出力数）
        self.n_action = kwargs.pop('n_action', -1)
        # 入力サイズ
        self.input_size = kwargs.pop('input_size', ())
        # 中間層のニューロン数
        self.n_dense = kwargs.pop('n_dense', 32)
        # 乱雑度
        self.epsilon = kwargs.pop('epsilon', 0.1)
        # 割引率
        self.gamma = kwargs.pop('gamma', 0.9)
        # 保存ファイル名
        # self.file_path = kwargs.pop('file_path', '')
        # ターゲットインターバル
        self.target_interval = kwargs.pop('target_interval', 10)
        # 記憶サイズ
        self.replay_memory_size = kwargs.pop('replay_memory_size', 100)
        # 学習のバッチサイズ
        self.replay_batch_size = kwargs.pop('replay_batch_size', 20)
        # 経験再生クラスのインスタンス生成
        self.replay_memory = ReplayMemory(memory_size=self.replay_memory_size)

        keras.backend.clear_session()

        # モデルの作成
        self.model = self._build_qnet()
        # self.model = None
        # ターゲットモデルの作成
        self.model_target = self._build_qnet()
        # self.model_target = None
        # タイムステップのカウンター
        self.time = 0

        # print(self.model.summary())

    def learn(self, obs, act, rwd, done, next_obs):
        """ 学習 """
        if rwd is None:
            return
        # ------------------------- 編集ここから
        # 経験の保存 (A)
        self.replay_memory.add((obs, act, rwd, done, next_obs))

        # 学習 (B)
        self._fit()

        # target_intervalの周期でQネットワークの重みをターゲットネットにコピー (C)
        if self.time % self.target_interval == 0 \
                and self.time > 0:
            self.model_target.set_weights(
                self.model.get_weights())

        self.time += 1
        # ------------------------- ここまで

    def _fit(self):
        """ 学習 """
        # 記憶された「経験」の量がバッチサイズに満たない場合は戻る (A)
        if len(self.replay_memory) < self.replay_batch_size:
            return

        # バッチサイズ個の経験を記憶からランダムに取得 (B)
        outs = self.replay_memory.sample(self.replay_batch_size)

        # 観測のバッチを入れる配列を準備 (C)
        obs = outs[0][0]  # 1番目の経験の観測
        obss = np.zeros(
            (20, obs.shape[1], obs.shape[2], obs.shape[3]),
            dtype=int
        )

        # ターゲットのバッチを入れる配列を準備 (D)
        targets = np.zeros((self.replay_batch_size, self.n_action))

        # 経験ごとのループ (E)
        for i, out in enumerate(outs):
            # 経験を要素に分解 (F)
            obs, action_id, reward, done, next_obs = out

            # obs に対するQネットワークの出力 yを得る (G)
            y = self.get_q(obs, type='main')

            # target にyの内容をコピーする (H)
            target = y.copy()

            if done is False:
                # 最終状態でなかったら next_obsに対する next_yを
                # ターゲットネットから得る (I)
                next_y = self.get_q(next_obs, type='target')

                # Q[obs][action_id]のtarget_act を作成
                target_act = reward + self.gamma * max(next_y)
            else:
                # 最終状態の場合は報酬だけでtarget_actを作成
                target_act = reward

            # targetのactの要素だけtarget_actにする
            target[action_id] = target_act

            # obsとtargetをバッチの配列に入れる
            obss[i, :] = obs
            targets[i, :] = target

        # obssと targets のバッチのペアを与えて学習
        self.model.fit(obss, targets, verbose=0, epochs=1)

    def get_q(self, obs, type='main'):
        """
        観測に対するQ値を出力
        """
        input_array = obs
        q = []
        if type == 'main':
            # Qネットワークに観測obsを入力し出力を得る (A)
            q = self.model.predict(input_array, verbose=0)
        elif type == 'target':
            # ターゲットネットに観測obsを入力し出力を得る (B)
            q = self.model_target.predict(input_array, verbose=0)
        else:
            raise ValueError('get_Q のtype が不適切です')
        output_q = q[0, :]

        return output_q

    def _build_qnet(self):
        """
        指定したパラメータでQネットワークを構築
        """
        keras.backend.clear_session()

        model = keras.models.Sequential()
        model.add(
            Conv2D(
                filters=16,
                input_shape=(96, 96, 1),
                kernel_size=(3, 3),
                activation='relu'
            )
        )
        model.add(
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation='relu'
            )
        )
        model.add(
            MaxPooling2D(
                pool_size=(2, 2)
            )
        )
        model.add(
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation='relu'
            )
        )
        model.add(
            MaxPooling2D(
                pool_size=(2, 2)
            )
        )
        model.add(
            Dropout(
                rate=0.25
            )
        )
        model.add(
            Flatten()
        )
        model.add(
            Dense(
                self.n_dense,
                activation='relu'
            )
        )
        model.add(
            Dropout(
                rate=0.25
            )
        )
        model.add(
            Dense(
                self.n_action,
                activation='softmax'
            )
        )

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

        return model

    def select_action(self, obs):
        """
        観測に対して行動を出力
        """
        action_resource = 'from_q'

        # 確率的に処理を分岐
        if np.random.rand() < self.epsilon:
            # ランダム行動
            action_id = np.random.randint(0, self.n_action)
            action_resource = 'random'
        else:
            # obsに対するQ値のリストを取得
            q = self.get_q(obs)

            # Qを最大にする行動
            action_id = np.argmax(q)

        return action_id, action_resource

    def save_weight(self, file_name):
        path = './weight'
        if not os.path.exists(path):
            os.mkdir(path)

        self.model.save(os.path.join(path, file_name + '.keras'))

    def load_weight(self, file_name: str):
        keras.backend.clear_session()

        self.model = keras.models.load_model(file_name)
        self.model_target = keras.models.load_model(file_name)
        if not self.model or not self.model_target:
            print('学習済みモデルをロードできませんでした.')
            sys.exit()

        print(self.model.summary())
