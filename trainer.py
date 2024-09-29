"""
trainer.py
環境とエージェントを組み合わせて動かす
学習シミュレーションと学習なしシミュレーション
"""
import os
import time
import datetime
import random
import cv2
import numpy as np

import moviepy.editor as mpy


class Recorder:
    """ シミュレーションの履歴保存クラス """

    def __init__(self):
        """ 初期処理 """
        self.reward_in_episode = 0  # 報酬/エピソード
        self.rewards_in_episode = []  # 報酬/エピソードのリスト
        self.step_in_episode = 0  # ステップ数/エピソード
        self.steps_in_episode = []  # ステップ数/エピソードのリスト

    def reset(self):
        """ 記録の初期化 """
        self.reward_in_episode = 0
        self.rewards_in_episode = []
        self.step_in_episode = 0
        self.steps_in_episode = []

    def add(self, reward, done):
        """ 記録の追加 """
        # 報酬がNoneだったらカウントしない
        if reward is None:
            return

        # 報酬とステップのカウント
        self.reward_in_episode += reward
        self.step_in_episode += 1

        # 最終状態なら
        # 今のreward_in_episodeとstep_in_episodeを
        # リストに保存してリセット
        if done is True:
            self.rewards_in_episode.append(self.reward_in_episode)
            self.reward_in_episode = 0
            self.steps_in_episode.append(self.step_in_episode)
            self.step_in_episode = 0

    def get_result(self):
        """ 報酬とステップのリストの平均を出力 """
        mean_rewards = np.mean(self.rewards_in_episode)
        self.rewards_in_episode = []
        mean_steps = np.mean(self.steps_in_episode)
        self.steps_in_episode = []
        return mean_rewards, mean_steps


class Trainer:
    """ トレーナークラス """

    def __init__(  # 引数とデフォルト値の設定 (A)
            self,
            agt=None,  # TableQAgt class: エージェント
            env=None,  # CorridorEnv class: 環境
            eval_env=None,  # CorridorEnv class: 評価用環境
            **kwargs,
    ):
        """ 初期処理 """
        # エージェントと環境のインスタンス (B)
        self.previous_total_reward = None
        self.agent = agt  # エージェント
        self.env = env  # 学習用
        self.eval_env = eval_env  # 評価用

        # 学習過程の記録関連 (C)
        self.history_rewards = []  # rewards/episodeの履歴
        self.history_steps = []  # steps/episodeの履歴
        self.history_time = []  # 上記のステップ数
        self.history_start_time = 0  # スタート時のステップ数
        self.obss = None  # Q値のチェック用の観測
        self.recorder = Recorder()  # データ保存用クラス

        # 学習パラメータ
        self.max_frames = kwargs.get('max_frames', 1000)
        self.start_episode = kwargs.get('start_episode', 1)
        self.max_episodes = kwargs.get('max_episodes', 500)
        self.skip_frames = kwargs.get('skip_frames', 5)
        self.skip_frames_from_start = kwargs.get('skip_frames_from_start', 50)
        self.max_negative_rewards = kwargs.get('max_negative_rewards', 25)
        self.is_learn = kwargs.get('is_learn', True)
        self.is_eval = kwargs.get('is_eval', True)
        self.evaluate_interval = kwargs.get('evaluate_interval', 100)
        self.save_interval = kwargs.get('save_interval', 100)
        self.stop_ratio = kwargs.get('stop_ratio', 1.6)
        self.seed = kwargs.get('seed', None)
        # 評価パラメータ
        self.evaluation_max_episodes = kwargs.get('evaluation_max_episodes', 5)
        self.evaluation_max_frames = kwargs.get('evaluation_max_frames', 1000)
        self.evaluation_epsilon = kwargs.get('evaluation_epsilon', 0.4)
        self.evaluation_seed = kwargs.get('evaluation_seed', 1)
        self.evaluation_max_negative_rewards = kwargs.get('evaluation_max_negative_rewards', 25)
        self.show_history = kwargs.get('show_history', True)
        # アニメーションパラメータ
        self.animation_fps = kwargs.get('animation_fps', 30)
        self.animation_epsilon = kwargs.get('animation_epsilon', 0.0)
        self.animation_wait_key_time = kwargs.get('animation_wait_key_time', 0.2)

        # 前ステップの累計報酬
        self.previous_total_reward = 0

        # 最初のフレームを無視する
        self.skip_starting_frame = False
        # 学習中のフレームを無視する
        self.skip_learning_frame = False
        # アニメーションのみ
        self.animation_only = kwargs.get('animation_only', False)

    def simulate(self):
        """
        シミュレーション
        """
        # 乱数の初期化
        # アニメーション時に評価の環境を再現するために必要
        if self.seed is not None:
            random.seed(self.seed)

        start_time = time.time()
        weight_file_name = self.get_weight_file_name()

        print(f'開始フレームのスキップ: 1〜{self.skip_frames_from_start}フレームをスキップします...')
        print(f'学習中のステップ: {self.skip_frames}フレーム')

        #
        # ここから学習シミュレーション開始
        #
        for episode in range(self.start_episode, self.max_episodes + 1):
            # 環境のリセット
            obs = self.env.reset()
            # 総フレーム数
            counter = 0
            # 学習のフレーム数
            learning_counter = 0
            # 総報酬
            total_reward = 0
            # マイナスの報酬を得た回数
            negative_counter = 0
            # 学習中のフレームをスキップしている現在の回数
            skip_count = 0
            # フレームのリスト
            frames = []

            #
            # エピソード
            #
            while True:
                # アニメーション描画
                if self.animation_only:
                # if True:
                    image = self.env.render()
                    frames.append(image)
                    window_name = f'trainer_{episode}'
                    cv2.imshow(window_name, image)
                    cv2.moveWindow(window_name, 100, 100)
                    key = cv2.waitKey(int(self.animation_wait_key_time * 1000))
                    if key == ord('n'):
                        break

                # 最初のフレームを無視(ズームがあるため)
                if counter < self.skip_frames_from_start:
                    self.skip_starting_frame = True
                else:
                    self.skip_starting_frame = False

                # 学習中のフレームをスキップ
                if not self.skip_starting_frame:
                    if skip_count < self.skip_frames:
                        skip_count += 1
                        self.skip_learning_frame = True
                    else:
                        skip_count = 0
                        self.skip_learning_frame = False

                # エージェントが行動を選ぶ
                action_id, action_resource = self.agent.select_action(obs)
                # if self.skip_starting_frame:
                #     action_id, action_resource = -1, 'no_action'
                action = self.env.get_action(action_id)

                # 環境が報酬と次の観測を決める
                reward, done, next_obs, breaking = self.env.step(action_id)
                # 過去の報酬と合算
                total_reward += reward

                # 学習をスキップ
                if self.skip_starting_frame or self.skip_learning_frame or self.animation_only:
                    pass
                # エージェントが学習する
                else:
                    if self.is_learn:
                        # frame_counterが100回を越え、かつtotal_rewordが0以下、かつrewardがマイナスの場合、
                        # negative_reward_counterをカウントアップ。
                        # 正の報酬が1回でも入ればnegative_reward_counterをクリア。
                        if learning_counter >= 100:
                            if total_reward < 0:
                                if reward < 0:
                                    negative_counter = negative_counter + 1
                                else:
                                    negative_counter = 0
                            else:
                                negative_counter = 0

                        # 学習
                        self.agent.learn(obs, action_id, reward, done, next_obs)
                        learning_counter += 1

                # next_obs, next_done を次の学習のために保持
                obs = next_obs
                counter += 1

                #
                # 報酬を出力
                #
                if self.animation_only:
                    self.print_console(
                        overwrite=False,
                        episode=None,
                        counter=counter,
                        action=action,
                        learning_counter=None,
                        reward=reward,
                        total_reward=total_reward,
                        negative_counter=0,
                        action_resource=action_resource,
                        breaking=breaking,
                    )
                else:
                    # 開始のスキップ中
                    if self.skip_starting_frame:
                        self.print_console(
                            overwrite=False,
                            episode=episode,
                            counter=counter,
                            action=action,
                            learning_counter=None,
                            reward=reward,
                            total_reward=total_reward,
                            negative_counter=0,
                            action_resource=action_resource,
                            breaking=breaking,
                        )
                    # 学習中のスキップ
                    elif self.skip_learning_frame:
                        pass
                    # 学習中
                    else:
                        self.print_console(
                            overwrite=False,
                            episode=episode,
                            counter=counter,
                            learning_counter=learning_counter,
                            action=action,
                            reward=reward,
                            total_reward=total_reward,
                            negative_counter=negative_counter,
                            action_resource=action_resource,
                            breaking=breaking,
                        )

                #
                # エピソードの終了判定
                #
                if self.is_done(done, counter, negative_counter):
                    break

            # フレームをビデオとして保存
            file_name = f'{weight_file_name}_no_randomize_ep{episode}'
            clip = mpy.ImageSequenceClip(frames, fps=30)
            clip.write_videofile(f"./video/{file_name}.mp4") #, codec="libx264")

            # コンソールの改行に必要
            print('')
            # ウィンドウを閉じる
            cv2.destroyAllWindows()
            cv2.waitKey(10)

            #
            # 一定のステップ数で記録と評価と表示を行う
            #
            if self.is_eval is True:
                if episode % self.evaluate_interval == 0:
                    # 評価を行う (C)
                    eval_rewards, eval_steps = self._evaluation(
                        self.evaluation_max_episodes,
                        self.evaluation_max_frames,
                        self.evaluation_max_negative_rewards,
                        self.evaluation_epsilon,
                        self.evaluation_seed,
                    )

                    # 記録 (D)
                    self.history_rewards.append(eval_rewards)
                    self.history_steps.append(eval_steps)
                    self.history_time.append(
                        self.history_start_time + counter)

                    # 途中経過表示
                    if self.show_history:
                        ptime = int(time.time() - start_time)
                        msg = f'{self.history_start_time + counter} steps, ' \
                              + f'{ptime} sec --- ' \
                              + f'rewards/episode {eval_rewards: .2f}, ' \
                              + f'steps/episode {eval_steps: .2f}'
                        print(msg)

                    # 評価eval_rwdsが指定値eary_stopよりも上だったら終了
                    # if self.stop_ratio is not None:
                    #     if eval_rewards > self.stop_ratio:
                    #         msg = f'eary_stop eval_reward {eval_rewards:.2f}' \
                    #               + f' > th {self.stop_ratio:.2f}'
                    #         print(msg)
                    #         break

            #
            # モデルを保存
            #
            if episode % self.save_interval == 0:
                file_name = f'{weight_file_name}_no_randomize_ep{episode}'
                self.agent.save_weight(file_name)

        print('全てのエピソードを終了.')

    def is_done(self, done, counter, negative_reward_counter) -> bool:
        ret = False
        if done:
            print(f'ゲームが終了しました. ({done=})')
            ret = True
        if negative_reward_counter > self.max_negative_rewards:
            print(f'負の報酬を連続{self.max_negative_rewards}回取得しました.')
            ret = True
        if self.max_frames is not None:
            if counter >= self.max_frames:
                # print(f'フレーム数が上限{max_frames}を越えました.')
                ret = True

        return ret

    def get_weight_file_name(self):
        now = datetime.datetime.now()
        file_name = now.strftime('%Y%m%d-%H%M%S')
        return file_name

    def print_console(
            self,
            overwrite,
            episode,
            counter,
            learning_counter,
            action,
            reward,
            total_reward,
            negative_counter,
            action_resource,
            breaking
    ):
        """
        コンソールへの出力
        """
        cr = ''
        end = os.linesep
        # 同じ行に出力
        if overwrite:
            cr = '\r'
            end = ''

        increased = ''
        if self.previous_total_reward < total_reward:
            increased = 'Increased!'
        self.previous_total_reward = total_reward

        breaking_notion = ''
        if breaking:
            breaking_notion = 'Breaking!'

        negative_notion = ''
        if negative_counter > 0:
            negative_notion = f'{negative_counter=},'

        if self.animation_only:
            print(
                f'{cr}animation only... '
                f'{counter=}, '
                f'{reward=:.2f}, '
                f'{total_reward=:.2f}, '
                f'{action_resource}, '
                f'{action=}, '
                f'{breaking_notion} '
                f'{increased}',
                end=end
            )
            return

        if self.skip_starting_frame:
            print(
                f'{cr}skipping... '
                f'{episode=}, '
                f'{counter=}',
                end=end
            )
        else:
            print(
                f'{cr}learning... '
                f'{episode=}, '
                f'{counter=}, '
                f'{learning_counter=}, '
                f'{reward=:.2f}, '
                f'{total_reward=:.2f}, '
                f'{action_resource}, '
                f'{action=}, '
                f'{negative_notion} '
                f'{breaking_notion} '
                f'{increased}',
                end=end
            )

    def _evaluation(
            self,
            evaluation_max_episodes,
            evaluation_max_frames,
            evaluation_max_negative_rewards,
            evaluation_epsilon,
            evaluation_seed,
    ):
        """ 評価 """
        if evaluation_seed is not None:
            np_random_state = np.random.get_state()
            random_state = random.getstate()
            np.random.seed(evaluation_seed)
            random.seed(evaluation_seed)

        # 乱雑状態のバックアップ
        epsilon_backup = self.agent.epsilon
        # 評価用の乱雑度を設定
        self.agent.epsilon = evaluation_epsilon

        # 記録クラス初期化
        self.recorder.reset()

        # 評価シミュレーション
        for episode in range(1, evaluation_max_episodes + 1):
            # 評価シミュレーションの準備
            obs = self.eval_env.reset()
            frame_counter = 1
            total_reward = 0
            negative_reward_counter = 0

            while True:
                # エージェントが行動を選ぶ
                act = self.agent.select_action(obs)

                # 環境が報酬と次の観測を決める
                reward, done, obs = self.eval_env.step(act)
                total_reward += reward

                # 報酬を記録
                self.recorder.add(reward, done)

                negative_reward_counter = negative_reward_counter + 1 if frame_counter >= 100 and total_reward < 0 else 0

                # 報酬を出力
                print(f'\revaluating... {episode=}, {frame_counter=}, {reward=:.2f}, {total_reward=:.2f}', end='')

                #
                # エピソード中の終了判定
                #
                frame_counter += 1
                if done:
                    # print('')
                    # print('エピソードが正常に終了しました.')
                    break
                if negative_reward_counter > evaluation_max_negative_rewards:
                    # print('')
                    # print(f'評価中のnegative_reward_counterが上限{evaluation_max_negative_rewards}を越えました.')
                    break
                if evaluation_max_frames is not None:
                    if frame_counter >= evaluation_max_frames:
                        # print('')
                        # print(f'評価中のframeが上限{evaluation_max_frames}を越えました.')
                        break

            print('')

        # 乱雑度の復元
        self.agent.epsilon = epsilon_backup

        # 結果の集計
        mean_rewards, mean_steps = self.recorder.get_result()

        # 乱数状態の復元
        if evaluation_seed is not None:
            np.random.set_state(np_random_state)
            random.setstate(random_state)

        return mean_rewards, mean_steps

    # def _show_Q(self):
    #     """ Q値の表示 """
    #     if self.obss is None:
    #         return
    #
    #     print('')
    #     print('学習後のQ値')
    #     for obs in self.obss:
    #         q_vals = self.agent.get_q(np.array(obs))
    #         if q_vals is not None:
    #             valstr = [f' {v: .2f}' for v in q_vals]
    #             valstr = ','.join(valstr)
    #             print('{}:{}'.format(str(np.array(obs)), valstr))

    def save_history(self, pathname):
        """ 学習履歴を保存 """
        np.savez(
            pathname + '.npz',
            eval_rwds=self.history_rewards,
            eval_steps=self.history_steps,
            eval_x=self.history_time,
        )

    def load_history(self, pathname):
        """ 学習履歴を読み込み """
        hist = np.load(pathname + '.npz')
        self.history_rewards = hist['eval_rwds'].tolist()
        self.history_steps = hist['eval_steps'].tolist()
        self.history_time = hist['eval_x'].tolist()
        self.history_start_time = self.history_time[-1]
