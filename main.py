"""
main_field.py
フィールドタスクの実行ファイル
"""
import os
import sys
import datetime

# 自作モジュール
from environment import Environment
from trainer import Trainer
from agent import TargetQAgent as Agt
import myutil


def get_params(args: list):
    if len(args) < 2:
        msg = 'パラメータを指定してください.'
        print(msg)
        sys.exit()

    process_type = ''
    weight_file_name = None
    play_video = False
    if len(args) == 2:
        process_type = args[1]
    elif len(args) == 3:
        process_type = args[1]
        weight_file_name = args[2]
    elif len(args) == 4:
        process_type = args[1]
        play_video = True
        weight_file_name = args[3]
    return process_type, play_video, weight_file_name


def get_start_episode(full_file_name: str):
    if animation_only:
        return 1
    else:
        if full_file_name:
            splitted_file_names = full_file_name.split(os.sep)
            file_name = splitted_file_names[-1]
            temp_file_name, extension = file_name.split(os.extsep)
            date_string, episode_string = temp_file_name.split('_')
            episode = episode_string.replace('ep', '')
            return int(episode) + 1
        else:
            return 1


args = sys.argv

animation_only = False
load_weight = False
learning = False
show_graph = True

# パラメータを取得
process_type, play_video, weight_file_name = get_params(args)

# process_typeでパラメータをセット
if process_type == 'learn':
    # 学習済みファイルをロード
    if weight_file_name:
        load_weight = True
        learning = True
    # さらの状態で学習開始
    else:
        learning = True
elif process_type in ('graph', 'G'):
    show_graph = True
elif process_type == 'play':
    load_weight = True
    animation_only = True
else:
    print('process type が間違っています。')
    sys.exit()

start_episode = get_start_episode(weight_file_name)

# 環境の設定 /////////////////////////////// (A)
# Envインスタンス生成
# 学習用環境
env = Environment()
obs = env.reset()  # obsのサイズがエージェントのパラメータ設定で使われる

# 評価用環境
eval_env = Environment()

# エージェントのパラメータ設定 ///////////////
# Agt 共通パラメータ
agent_param = {
    'n_action': env.n_action,
    'input_size': obs.shape,
    'n_dense': 32,
    'replay_memory_size': 100,
    'replay_batch_size': 20,
    'target_interval': 10,
    # 割引率
    'gamma': 0.9,
    # 乱雑度
    'epsilon': 0.15,
}

# トレーナーのパラメータ設定 ////////////////
# シミュレーションパラメータ
simulation_params = {}

# 共通パラメータ
common_params = {
    # 学習する
    'is_learn': True,
    # 評価する
    'is_eval': False,
    # 開始エピソード数
    'start_episode': start_episode,
    # 最大エピソード数
    'max_episodes': 1000,
    # 評価を実施するエピソードの間隔
    'evaluate_interval': 50,
    # モデルの保存間隔
    'save_interval': 10,
    # 最大フレーム数
    'max_frames': 5000,
    # スキップするフレーム数
    'skip_frames': 0,
    # ゲーム開始からスキップするフレーム数
    'skip_frames_from_start': 50,
    # 連続で受け取る負の報酬の最大数
    'max_negative_rewards': 25,
    # 乱数シード
    'seed': None,
    # アニメーションの表示
    'animation_only': animation_only,
    # rewards/episodeがこの値を超えたら学習シミュレーション終了
    'stop_ratio': 1.6,
}
simulation_params.update(common_params)

# 評価パラメータ
evaluation_params = {
    # 最大エピソード数
    'evaluation_max_episodes': 5,
    # 最大フレーム数
    'evaluation_max_frames': 5000,
    # 連続で受け取る負の報酬の最大数
    'evaluation_max_negative_rewards': 25,
    # 乱雑度
    'evaluation_epsilon': 0.0,
    # 乱数シード
    'evaluation_seed': 222,
    # 履歴を表示する
    'show_history': True,
}
if simulation_params['is_eval']:
    simulation_params.update(evaluation_params)

# アニメーションパラメータ
animation_params = {
    # フレーム時
    'animation_fps': 30,
    # 乱雑度
    'animation_epsilon': 0.0,
    # cv2フレームのキー待ち時間(sec)
    'animation_wait_key_time': 0.2,
    # 乱数シード
    'animation_seed': 321,
}
if simulation_params['animation_only']:
    simulation_params.update(animation_params)

# グラフパラメータ
graph_params = {
    'target_reward': 1.6,
    'target_step': 22.0,
}

if (load_weight is True) or (learning is True) or (animation_only is True):
    # エージェントのインスタンス生成
    agent = Agt(**agent_param)

    # trainer インスタンス作成
    trainer = Trainer(agent, env, eval_env, **simulation_params)

    if load_weight is True:
        # エージェントのデータロード
        try:
            agent.load_weight(weight_file_name)
            # trainer.load_history(agent.file_path)
        except Exception as e:
            print(e)
            print('エージェントのパラメータがロードできません')
            sys.exit()

    if learning is True:
        # 学習
        trainer.simulate()

    if animation_only is True:
        agent.epsilon = 0.0
        trainer.simulate()

if show_graph is True:
    pass
    # グラフ表示
    # myutil.show_graph(agent_param['file_path'], **graph_params)
