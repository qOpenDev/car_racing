"""
共通で使う関数
"""
import json
import numpy as np
import matplotlib.pyplot as plt


def copy_img(img_back, img_obj, x, y, isTrans=False):
    """
    img_back にimg_objをコピーする

    Parameters
    ----------
    img_back: 3d numpy.ndarray
        背景画像
    img_obj: 3d numpy.ndarray
        コピーする物体の画像
    x, y: int
        img 上で張り付ける座標
    isTrans: bool
        True: 白(255, 255, 255)を透明にする

    Returns
    -------
    img_back2: 3d numpy.ndarray
        コピー後の画像

    """

    # 引数のimg_backとimg_objが書き変わらないようにコピーする (A)
    img_obj2 = img_obj.copy()
    img_back2 = img_back.copy()
    h, w, _ = img_obj2.shape

    if isTrans is True:
        # img_obj2の白領域を透明にする処理 (B)
        # img_obj2の白領域に背景画像をコピーする
        idx = np.where((img_obj2 == (255, 255, 255)).all(axis=2))
        img_back_rect = img_back[y:y+h, x:x+w, :].copy()
        img_obj2[idx] = img_back_rect[idx]

    # img_obj2をimg_back2にコピー(C)
    img_back2[y:y+h, x:x+w, :] = img_obj2
    return img_back2


def show_graph(pathname, target_reward=None, target_step=None):
    """
    学習曲線の表示

    Parameters
    ----------
    target_reward: float or None
        rewardの目標値に線を引く
    target_step: float or None
        stepの目標値に線を引く
    """
    hist = np.load(pathname + '.npz')
    eval_rwd = hist['eval_rwds'].tolist()
    eval_step = hist['eval_steps'].tolist()
    eval_x = hist['eval_x'].tolist()

    plt.figure(figsize=(8, 4))
    plt.subplots_adjust(hspace=0.6)

    # reward / episode
    plt.subplot(211)
    plt.plot(eval_x, eval_rwd, 'b.-')
    if target_reward is not None:
        plt.plot(
            [eval_x[0], eval_x[-1]],
            [target_reward, target_reward],
            'r:')

    plt.title('rewards / episode')
    plt.grid(axis='both')

    # steps / episode
    plt.subplot(212)
    plt.plot(eval_x, eval_step, 'b.-')
    if target_step is not None:
        plt.plot(
            [eval_x[0], eval_x[-1]],
            [target_step, target_step],
            'r:')
    plt.title('steps / episode')
    plt.xlabel('steps')
    plt.grid(axis='both')

    plt.show()


# 以下グリッドサーチでのみ使用
def save_json(filename, data):
    """ dict型変数を json形式で保存 """
    with open(filename, mode='wt', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(filename):
    """ json形式のファイルをdict型変数に読み込み """
    with open(filename, mode='rt', encoding='utf-8') as file:
        data = json.load(file)
    return data
