# CarRacing
a car racing reinforcement learning.

## パッケージのインストール

```
pip install numpy opencv-python gymnasium gymnasium[box2d] tensorflow matplotlib
```

## コマンド

### 学習する

```
python main.py learn
```

### 保存済みのウェイトをロードして学習する

```
python main.py learn [weight_file_name].keras
```

### 学習結果をゲーム画面で確認する

```
python main.py play [weight_file_name].keras
```
