# 間違い探しを生成するシステム
最新の画像処理技術を用いて入力画像から間違い探しを生成する。
このリポジトリでは、バックエンドシステムをPython, FastAPIで提供する。

## 環境
- Python 3.10以上
- FastAPI
- その他必要なライブラリは`requirements.txt`に記載　

## DeepLabv3+ 実験実装
`experiment/deeplabv3plus_experiment.py` に、`segmentation-models-pytorch` の `DeepLabV3Plus` を使った実験用 CLI を追加しています。

### データ配置
画像とマスクを同名ファイルで以下のように配置します。

```text
data/
	train/
		images/
		masks/
	valid/
		images/
		masks/
```

### 学習
```bash
uv run python experiment/deeplabv3plus_experiment.py train \
	--train-images data/train/images \
	--train-masks data/train/masks \
	--valid-images data/valid/images \
	--valid-masks data/valid/masks \
	--output-dir experiment/runs/deeplabv3plus \
	--num-classes 1 \
	--image-size 512 \
	--epochs 10 \
	--batch-size 2
```

### 推論
```bash
uv run python experiment/deeplabv3plus_experiment.py predict \
	--checkpoint experiment/runs/deeplabv3plus/best.pt \
	--input sample.jpg \
	--output experiment/runs/deeplabv3plus/pred_mask.png
```

- 2値セグメンテーションは `--num-classes 1` を使用。
- 多クラスの場合はマスク画像の画素値を `0..num_classes-1` に揃えて `--num-classes N` を指定。
- 低VRAM環境で `--batch-size 1` を使う場合は、スクリプト内部で BatchNorm を凍結して学習します（実験向けの安定化処理）。