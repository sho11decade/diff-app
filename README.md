# 間違い探しを生成するシステム
最新の画像処理技術を用いて入力画像から間違い探しを生成する。
このリポジトリでは、バックエンドシステムをPython, FastAPIで提供する。

## 環境
- Python 3.10以上
- FastAPI
- その他必要なライブラリは`requirements.txt`に記載　

## APIサーバー

### 起動
```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### エンドポイント
- `GET /health`: ヘルスチェック
- `POST /generate`: 間違い探し画像の生成
- `GET /experiments/{trace_id}`: traceログの取得

`POST /generate` の入力:
- `image`: 画像ファイル（JPEG/PNG, 5MB以下）
- `num_differences`: 間違いの数（1-10, 省略時3）
- `difficulty`: 難易度（`easy` / `medium` / `hard`, 省略時`medium`）
- `seed` (Query): 乱数シード（同一入力＋同一seedで再現）
- `trace` (Query): `true` で差分カードとログ出力を有効化
- `X-API-Key` ヘッダー: APIキー（デフォルトは `dev-api-key`）

実行例:
```bash
curl -X POST "http://localhost:8000/generate?seed=42&trace=true" \
	-H "X-API-Key: dev-api-key" \
	-F "image=@sample.jpg" \
	-F "num_differences=3" \
	-F "difficulty=medium"
```

返却値:
- `puzzle_image_base64`: 間違い探し画像（PNG, Base64）
- `answer_image_base64`: 答え画像（PNG, Base64）
- `positions`: 間違い箇所の座標配列
- `processing_time_ms`: 生成処理時間（ms）
- `artifact_dir`: リクエストごとの保存先ディレクトリ
- `trace_id`: リクエストごとの実験ID
- `difference_cards`: trace有効時の編集根拠データ
- `trace_log_path`: trace有効時のログ保存先（保存成功時）

### 検証用アーティファクト保存
全リクエストで、`artifact_dir` に処理過程データを保存します。

保存内容:
- `params.json`: 入力パラメータ、差分座標、差分カード、処理時間
- `source.png`: 入力画像
- `puzzle.png`: 生成後画像
- `answer.png`: 答え画像
- `step_00_source.png`: 処理開始時点
- `step_XX_<edit_type>.png`: 各差分適用後の中間画像

保存先例:
```text
experiments/
	20260320/
		20260320T120000Z-a1b2c3d4/
			params.json
			source.png
			puzzle.png
			answer.png
			step_00_source.png
			step_01_color.png
			step_02_flip.png
```

## DeepLabv3+ 実験実装
`experiment/deeplabv3plus_experiment.py` に、`segmentation-models-pytorch` の `DeepLabV3Plus` を使った実験用 CLI を追加しています。

## 研究アーキテクチャ提案
- システム全体の研究志向アーキテクチャは `docs/研究アーキテクチャ提案.md` を参照。
- 要件定義 (`docs/要件定義.md`) と技術調査 (`docs/deep-research-report.md`) を統合し、説明可能性・再現性・難易度制御を重視した設計を整理。

## 実装計画
- 具体的なマイルストーン・WBS・完了条件は `docs/実装計画.md` を参照。

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