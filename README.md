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

### 自然な間違い生成（指標 + 自己改善）
各差分生成では、候補編集を複数回試行し、自然さスコアが最も高い候補を採用します。

自然さスコアの構成:
- `mean_abs_diff`: 元領域と編集領域の平均差分（小さすぎても大きすぎても減点）
- `edge_delta`: エッジ構造差分（構造破壊が大きいほど減点）
- `change_score`, `edge_score`: 上記を 0-1 に正規化した部分スコア
- `naturalness_score`: 最終自然さスコア（0-1）

自己改善ループ:
- 難易度別に試行回数を設定（easy=3, medium=5, hard=7）
- 各試行で編集強度を自動調整して自然さを改善
- `difference_cards` の `improvement_attempts` に採用時の試行回数を保存

難易度調整（特徴量サイズ）:
- 難易度ごとに差分領域サイズ倍率と初期編集強度を分離
- `easy` は大きめ・強め、`hard` は小さめ・弱めに設定
- `score_breakdown.target_change` で目標差分量を記録し、過小な差分を抑制

環境変数での調整:
- 再デプロイ不要で難易度プロファイルを変更できるよう、以下の環境変数をサポート
- 例:

```bash
export DIFF_EASY_SIZE_MULTIPLIER=1.30
export DIFF_MEDIUM_SIZE_MULTIPLIER=1.10
export DIFF_HARD_SIZE_MULTIPLIER=1.00

export DIFF_EASY_INITIAL_STRENGTH=1.35
export DIFF_MEDIUM_INITIAL_STRENGTH=1.20
export DIFF_HARD_INITIAL_STRENGTH=1.05

export DIFF_EASY_TARGET_CHANGE=0.20
export DIFF_MEDIUM_TARGET_CHANGE=0.15
export DIFF_HARD_TARGET_CHANGE=0.11

export DIFF_EASY_ATTEMPTS=4
export DIFF_MEDIUM_ATTEMPTS=6
export DIFF_HARD_ATTEMPTS=8
```

境界なじみ改善:
- 幾何編集を含む全編集で、貼り付け時にフェザーブレンド（境界ぼかし合成）を適用
- 自然さ評価もブレンド後の見た目で実施し、境界が不自然な候補は採用されにくくする
- `score_breakdown.feather_radius` に適用半径を記録

矩形貼り付けの不自然さ対策:
- 編集領域は矩形全体ではなく、不定形のソフトマスク（楕円・角丸形状）で適用
- 境界フェザーと不定形マスクを合成して、四角い境界線が残らないようにする
- `score_breakdown.mask_coverage` で実際に変化した領域率を追跡

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

### トラブルシュート
- Swagger UIで 422 が返る場合:
	- `POST /generate` は `image` が必須です。Try it out で画像ファイルを選択してください。
	- 422の典型例: `{"detail":[{"loc":["body","image"],"msg":"Field required"...}]}`
- Swaggerの `escaping deep link whitespace` 警告:
	- API機能には影響しないUI警告です。
	- 本実装では deep linking を無効化して警告を抑制しています。

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