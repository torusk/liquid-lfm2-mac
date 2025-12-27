# Liquid AI LFM2-2.6B-Exp for macOS (Apple Silicon)

Liquid AIが開発した最新の軽量言語モデル **LFM2-2.6B-Exp** を、Apple Silicon (M1/M2/M3/M4) 搭載のMac上で実行するためのスクリプトです。

PyTorchのMPS（Metal Performance Shaders）アクセラレーションを利用し、M4チップに最適化された `float16` 精度で高速に動作します。

## 動作環境

- **Machine**: Mac mini (M4) / MacBook Air/Pro (M1以降)
- **OS**: macOS Sequoia / Sonoma
- **Python**: 3.10 以上

## セットアップ手順

### 1. リポジトリのクローン
```bash
git clone https://github.com/torusk/liquid-lfm2-mac.git
cd liquid-lfm2-mac
```

### 2. ライブラリのインストール
必要なライブラリを一括でインストールします。
```bash
pip install -r requirements.txt
```

### 3. Hugging Faceへのログイン
モデルをダウンロードするために、Hugging Faceのアクセストークン（Read権限）が必要です。
```bash
python -c "from huggingface_hub import login; login()"
```

## 使い方

以下のコマンドでチャットを開始します。`HF_HUB_ENABLE_HF_TRANSFER=1` を付けることで、高速ダウンロードモードで起動します。

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python chat.py
```

- **終了方法**: チャット入力欄で `exit` と入力するか、`Ctrl + C` を押します。

## トラブルシューティング

キャッシュエラーなどでダウンロードが止まる場合は、以下のコマンドでキャッシュを削除して再試行してください。
```bash
python -m huggingface_hub.cli delete-cache
```

## ライセンス (License)

### コード (This Repository)
本リポジトリに含まれるスクリプト（chat.py）は **MIT License** の下で公開されています。

### モデル (Model)
使用しているモデル `LiquidAI/LFM2-2.6B-Exp` のライセンスは、Liquid AI社の規定に従います。詳細は公式モデルカードをご確認ください。

- **Model**: [LiquidAI/LFM2-2.6B-Exp](https://huggingface.co/LiquidAI/LFM2-2.6B-Exp)
- **License**: Liquid AI Community License (See the model card for details)
