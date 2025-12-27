# Liquid AI LFM2-2.6B-Exp for macOS (Apple Silicon)

Liquid AIが開発した最新の軽量言語モデル **LFM2-2.6B-Exp** を、Apple Silicon (M1/M2/M3/M4) 搭載のMac上で実行するためのスクリプトです。

### � 開発の経緯
2025/12/27現在、Hugging Faceからダウンロードしたモデルを LM Studio でロードしようとするとエラーが発生する場合があったため、ターミナルでサクッと動作確認とチャットができるように作成しました。

### �🚀 特徴
- **GPT-4級の知能**: 小型モデルながら、特定の指示遂行能力（IFEval）や推論において、GPT-4に匹敵、あるいはGPT-3.5を大きく超える性能を誇ります。
- **完全オフライン動作**: 初回のモデルダウンロード後は、インターネット環境がなくてもローカルで安全かつ高速に実行可能です。
- **Apple Siliconに最適化**: PyTorchのMPS（Metal Performance Shaders）を利用し、M4チップ等での `float16` 演算により圧倒的なパフォーマンスを発揮します。

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

### 3. Hugging Faceへのログイン（推奨）
必須ではありませんが、ログインしておくことでモデルのダウンロードをより安定・高速に行うことができます。Hugging Faceのアクセストークン（READ権限がシンプルでおすすめ）を用意して実行してください。
```bash
python -c "from huggingface_hub import login; login()"
```

## 使い方

以下のコマンドでチャットを開始します。`HF_HUB_ENABLE_HF_TRANSFER=1` を付けることで、高速ダウンロードモードで起動します。

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python chat.py
```

- **終了方法**: チャット入力欄で `exit` と入力するか、`Ctrl + C` を押します。

## カスタマイズ

`chat.py` 内の `GEN_CONFIG` を書き換えることで、応答の長さを変えたり、AIの性格（創造性）を調整したりできます。

```python
GEN_CONFIG = {
    "max_new_tokens": 1024,  # 応答の最大文字数
    "temperature": 0.7,      # 創造性（0.0〜1.0）
    "do_sample": True,
    "top_p": 0.9,
}
```

## トラブルシューティング

キャッシュエラーなどでダウンロードが止まる場合は、以下のコマンドでキャッシュを削除して再試行してください。
```bash
python -m huggingface_hub.cli delete-cache
```

## ライセンス (License)

### コード (This Repository)
本リポジトリに含まれるスクリプト（chat.py）は **MIT License** の下で公開しています。

### モデル (Model)
使用しているモデル `LiquidAI/LFM2-2.6B-Exp` のライセンスは、Liquid AI社の規定に従います。
- **Model**: [LiquidAI/LFM2-2.6B-Exp](https://huggingface.co/LiquidAI/LFM2-2.6B-Exp)
- **License**: Liquid AI Community License
