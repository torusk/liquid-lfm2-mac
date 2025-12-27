import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ==========================================
# 設定パラメータ (好みに合わせて調整してください)
# ==========================================
model_id = "LiquidAI/LFM2-2.6B-Exp"

# 生成時のパラメータ
GEN_CONFIG = {
    "max_new_tokens": 1024,  # 応答の最大文字数（トークン数）。長文にしたい場合は増やしてください。
    "temperature": 0.7,      # 創造性。高いほど独創的（ランダム）、低いほど堅実な回答になります。
    "do_sample": True,       # Trueでランダムサンプリングを有効化。
    "top_p": 0.9,            # 累積確率がこれに達するまでの上位トークンのみを考慮。
}

# 実行デバイスの設定
# Apple Silicon Macでは "mps" を指定することでGPU加速（Metal）が効きます。
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def main():
    print(f"Loading {model_id} ...")
    print("-" * 50)
    print("【ヒント】")
    print("・モデルは約5GBのダウンロードが必要です（初回のみ）。")
    print("・一度ダウンロードが終われば、オフライン（ネットなし）でも動作します。")
    print("-" * 50)

    # トークナイザー（文字と数字を変換する辞書）の読み込み
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # モデル本体の読み込み
    # torch_dtype=torch.float16: M4 Mac等で最も効率よく動く浮動小数点精度を指定
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(DEVICE)

    print(f"\n✅ 準備完了！ (デバイス: {DEVICE})")
    print("終了するには 'exit' と入力、または Ctrl + C を押してください。")

    # チャット履歴を保持するリスト
    messages = []

    while True:
        try:
            user_input = input("\n👤 あなた: ")
            if not user_input.strip():
                continue
            
            if user_input.lower() in ["exit", "quit", "終了"]:
                break

            # ユーザーの入力を履歴に追加
            messages.append({"role": "user", "content": user_input})
            
            # --- 1. デバイス向けに入力データを準備 ---
            # apply_chat_template: モデルが理解しやすい対話形式のプロンプトに変換
            inputs = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt",
                return_dict=True
            ).to(model.device)

            # --- 2. AIの応答生成 ---
            # **GEN_CONFIG を展開して引数に渡しています
            outputs = model.generate(
                **inputs, 
                **GEN_CONFIG,
                pad_token_id=tokenizer.eos_token_id
            )

            # --- 3. 応答のデコード（数値から文字へ変換） ---
            # 入力部分の長さ（input_length）をスキップして、新しく生成された部分のみを取り出す
            input_length = inputs["input_ids"].shape[1]
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            print(f"\n💧 Liquid AI: {response}")

            # AIの応答も履歴に保存（これによって過去の文脈を理解できます）
            messages.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n\nプログラムを終了します。")
            break
        except Exception as e:
            print(f"\n⚠️ エラーが発生しました: {e}")
            continue

if __name__ == "__main__":
    main()