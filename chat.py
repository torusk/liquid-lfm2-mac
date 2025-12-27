import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# モデルID
model_id = "LiquidAI/LFM2-2.6B-Exp"

def main():
    print(f"Loading {model_id} ...")
    print("※初回はダウンロードに時間がかかります。")

    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # モデルの読み込み (M4 Mac向け最適化設定)
    # MPS環境では bfloat16 より float16 の方が高速・安定するため採用
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to("mps")

    print("\n=== 準備完了 (終了するには 'exit' と入力) ===")

    messages = []

    while True:
        try:
            user_input = input("\nあなた: ")
            if user_input.lower() == "exit":
                break

            messages.append({"role": "user", "content": user_input})
            
            # 1. データをGPU向けに作成
            # return_dict=True を指定して明示的に辞書形式を取得
            inputs = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt",
                return_dict=True
            ).to(model.device)

            # 2. 生成実行
            # transformersのバージョンによる戻り値の違い(辞書型/Tensor)を吸収
            if isinstance(inputs, dict) or hasattr(inputs, "input_ids"):
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=1024,  # 長文も切れないように少し多めに設定
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                outputs = model.generate(
                    inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )

            # 3. 応答のデコード (入力部分をスキップして応答だけを取り出す)
            if isinstance(inputs, dict) or hasattr(inputs, "input_ids"):
                input_length = inputs["input_ids"].shape[1]
            else:
                input_length = inputs.shape[1]

            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            print(f"Liquid: {response}")

            messages.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n中断しました。")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            continue

if __name__ == "__main__":
    main()