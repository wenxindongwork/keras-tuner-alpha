from trainer import FSDPTrainer
import keras
from datasets import load_dataset
import keras_hub
from transformers import AutoTokenizer

if __name__ == "main":
    # Log TPU device information
    devices = keras.distribution.list_devices()
    print(f"Available devices: {devices}")

    # use bf16 training
    keras.mixed_precision.set_global_policy("mixed_bfloat16")

    # Load HF dataset
    hf_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    hf_dataset = hf_dataset.batch(batch_size=8)

    model_id = "google/gemma-2-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id, pad_token="<pad>")

    gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(
        f"hf://google/gemma-2-2b", preprocessor=None
    )
    gemma_lm.backbone.enable_lora(rank=4)
    optimizer = keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)

    trainer = FSDPTrainer(
        gemma_lm,
        hf_dataset,
        optimizer,
        tokenizer,
        seq_len=2048,
        steps=100,
        log_steps=1,
        input_field="text",
    )

    trainer.train()

    # Test after tuning
    pred = trainer.generate("What is your name?")
    print("after tuning:", pred)

