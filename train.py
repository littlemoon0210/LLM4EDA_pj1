# -*- coding: utf-8 -*-
import os
import wandb
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from model import get_model
from dataset import get_dataset_and_tokenizer

# trainer.py: 启动训练流程并监控进度

# 设置离线模式，不尝试联网到 wandb
os.environ["WANDB_MODE"] = "offline"
# HF_DATASETS_CACHE 建议在运行脚本前通过环境变量统一设置

# WandB 初始化
wandb.init(
    project="gpt2-small-train",
    name="bookcorpus-bsz2-epoch3",
    config={
        "model": "GPT2-small",
        "dataset": "small",
        "epochs": 3,
        "batch_size": 2
    }
)

print("[DEBUG] Start training script")
checkpoint_dir = "./logs/checkpoint-last"
if os.path.exists(checkpoint_dir):
    print(f"[DEBUG] Found checkpoint: {checkpoint_dir}. Will resume from checkpoint.")
    resume_ckpt = checkpoint_dir
else:
    print("[DEBUG] No checkpoint found. Training will start from scratch.")
    resume_ckpt = None

# 加载 tokenizer 和 数据集（传入本地路径，可修改）
tokenizer, train_dataset, val_dataset = get_dataset_and_tokenizer(
    dataset_name="small",
    dataset_path="./small-117M/",
    debug=True
)

# Debug: 数据集大小输出
total_train_blocks = len(train_dataset)
total_val_blocks = len(val_dataset)
print(f"[DEBUG] Number of train blocks: {total_train_blocks}")
print(f"[DEBUG] Number of validation blocks: {total_val_blocks}")

# Debug: 展示一个样本形状
if total_train_blocks > 0:
    sample = train_dataset[0]
    print(f"[DEBUG] Sample keys: {list(sample.keys())}")
    print(f"[DEBUG] input_ids length: {len(sample['input_ids'])}")
    print(f"[DEBUG] attention_mask length: {len(sample['attention_mask'])}")

# 构建模型
model = get_model(vocab_size=tokenizer.vocab_size)
print(f"[DEBUG] Model initialized with vocab_size={tokenizer.vocab_size}")

# 训练参数，兼容当前 transformers 版本
training_args = TrainingArguments(
    output_dir="./logs",
    overwrite_output_dir=False,            # 保留已有 checkpoint 以支持续训
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    logging_steps=100,
    fp16=True,
    report_to="wandb",
)
print(f"[DEBUG] TrainingArguments: {training_args}")


# Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer 初始化
t = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
print("[DEBUG] Trainer initialized")

# 启动训练
if resume_ckpt:
    print(f"[DEBUG] Starting training from checkpoint {resume_ckpt}")
    t.train(resume_from_checkpoint=resume_ckpt)
else:
    print("[DEBUG] Starting training from scratch")
    t.train()

# 保存最终模型
t.save_model("./logs/final_model")
print("[DEBUG] Training complete! Model saved to ./logs/final_model")
