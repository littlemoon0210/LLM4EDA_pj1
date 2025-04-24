# -*- coding: utf-8 -*-
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import os
import itertools

# dataset.py: 负责加载并预处理数据集，支持本地文本文档(.txt)或JSON(.json/.jsonl)文件，以及 HuggingFace Hub 数据集

def get_dataset_and_tokenizer(
    dataset_name: str = "small",
    dataset_path: str = "./small-117M/",
    val_split: float = 0.05,
    block_size: int = 128,
    debug: bool = False
):
    # 使用 GPT2TokenizerFast 加速分词
    tokenizer = GPT2TokenizerFast.from_pretrained("./gpt2_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载原始数据集
    if dataset_path:
        data_files = {"train": os.path.join(dataset_path, "*")}
        if dataset_path.endswith(('.json', '.jsonl')):
            dataset = load_dataset("json", data_files=data_files, split="train")
        else:
            dataset = load_dataset("text", data_files=data_files, split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")

    # 确定文本字段名，在整个模块作用域使用
    if "text" in dataset.column_names:
        text_key = "text"
    else:
        # 取第一个字段作为文本字段
        text_key = dataset.column_names[0]

    # 分词函数
    def tokenize_function(examples):
        return tokenizer(
            examples[text_key],
            truncation=True,
            max_length=512
        )

    # 批量分词，并移除原始列
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[c for c in dataset.column_names],
        num_proc=os.cpu_count(),
        desc="Tokenizing dataset"
    )

    # Debug 输出示例长度
    if debug and len(tokenized) > 0:
        sample = tokenized[0]
        print("[DEBUG] Sample keys:", sample.keys())
        print(f"[DEBUG] input_ids len={len(sample['input_ids'])}, mask len={len(sample['attention_mask'])}")

    # 划分训练/验证集
    split = tokenized.train_test_split(test_size=val_split, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    # 按块分割函数
    def group_texts(examples):
        # 扁平化所有序列
        all_ids = list(itertools.chain.from_iterable(examples["input_ids"]))
        all_mask = list(itertools.chain.from_iterable(examples["attention_mask"]))
        # 丢弃多余长度
        total_len = (len(all_ids) // block_size) * block_size
        # 切分块
        input_ids = [all_ids[i : i + block_size] for i in range(0, total_len, block_size)]
        attention_mask = [all_mask[i : i + block_size] for i in range(0, total_len, block_size)]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    # 应用分块，并移除旧列，避免长度不匹配
    train_dataset = train_dataset.map(
        group_texts,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=os.cpu_count(),
        desc="Grouping train set"
    )
    val_dataset = val_dataset.map(
        group_texts,
        batched=True,
        remove_columns=val_dataset.column_names,
        num_proc=os.cpu_count(),
        desc="Grouping val set"
    )

    # Debug 输出块数
    if debug:
        print(f"[DEBUG] Train blocks: {len(train_dataset)}, Val blocks: {len(val_dataset)}")

    return tokenizer, train_dataset, val_dataset
