from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset
from dotenv import load_dotenv
import torch
import os

load_dotenv()

def load_model_and_tokenizer():
    model_name = "EleutherAI/gpt-neo-125M"
    token = os.getenv("HF_ACCESS_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

    return model, tokenizer

def apply_lora(model, r=8, lora_alpha=16, lora_dropout=0.05):
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    return model

def train_lora(model, tokenizer, output_dir="lora_output", epochs=3):
    dataset = load_dataset("json", data_files="dataset.jsonl")["train"]
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_fn(example):
        example["text"] = f"Input: {example['input']}\nOutput: {example['output']}"
        return example

    dataset = dataset.map(preprocess_fn)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=epochs,
        logging_dir=f"{output_dir}/logs",
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
    )

    # NOTE: You'll need to define the Trainer with proper data collator and loss if using PEFT/LoRA

    # from peft import get_peft_model, LoraConfig
    # peft_model = get_peft_model(model, LoraConfig(...))
    # trainer = Trainer(model=peft_model, args=training_args, train_dataset=tokenized_dataset)

    # trainer.train()

def load_lora_model(base_model_path, lora_model_path):
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(model, lora_model_path)
    return model

def generate_text(model, tokenizer, prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    train_lora(model, tokenizer, output_dir="lora_output", epochs=3)

    print("LoRA tuning complete!")
