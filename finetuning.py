import pandas as pd
import argparse
import warnings
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
from templates.prompt_templates import prompt_style, question

warnings.filterwarnings("ignore")

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Train a language model on a given CSV file"
)
parser.add_argument("csv_path", type=str, help="Path to the CSV file")
args = parser.parse_args()

# Load and preprocess data
df = pd.read_csv(args.csv_path, parse_dates=["publish_date"])
df["publish_date"] = df["publish_date"].dt.strftime("%Y-%m-%d")
df.sort_values(by="publish_date", inplace=True, ascending=True)
df.drop_duplicates(subset=["article"], inplace=True)
df["response"] = df["response"].str.replace("<｜end▁of▁sentence｜>", "")

# Load model and tokenizer
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

EOS_TOKEN = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,  # True or "unsloth" for very long context
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)


# Format prompt
def format_prompt(examples):
    publish_date = examples["publish_date"]
    news = examples["article"]
    title = examples["title"]
    response = examples["response"]
    texts = []
    for date, article in zip(publish_date, news):
        texts.append(
            prompt_style.format(date, title, article, question, response, "")
            + EOS_TOKEN
        )
    return {"text": texts}


# Prepare dataset
training_data = Dataset.from_pandas(df)
training_data = training_data.map(format_prompt, batched=True)

# Training setup
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=training_data,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="qwentune-metrics",
        seed=42,
    ),
)

# Train the model
trainer.train()

# Save model and tokenizer
model.save_pretrained("deepseek/qwentune")
tokenizer.save_pretrained("deepseek/qwentune")
