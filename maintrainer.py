import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AdamW, get_linear_schedule_with_warmup
import json

# Initialize the accelerator
accelerator = Accelerator()

# Load the model and tokenizer
model_name = "DiscoResearch/Llama3-German-8B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Set the padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Apply quantization
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Load the dataset
def load_dataset_from_json(file_path):
    human_texts = []
    assistant_texts = []
    with open(file_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            text = obj['text']
            human_text = text.split('### Human: ')[1].split('### Assistant: ')[0]
            assistant_text = text.split('### Assistant: ')[1]
            human_texts.append(human_text)
            assistant_texts.append(assistant_text)
    return DatasetDict({'train': Dataset.from_dict({'human_text': human_texts, 'assistant_text': assistant_texts})})

dataset = load_dataset_from_json('NewDataset.json')

# Tokenize the dataset
def tokenize_function(examples):
    input_texts = [human_text + '\n' + assistant_text for human_text, assistant_text in zip(examples['human_text'], examples['assistant_text'])]
    tokenized_inputs = tokenizer(input_texts,
                                 padding="max_length",
                                 truncation=True,
                                 max_length=256,
                                 return_tensors='pt')
    return {'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask']}

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Prepare the dataloader
batch_size = 16
train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=batch_size, shuffle=True)

# Print the size of the tokenized dataset
print(f"Size of tokenized train dataset: {len(tokenized_dataset['train'])}")

print("DataLoader size:", len(train_dataloader))  # Debug statement to check the DataLoader size

# Configure training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    save_steps=1800,
    save_total_limit=2,
    prediction_loss_only=True,
    dataloader_pin_memory=True,
    fp16=True,
    bf16=False,
    dataloader_num_workers=4,
    logging_steps=10,
    report_to="none",
    skip_memory_metrics=True,
    deepspeed="ds_config.json"
)

# Get the number of training steps
num_training_steps = len(train_dataloader) * training_args.num_train_epochs

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # Initialize the optimizer
        optimizer = AdamW(self.model.parameters(), lr=5e-5)

        # Create a learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        return optimizer, scheduler

# Initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
)

# Prepare everything with accelerator
model, train_dataloader = accelerator.prepare(model, train_dataloader)

# Train the model
trainer.train()

# Save the fine-tuned model
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("runs/run1/model")
tokenizer.save_pretrained("runs/run1/model")
print("Model saved successfully.")
