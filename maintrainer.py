import torch
import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from huggingface_hub import notebook_login
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

2
huggingface_user = "Blackspell1"
dataset_name = "openassistant_german_costum_dataset"

class Llama3InstructDataset:
    def __init__(self, data):
        self.data = data
        self.prompts = []
        self.create_prompts()

    def create_prompt(self, row):
        prompt = f"### Human: {row} ### Assistant: {self.data[row]}"
        return prompt

    def create_prompts(self):
        for row in self.data:
            prompt = self.create_prompt(row)
            self.prompts.append(prompt)

    def get_dataset(self):
        df = pd.DataFrame({'prompt': self.prompts})
        return df

def create_dataset_hf(dataset):
    dataset.reset_index(drop=True, inplace=True)
    return DatasetDict({"train": Dataset.from_pandas(dataset)})

if __name__ == "__main__":
    with open('dataset.json', 'r') as f:
        data = json.load(f)

    dataset = Llama3InstructDataset(data)
    df = dataset.get_dataset()

    processed_data_path = 'processed_data'
    os.makedirs(processed_data_path, exist_ok=True)

    llama3_dataset = create_dataset_hf(df)
    llama3_dataset.save_to_disk(os.path.join(processed_data_path, "llama3_dataset"))
    llama3_dataset.push_to_hub(f"{huggingface_user}/{dataset_name}")

