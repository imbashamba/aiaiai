# https://github.com/microsoft/Phi-3CookBook/blob/main/code/04.Finetuning/Phi-3-finetune-qlora-python.ipynb
# https://github.com/NVIDIA/workbench-example-phi3-finetune/blob/main/code/phi3-finetune.ipynb
from datasets import load_dataset, Dataset
import time
import pandas as pd
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import random
import json
from typing import List, Dict, Tuple

data_dir = "/home/user/phi-finetune/data"

class LadybugDatasetPreparator:
    
    def __init__(self, tokenizer, max_length: int = 512):
        """
        Initialize the dataset preparator
        Args:
            model_name: Name of the base model to use for tokenization
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Set special tokens for instruction format
        self.system_token = "<|system|>"
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.end_token = "<|end|>"
        
    def generate_schema_prompts(self) -> List[Dict[str, str]]:
        """Generate prompts about database schema"""
        schema_prompts = [
            {
                "instruction": "What is the primary key of the ladybug_specimens table?",
                "response": "The primary key of the ladybug_specimens table is the 'specimen_id' column. This column is unique for every record in the table and is used to identify each specimen individually. The data type for this column is UUID, ensuring global uniqueness."
            },
            {
                "instruction": "Identify the primary key of the ladybug_specimens table.",
                "response": "The primary key is 'specimen_id', which uniquely identifies each row in the table."
            },
            {
                "instruction": "Which column acts as the primary key in the ladybug_specimens table?",
                "response": "The primary key column is 'specimen_id', with a UUID data type."
            },
            {
                "instruction": "What fields are required for a valid record?",
                "response": "The required fields are specimen_id, species_name, and collection_date. All other fields can be NULL."
            },
            {
                "instruction": "What indexes are created for the ladybug_specimens table?",
                "response": "There are two indexes: idx_species_name on the species_name column and idx_collection_date on the collection_date column."
            },
            {
                "instruction": "What are all the columns in the ladybug_specimens table?",
                "response": "The ladybug_specimens table has the following columns: specimen_id (UUID), species_name (VARCHAR(100)), subspecies_name (VARCHAR(100)), spot_count (INTEGER), body_length_mm (DECIMAL(4,2)), spot_pattern_type (VARCHAR(50)), dominant_color (VARCHAR(30)), secondary_color (VARCHAR(30)), collection_date (TIMESTAMP), habitat_type (VARCHAR(50)), gps_latitude (DECIMAL(9,6)), gps_longitude (DECIMAL(9,6)), altitude_meters (INTEGER), temperature_celsius (DECIMAL(4,1)), humidity_percentage (INTEGER), behavior_observed (TEXT), health_status (VARCHAR(20)), genetic_sequence (TEXT), and last_modified (TIMESTAMP)."
            },
            {
                "instruction": "What are the measurement fields in the ladybug_specimens table?",
                "response": "The measurement fields are: body_length_mm (DECIMAL(4,2)) for specimen size, temperature_celsius (DECIMAL(4,1)) for ambient temperature, humidity_percentage (INTEGER) for relative humidity, altitude_meters (INTEGER) for elevation, and gps_latitude/gps_longitude (DECIMAL(9,6)) for precise location coordinates."
            },
            {
                "instruction": "Which columns are required (NOT NULL) in the ladybug_specimens table?",
                "response": "The required columns (NOT NULL) in the ladybug_specimens table are: specimen_id (primary key), species_name, and collection_date. All other columns can contain NULL values."
            },
            {
                "instruction": "What data type constraints exist in the ladybug_specimens table?",
                "response": "The table has several data type constraints: body_length_mm must be positive (CHECK > 0), humidity_percentage must be between 0 and 100, spot_count must be non-negative (CHECK >= 0), and specified precision limits on decimals (body_length_mm DECIMAL(4,2), temperature_celsius DECIMAL(4,1), GPS coordinates DECIMAL(9,6))."
            },
            {
                "instruction": "What are the text fields and their maximum lengths in the ladybug_specimens table?",
                "response": "The text fields and their lengths are: species_name VARCHAR(100), subspecies_name VARCHAR(100), spot_pattern_type VARCHAR(50), dominant_color VARCHAR(30), secondary_color VARCHAR(30), habitat_type VARCHAR(50), health_status VARCHAR(20). Additionally, behavior_observed and genetic_sequence are TEXT type with unlimited length."
            },
            {
                "instruction": "What indexes are defined on the ladybug_specimens table?",
                "response": "The table has three indexes: the primary key index on specimen_id, idx_species_name on the species_name column, and idx_collection_date on the collection_date column."
            },
            {
                "instruction": "What timestamp fields exist in the ladybug_specimens table and what are they used for?",
                "response": "The table has two timestamp fields: collection_date (NOT NULL) which records when the specimen was collected, and last_modified which automatically updates to the current timestamp when the record is changed."
            },
            {
                "instruction": "What fields are used for tracking the physical characteristics of ladybugs?",
                "response": "Physical characteristics are tracked using: spot_count (INTEGER), body_length_mm (DECIMAL(4,2)), spot_pattern_type (VARCHAR(50)), dominant_color (VARCHAR(30)), secondary_color (VARCHAR(30)), and health_status (VARCHAR(20))."
            },
            {
                "instruction": "How are location and environmental data stored in the ladybug_specimens table?",
                "response": "Location and environmental data are stored in: gps_latitude and gps_longitude (DECIMAL(9,6)) for coordinates, altitude_meters (INTEGER) for elevation, habitat_type (VARCHAR(50)) for environment type, temperature_celsius (DECIMAL(4,1)) for ambient temperature, and humidity_percentage (INTEGER) for relative humidity."
            }
        ]
        return schema_prompts

    def generate_constraint_prompts(self) -> List[Dict[str, str]]:
        """Generate prompts about data constraints"""
        constraint_prompts = [
            {
                "instruction": "What are the valid values for the spot_count field?",
                "response": "The spot_count field must be greater than or equal to 0, enforced by a CHECK constraint."
            },
            {
                "instruction": "What is the valid range for humidity_percentage?",
                "response": "The humidity_percentage must be between 0 and 100, enforced by a CHECK constraint."
            },
            {
                "instruction": "What are the measurement constraints for body_length_mm?",
                "response": "The body_length_mm field must be a positive decimal number with up to 4 digits total and 2 decimal places."
            }
        ]
        return constraint_prompts

    def generate_query_prompts(self) -> List[Dict[str, str]]:
        """Generate prompts about SQL queries"""
        query_prompts = [
            {
                "instruction": "Write a SQL query to find all ladybugs with more than 7 spots collected above 1000 meters altitude.",
                "response": "SELECT * FROM ladybug_specimens WHERE spot_count > 7 AND altitude_meters > 1000;"
            },
            {
                "instruction": "How would you query the average body length by species for healthy specimens only?",
                "response": "SELECT species_name, AVG(body_length_mm) as avg_length FROM ladybug_specimens WHERE health_status = 'healthy' GROUP BY species_name;"
            }
        ]
        return query_prompts

    def format_prompt(self, instruction: str, response: str) -> str:
        """Format the instruction and response into a single prompt string"""
        return f"{self.system_token}You are a helpful database expert.{self.end_token}\n" \
               f"{self.user_token}{instruction}{self.end_token}\n" \
               f"{self.assistant_token}{response}{self.end_token}"

    def prepare_dataset(self) -> Tuple[Dataset, Dataset]:
        """Prepare the complete dataset for training"""
        all_prompts = (
            self.generate_schema_prompts() +
            self.generate_constraint_prompts() +
            self.generate_query_prompts()
        )
        
        formatted_data = []
        for item in all_prompts:
            # Format with clear separation between input and output
            prompt = f"{self.system_token}You are a helpful database expert.{self.end_token}\n{self.user_token}{item['instruction']}{self.end_token}\n{self.assistant_token}"
            completion = f"{item['response']}{self.end_token}"
            
            formatted_data.append({
                "prompt": prompt,
                "completion": completion
            })
        
        # Create train/validation split
        random.shuffle(formatted_data)
        split_idx = int(len(formatted_data) * 0.8)
        train_data = formatted_data[:split_idx]
        val_data = formatted_data[split_idx:]
        
        # Convert to Datasets
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
        val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))
        
        def tokenize_function(examples):
            # Tokenize prompts
            model_inputs = self.tokenizer(
                examples["prompt"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Tokenize completions for labels
            labels = self.tokenizer(
                examples["completion"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Tokenize with proper labels
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        return train_dataset, val_dataset

for i in range(0,11):
    torch.cuda.empty_cache()

base_model_id = "microsoft/Phi-4"

bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, 
                                             cache_dir="/home/user/phi-finetune/models",
                                             torch_dtype=torch.bfloat16,
                                             quantization_config=bnb_config,
                                             device_map="auto", 
                                             trust_remote_code=True)

model = prepare_model_for_kbit_training(model)


tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_eos_token=True,
    add_bos_token=True, 
    use_fast=True
)

preparator = LadybugDatasetPreparator(tokenizer=tokenizer)
    
    # Prepare datasets
train_dataset, val_dataset = preparator.prepare_dataset()


max_length = 320 # This was an appropriate max length for my dataset

device = "cuda"


config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
    bias="none",
    lora_dropout=0.08,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
#print_trainable_parameters(model)

import transformers
from datetime import datetime

project = "vistgroup-finetune3"
base_model_name = "phi3"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_ratio=0.1,              # Используем ratio вместо steps
        per_device_train_batch_size=2, # Увеличиваем batch size
        gradient_accumulation_steps=6,
        max_steps=400,               # Увеличиваем количество шагов
        learning_rate=5e-6,          # Увеличиваем learning rate
        logging_steps=50,
        optim="adamw_torch",         # Используем adamw_torch
        save_strategy="steps",
        save_steps=100,
        weight_decay=0.01,           # Добавляем weight decay
        lr_scheduler_type="linear",  # Используем косинусный scheduler
        gradient_checkpointing=True, # Включаем gradient checkpointing
        fp16=True,                  # Включаем mixed precision
        evaluation_strategy="steps",        # Оценка каждые N шагов
        eval_steps=100,
        report_to="none"
        ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

trainer.save_model("/home/user/phi-finetune/models/phi3-vistgroup-finetune3")
tokenizer.save_pretrained("/home/user/phi-finetune/models/phi3-vistgroup-finetune3")