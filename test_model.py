import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

#torch.cuda.empty_cache()

base_model_id = "microsoft/Phi-4"

bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
)
# model = AutoModelForCausalLM.from_pretrained(base_model_id, 
#                                              cache_dir="/home/user/phi-finetune/models",
#                                              torch_dtype=torch.bfloat16,
#                                              quantization_config=bnb_config,
#                                              device_map="auto", 
#                                              trust_remote_code=True)



# tokenizer = AutoTokenizer.from_pretrained(
#     base_model_id
# )

tokenizer = AutoTokenizer.from_pretrained("/home/user/phi-finetune/models/phi3-vistgroup-finetune3")
model = AutoModelForCausalLM.from_pretrained("/home/user/phi-finetune/models/phi3-vistgroup-finetune3", quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map="auto")

# Test prompt from training data
#prompt = "What is the primary key of the ladybug_specimens table?"
#prompt = "What do you know about ladybug_specimens table?"
prompt = "pls describe ladybug_specimens table structure"
expected_response = "The primary key is specimen_id, which uses the UUID data type."

# Format prompt with special tokens
input_text = f"<|user|>{prompt}<|end|>\n<|assistant|>"
inputs = tokenizer(input_text, return_tensors="pt")
# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = response.split("<|assistant|>")[-1].strip()

# Print results
print("Prompt:", prompt)
print("\nExpected response:", expected_response)
print("\nModel response:", response)
print("-----")
    