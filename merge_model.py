import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

#torch.cuda.empty_cache()

base_model_id = "microsoft/Phi-4"

bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
)

base_model = AutoModelForCausalLM.from_pretrained(base_model_id, 
                                             cache_dir="/home/user/phi-finetune/models",
                                             torch_dtype=torch.bfloat16,
                                             quantization_config=bnb_config,
                                             device_map="auto", 
                                             trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id
)

peft_model = PeftModel.from_pretrained(base_model, "/home/user/phi-finetune/models/phi3-vistgroup-finetune3")

merged_model = peft_model.merge_and_unload()
output_path = "/home/user/phi-finetune/models/phi4-ftuned"
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print('done')