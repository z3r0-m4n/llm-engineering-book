import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

torch.random.manual_seed(0) 
model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    device_map="mps",  
    torch_dtype="auto",  
    trust_remote_code=True,  
    attn_implementation="eager",  # Disable Flash Attention
) 

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 

# Prompt
messages = [
    {"role": "user", "content": "Create a funny joke about chickens. Show your reasoning process step by step."}
]

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 500, 
    "return_full_text": False, 
    "temperature": 0.5, 
    "do_sample": False, 
    "use_cache": False, 
} 

output = pipe(messages, **generation_args) 
print(output[0]['generated_text'])
