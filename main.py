from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import os
import torch

# Define Model ID
model_id = "tiiuae/falcon-7b-instruct"

device_map = {'model.decoder.embed_tokens': 0,
 'lm_head': 0,
 'model.decoder.embed_positions': 0,
 'model.decoder.final_layer_norm': 0,
 'model.decoder.layers.0': 0,
 'model.decoder.layers.1': 0,
 'model.decoder.layers.2': 0,
 'model.decoder.layers.3': 0,
 'model.decoder.layers.4': 0,
 'model.decoder.layers.5': 0,
 'model.decoder.layers.6': 0,
 'model.decoder.layers.7': 1,
 'model.decoder.layers.8': 1,
 'model.decoder.layers.9': 1,
 'model.decoder.layers.10': 1,
 'model.decoder.layers.11': 1,
 'model.decoder.layers.12': 1,
 'model.decoder.layers.13': 1,
 'model.decoder.layers.14': 1,
 'model.decoder.layers.15': 1,
 'model.decoder.layers.16': 'cpu',
 'model.decoder.layers.17': 'cpu',
 'model.decoder.layers.18': 'cpu',
 'model.decoder.layers.19': 'cpu',
 'model.decoder.layers.20': 'cpu',
 'model.decoder.layers.21': 'cpu',
 'model.decoder.layers.22': 'cpu',
 'model.decoder.layers.23': 'cpu',
 'model.decoder.layers.24': 'cpu',
 'model.decoder.layers.25': 'cpu',
 'model.decoder.layers.26': 'cpu',
 'model.decoder.layers.27': 'cpu',
 'model.decoder.layers.28': 'cpu',
 'model.decoder.layers.29': 'cpu',
 'model.decoder.layers.30': 'cpu',
 'model.decoder.layers.31': 'cpu',
 'model.decoder.layers.32': 'cpu',
 'model.decoder.layers.33': 'cpu',
 'model.decoder.layers.34': 'cpu',
 'model.decoder.layers.35': 'cpu',
 'model.decoder.layers.36': 'cpu',
 'model.decoder.layers.37': 'cpu',
 'model.decoder.layers.38': 'cpu',
 'model.decoder.layers.39': 'cpu',
 'model.decoder.layers.40': 'cpu',
 'model.decoder.layers.41': 'cpu',
 'model.decoder.layers.42': 'cpu',
 'model.decoder.layers.43': 'cpu',
 'model.decoder.layers.44': 'cpu',
 'model.decoder.layers.45': 'cpu',
 'model.decoder.layers.46': 'cpu',
 'model.decoder.layers.47': 'cpu'}



# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load Model
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='./workspace/',
    torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", offload_folder="offload")

model = torch.compile(model)

# Set PT model to inference mode
model.eval()
# Build HF Transformers pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_length=400,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

pipeline('who is kim kardashian?')