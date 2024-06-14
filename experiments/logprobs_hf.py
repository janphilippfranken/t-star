import json
import fire
import random
import hydra
import re
import numpy as np
import os
from omegaconf import DictConfig
import tqdm
import torch
from datasets import load_dataset
from tstar.models.vllm_models.inference_model_2 import VLLMInferenceModel
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

N_ITEMS = 2000
CoT = False

batch_size = 100

LOG_PROMPT ="""{original_prompt} {answer}

Is the above answer to the question correct?

A: Yes
B: No

The answer is: A"""


def format_response(response):
    formatted_response = ""
    try:
        formatted_response = response.split("Q:")[0].strip().lower()
    except:
        print("invalid, continue")
    return formatted_response

def extract_answer(answer):
    if '=' in answer:
        answer = answer.split('=')[-1].strip()
    answer = answer.replace(",", "")
    try:
        answer = re.findall(r"\d+", answer.strip())[-1]
        answer = int(answer)
    except:
        answer = "[invalid]"
    return answer

def evaluate_model_response(model_answer, gt_answer):
    try:
        result = int(model_answer) == int(gt_answer)
        return result
    except:
        return False


# main evaluation script
@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(args: DictConfig) -> None:
   
    # model
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="meta-llama/Meta-Llama-3-8B", cache_dir="/scr/jphilipp/sami-online/pretrained_models/Meta-Llama-3-8B", output_hidden_states=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="meta-llama/Meta-Llama-3-8B", cache_dir="/scr/jphilipp/sami-online/pretrained_models/Meta-Llama-3-8B")

    dataset = load_dataset(
        "gsm8k",
        "main",
        split="train",
        cache_dir="/scr/jphilipp/tstar/datasets/gsm",
    )
    
    dataset = json.load(open('gsm_0_shot_llama_8b.json'))

    log_batch_prompts = [LOG_PROMPT.format(original_prompt=datum['prompt'], answer=datum['response']) for datum in dataset[:N_ITEMS]]
    tokenizer.pad_token = tokenizer.eos_token
    
    hidden_states = []
    for i in tqdm.tqdm(range(0, len(log_batch_prompts))):
        with torch.no_grad():
            inputs = tokenizer(log_batch_prompts[i], padding=False, truncation=True, return_tensors="pt")
            inputs.to('cuda')
            output = model(**inputs)
            hidden_states.append(output.hidden_states[-1].squeeze()[-1])
    
    breakpoint()
    new_hidden_states = torch.stack(hidden_states, dim=0)
    torch.save(new_hidden_states, 'hidden_states.pt')
    labels = [datum['label'] for datum in dataset]
    with open('labels.json', 'w') as f: 
        json.dump(labels, f, indent=4)
    breakpoint()
    # with open(f"gsm_logprobs_llama_0_shot.json", "w") as file:
    #     json.dump(training_data , file, indent=4)

if __name__ == "__main__":
    fire.Fire(main())
