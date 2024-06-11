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

batch_size = 50


LOG_PROMPT ="""Q: {question}\nA: """



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

    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="meta-llama/Meta-Llama-3-8B", cache_dir="/scr/jphilipp/sami-online/pretrained_models/Meta-Llama-3-8B", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="meta-llama/Meta-Llama-3-8B", cache_dir="/scr/jphilipp/sami-online/pretrained_models/Meta-Llama-3-8B")
 
    dataset = load_dataset(
        "gsm8k",
        "main",
        split="test",
        cache_dir="/scr/jphilipp/tstar/datasets/gsm",
    )

    log_batch_prompts = [LOG_PROMPT.format(question=question) for question in dataset['question'][:N_ITEMS]]
    tokenizer.pad_token = tokenizer.eos_token
    

    hidden_states = []
    for i in tqdm.tqdm(range(0, 1000, batch_size)):
        with torch.no_grad():
            inputs = tokenizer(log_batch_prompts[i:i+batch_size], padding='max_length', truncation=True, return_tensors="pt", max_length=200)
            input = inputs
            output = model(**input)
            print(output.hidden_states[-1].shape)
            hidden_states.append(output.hidden_states[-1])
        

    
    breakpoint()
    with open(f"gsm_results_llama_0_shot.json", "w") as file:
        json.dump(np.mean(evaluated_responses), file, indent=4)
    with open(f"gsm_logprobs_llama_0_shot.json", "w") as file:
        json.dump(training_data , file, indent=4)

if __name__ == "__main__":
    fire.Fire(main())