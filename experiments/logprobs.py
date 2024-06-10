import json
import fire
import random
import hydra
import re
import numpy as np
import os
from omegaconf import DictConfig
from tqdm import tqdm
import torch
from datasets import load_dataset
from tstar.models.vllm_models.inference_model_2 import VLLMInferenceModel

N_ITEMS = 2000
CoT = False
N_LOGPROBS_PER_TOKEN = 50

LOG_PROMPT = """Q: {question}\nA: <|end_of_text|>"""
PROMPT = """Q: {question}\nA: """


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
    model = VLLMInferenceModel(
        **args.model_config_vllm_llama,
    )
    
    # data  
    dataset = load_dataset(
        "gsm8k",
        "main",
        split="test",
        cache_dir="/scr/jphilipp/tstar/datasets/gsm",
    )

    batch_prompts = [PROMPT.format(question=question) for question in dataset['question'][:N_ITEMS]]
    log_batch_prompts = [LOG_PROMPT.format(question=question) for question in dataset['question'][:N_ITEMS]]
    token_logprobs = model.prompt_logprobs(log_batch_prompts, n_logprobs_per_token=N_LOGPROBS_PER_TOKEN)
    
    final_token_logprobs = []
    
    for logprobs in token_logprobs:
        idx = logprobs.prompt_token_ids.index(model.tokenizer.eos_token_id)
        final_token_logprob = logprobs.prompt_logprobs[idx]
        del final_token_logprob[model.tokenizer.eos_token_id]
        final_token_logprobs.append(final_token_logprob)
   
    batch_responses = model.batch_prompt(
        prompts=batch_prompts,
        temperature=0.0,
        **args.generation_config,
        
    )
    

    formatted_responses = [format_response(response) for response in batch_responses]
   
    extracted_responses = [extract_answer(response) for response in formatted_responses]
    gt_answers = [int(gt_answer.split('####')[1].strip().lower().replace(",", "")) for gt_answer in dataset['answer']]
    evaluated_responses = [evaluate_model_response(model_answer, gt) for model_answer, gt in zip(extracted_responses, gt_answers)]
    
    training_data = [
        {'logprobs': list(logprobs.values()), 'label': label} 
        for logprobs, label in zip(final_token_logprobs, evaluated_responses)
    ]
    breakpoint()
    with open(f"gsm_results_llama_0_shot.json", "w") as file:
        json.dump(np.mean(evaluated_responses), file, indent=4)
    with open(f"gsm_logprobs_llama_0_shot.json", "w") as file:
        json.dump(training_data , file, indent=4)

if __name__ == "__main__":
    fire.Fire(main())