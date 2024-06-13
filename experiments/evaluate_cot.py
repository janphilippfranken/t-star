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

N_ITEMS = 20000
CoT = False

PROMPT = """Q: {question}\nA: Let's think step-by-step"""


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
        **args.model_config_vllm_qwen,
    )
 
    # data  
    dataset = load_dataset(
        "gsm8k",
        "main",
        split="test",
        cache_dir="/scr/jphilipp/tstar/datasets/gsm",
    )

    batch_prompts = [PROMPT.format(question=question) for question in dataset['question'][:N_ITEMS]]

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
        {'prompt': prompt,'label': label, 'response': response} 
        for prompt, label, response in zip(batch_prompts, evaluated_responses, formatted_responses)
    ]
    
    breakpoint()
    with open(f"gsm_results_qwen_0_shot_cot.json", "w") as file:
        json.dump(training_data, file, indent=4)


if __name__ == "__main__":
    fire.Fire(main())