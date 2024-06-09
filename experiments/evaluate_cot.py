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
from tstar.models.vllm_models.inference_model import VLLMInferenceModel

N_ITEMS = 2000

PROMPT_COT = """Q: {question}\nA: Let's think step-by-step"""

def extract_answer(answer):
    answer = answer.lower().strip()
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
        **args.model_config_vllm_mistral,
    )

    # data  
    dataset = load_dataset(
        "gsm8k",
        "main",
        split="test",
        cache_dir="/scr/jphilipp/tstar/datasets/gsm",
    )

    batch_prompts = [PROMPT_COT.format(question=question) for question in dataset['question'][:N_ITEMS]]

    batch_responses = model.batch_prompt(
        prompts=batch_prompts,
        temperature=0.0,
        **args.generation_config,
        
    )

    breakpoint()
    extracted_responses = [extract_answer(response) for response in batch_responses]
    gt_answers = [int(gt_answer.split('####')[1].strip().lower().replace(",", "")) for gt_answer in dataset['answer']]
    evaluated_responses = [evaluate_model_response(model_answer, gt) for model_answer, gt in zip(extracted_responses, gt_answers)]

    breakpoint()
    with open(f"gsm_results_mistral_base_0_shot_cot.json", "w") as file:
        json.dump(np.mean(evaluated_responses), file, indent=4)


if __name__ == "__main__":
    fire.Fire(main())