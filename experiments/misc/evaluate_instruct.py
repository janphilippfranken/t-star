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
from transformers import AutoTokenizer



N_ITEMS = 2000
CoT = False

PROMPT_INSTRUCT = """Respond to the query below. Format your answer as follows:

Q: <query here>
A: <your final answer here>

Q: {question}"""


def format_response(response):
    formatted_response = ""
    try:
        formatted_response = response.split("A:")[1].strip().lower()
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
        **args.model_config_vllm_llama_instruct,
    )

    # data  
    dataset = load_dataset(
        "gsm8k",
        "main",
        split="test",
        cache_dir="/scr/jphilipp/tstar/datasets/gsm",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
        cache_dir="/scr/jphilipp/sami-online/pretrained_models/Meta-Llama-3-8B-Instruct",
        model_max_length=8000
    )
    # breakpoint()
    batch_prompts = [PROMPT.format(question=question) for question in dataset['question'][:N_ITEMS]]
    batch_prompts = [[{"role" : "user", "content": prompt}] for prompt in batch_prompts]
    batch_prompts = [tokenizer.apply_chat_template(prompt, tokenize=False) for prompt in batch_prompts]
    # breakpoint()
    batch_responses = model.batch_prompt(
        prompts=batch_prompts,
        temperature=0.0,
        **args.generation_config,
        
    )
    
    formatted_responses = [format_response(response) for response in batch_responses]
   
    extracted_responses = [extract_answer(response) for response in formatted_responses]
    gt_answers = [int(gt_answer.split('####')[1].strip().lower().replace(",", "")) for gt_answer in dataset['answer']]
    evaluated_responses = [evaluate_model_response(model_answer, gt) for model_answer, gt in zip(extracted_responses, gt_answers)]

    breakpoint()
    with open(f"gsm_results_llama_instruct_0_shot.json", "w") as file:
        json.dump(np.mean(evaluated_responses), file, indent=4)


if __name__ == "__main__":
    fire.Fire(main())