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

N_ITEMS = 10000
CoT = False

batch_size = 50

LOG_PROMPT ="""Human: Here is a question: 

{question}

Do you know the answer?

Assistant: """


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

    log_batch_prompts = [LOG_PROMPT.format(question=question) for question in dataset['question'][:N_ITEMS]]
    tokenizer.pad_token = tokenizer.eos_token
    

    hidden_states = []
    for i in tqdm.tqdm(range(0, len(log_batch_prompts))):
        with torch.no_grad():
            inputs = tokenizer(log_batch_prompts[i], padding=False, truncation=True, return_tensors="pt")
            inputs.to('cuda')
            output = model(**inputs)
            print(output.hidden_states[-1].shape)
            # hidden_states.append(output.hidden_states[-1].squeeze()[-1])
            next = output.logits[0, -2, :]
            test = model.lm_head(output.hidden_states[-1][-1])[-1, :]
            probs = torch.nn.functional.softmax(next, dim=-1)
            probs_test = torch.nn.functional.softmax(test, dim=-1)
            max_token = torch.argmax(probs)
            max_test = torch.argmax(probs_test)
            print(tokenizer.decode(max_token) == tokenizer.decode(max_test))
            print(torch.max(probs))
            hidden_states.append(torch.max(probs))
            breakpoint()

    new_hidden_states = torch.stack(hidden_states, dim=0)
    # # new_hidden_states = new_hidden_states.view(20 * 50, 200, 4096)
    torch.save(new_hidden_states, '/scr/jphilipp/tstar/data/hiddens_train.pt')
    breakpoint()
    # with open(f"gsm_logprobs_llama_0_shot.json", "w") as file:
    #     json.dump(training_data , file, indent=4)

if __name__ == "__main__":
    fire.Fire(main())
