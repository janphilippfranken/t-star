import os
from typing import Optional, List

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import random


class VLLMInferenceModel():
    """Wrapper for running inference with VLLM."""
    def __init__(
        self, 
        model: str,
        download_dir: str,
        dtype: str,
        tensor_parallel_size: int,
        quantization: Optional[str] = "none",
        seed: Optional[int] = 1,
    ):
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model,
            cache_dir=download_dir,
            token=os.getenv("HF_TOKEN"),
        )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = LLM(
            model=model,
            download_dir=download_dir,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization if quantization != "none" else None,
        )
        
    @property
    def model_type(self):
        return "VLLMInferenceModel"
    
    def prompt_logprobs(
        self, 
        prompts: List[str],
        n_logprobs_per_token=50,
    ) -> torch.Tensor:
        """Returns n logprobs of prompt tokens."""    
   
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            n=1,
            prompt_logprobs=n_logprobs_per_token,
            spaces_between_special_tokens=False,
        )
            
        output_responses = self.model.generate(
            sampling_params=sampling_params,
            prompts=prompts,
        )
        breakpoint()

        return output_responses
        
    def batch_log_probs(
        self, 
        prompts: List[str],
        responses: List[str], 
    ) -> torch.Tensor:
        """Returns log probabilities for a batch of responses."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        with torch.no_grad():
            # tokenize responses first to get max length (responses = prompts + responses)            
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                n=1,
                prompt_logprobs=5,
                spaces_between_special_tokens=False,
            )
            # might need to give up to answer and record logprobs from that last part 
            # generate responses
            output_responses = self.model.generate(
                prompt_token_ids=tokenized_responses_input_ids,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
  
            # now get the tokens back
            log_probs_responses = torch.tensor([
                [v for prob in output_answer.prompt_logprobs[1:] for _, v in prob.items()]
                for output_answer in output_responses
            ])
            breakpoint()
            
            # mask responses 
            mask_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id != 0 else 0
            labels = tokenized_responses.input_ids[:, 1:]
            mask = torch.logical_and(tokenized_prompts.input_ids[:, 1:] == mask_id, labels != 0)
            log_probs_responses.masked_fill_(~mask, 0) 
            log_probs = log_probs_responses.sum(dim=-1)

            torch.cuda.empty_cache()

            return log_probs
                     
    def batch_prompt(self, 
        prompts: List[str], 
        max_new_tokens: Optional[int] = 500,
        do_sample: Optional[bool] = True,
        top_p: Optional[float] = 0.9,
        top_k: Optional[int] = -1,
        temperature: Optional[float] = 0.1,
        num_return_sequences: Optional[int] = 1,
        best_of: Optional[int] = 1,
        use_beam_search: Optional[bool] = False,
        presence_penalty: Optional[float] = 0.0,
        frequency_penalty: Optional[float] = 0.0
        
    ) -> List[str]:
        """Batched text generation."""     
        temperature = 0.0  
        # sampling params
        if temperature == 0.0:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_new_tokens,
                n=num_return_sequences,
                best_of=1,
                use_beam_search=use_beam_search,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
        else:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                n=num_return_sequences,
            )
        
        # sample
        outputs = self.model.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )
        
        # extract generations
        generations = []
        for output in outputs: 
            for generated_sequence in output.outputs:
                generations.append(generated_sequence.text)
                
        return generations