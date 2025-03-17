# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from unittest.mock import patch
import deepspeed
import torch
import torch.utils.data
import transformers
import  numpy as np
from accelerate.utils import broadcast_object_list, gather_object
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from tqdm import tqdm
from transformers import (
    AriaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2_5_VLForConditionalGeneration,Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from transformers.trainer_utils import EvalLoopOutput
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from math_verify import LatexExtractionConfig, parse, verify
import re
if is_peft_available():
    from peft import PeftConfig, get_peft_model
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from .utils import pad
from liger_kernel.transformers import monkey_patch
from deepspeed.accelerator import get_accelerator
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM,BatchCollateOutput
from deepseek_vl2.utils.io import load_pil_images

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


class Deepseek2VLGRPOTrainer(Trainer):

    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: GRPOConfig = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
                    None, None),
            peft_config: Optional["PeftConfig"] = None,
            max_pixels: Optional[int] = 12845056,
            min_pixels: Optional[int] = 3136,
            attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            self.model_id =model_id
            if "Qwen2.5-V" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif 'deepseek' in model_id:
                print('using deepseek')
                model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            else:
                raise NotImplementedError('not implemented')
                exit()
            
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        # vision_model_params = model.visual.parameters()
        # set_requires_grad(vision_model_params, False)

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2.5-V" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                
            elif 'deepseek' in model_id:
                print('using deepseek')
                vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
                self.ref_model=vl_gpt
            else:
                raise NotImplementedError('not implemented')
                exit()
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        for name, param in self.ref_model.named_parameters():
            param.requires_grad = False
        # Processing class
        if processing_class is None:
            if "Qwen2.5-VL" in model_id :
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels
            elif 'deepseek' in model_id:
                processing_class: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_id)

            else:
                raise NotImplementedError('not implemented')

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs
        
        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper, 8

        self.beta = args.beta



        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        if self.accelerator.is_main_process:
        # if True:
            # load vllm
            vllm_device = "auto"
            if vllm_device == "auto":
                # vllm_device = f"cuda:0"  # take the next GPU idx
                vllm_device = "cuda:0"
            # Check that the requested device is available
            if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                raise ValueError(
                    f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                    "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                    "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                    f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                )
            # Check that the requested device is not also used for training
            # if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
            #     print(
            #         f"The requested device {vllm_device} is also used for training. This may lead to unexpected "
            #         "behavior. It is recommended to use a dedicated device for vLLM."
            #     )
            # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
            # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
            # setting (profiling_patch).
            world_size_patch = patch("torch.distributed.get_world_size", return_value=1)

            profiling_patch = patch(
                "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
            )
            with world_size_patch, profiling_patch:
                self.llm = LLM(
                    model=model.name_or_path,
                    device=vllm_device,
                    gpu_memory_utilization=0.5,
                        enable_sleep_mode=True,
                            hf_overrides= {"architectures":["DeepseekVLV2ForCausalLM"]},
                )
            self.sampling_params = SamplingParams( #todo refactor sampleing to args
                temperature=args.temperature,
                top_p=0.95,
                top_k=50,
                max_tokens=self.max_completion_length,
                repetition_penalty=1.1
            )
            self.llm.sleep(1)
            print('llm has been slept')
        self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

        self.accelerator.wait_for_everyone()

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    def evaluation_loop(self,eval_dataloader,description,
                        prediction_loss_only,
            ignore_keys,
            metric_key_prefix):
        device = self.accelerator.device
        
        # print('begin custom eval!!!!!!!!!!!!!!!!!')
        # print('begin custom eval!!!!!!!!!!!!!!!!!')
        # print('begin custom eval!!!!!!!!!!!!!!!!!')
        print(len(eval_dataloader.dataset))
        num_samples = len(eval_dataloader.dataset)
        correct = 0 
        number_samples = 0
        progress_bar = tqdm(enumerate(eval_dataloader))
        pattern = re.compile(r'<answer>(.*?)<\/answer>', re.DOTALL)  # 支持多行
        accracy=0.
        self.model.eval()
        inputs_vllm = []
        # if self.accelerator.is_main_process:
        for data_item in eval_dataloader:
            # print(len(data_item),410)
            messages = eval(data_item[0]['prompt'])
            # print('messages',messages)
            prompt = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_data = load_pil_images(messages)
            if 'OA pointing in the direction of N20°E, and ray OB pointing in the direction of S50°W' in prompt:
                continue
            inputs_vllm.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_data
                },
                'solution':data_item[0]['solution']
            })
        # print('inputs_vllm',len(inputs_vllm))
        all_inputs_vllm = gather_object(inputs_vllm)
        self.accelerator.wait_for_everyone()
        # print('all_inputs_vllm',len(all_inputs_vllm))
        if self.accelerator.is_main_process:
            original_model_device = next(self.model.parameters()).device
            original_ref_model_device = next(self.ref_model.parameters()).device
            current_memory = torch.cuda.memory_allocated() / 1024**2  # 转换为MB
            print(f"1[主进程] CUDA显存占用: {current_memory:.2f} MB")
            # 可选：打印峰值显存占用
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"1[主进程] 峰值CUDA显存占用: {peak_memory:.2f} MB")
            self.model.to("cpu")
            self.ref_model.to("cpu")
            torch.cuda.empty_cache()  # 清空缓存
            get_accelerator().empty_cache()
            current_memory = torch.cuda.memory_allocated() / 1024**2  # 转换为MB
            print(f"2[主进程] CUDA显存占用: {current_memory:.2f} MB")
            # 可选：打印峰值显存占用
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"2[主进程] 峰值CUDA显存占用: {peak_memory:.2f} MB")

            self.llm.wake_up()
            current_memory = torch.cuda.memory_allocated() / 1024**2  # 转换为MB
            print(f"3[主进程] CUDA显存占用: {current_memory:.2f} MB")
            # 可选：打印峰值显存占用
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"3[主进程] 峰值CUDA显存占用: {peak_memory:.2f} MB")
            
            stop_token_ids = None
            sampling_params = SamplingParams(temperature=0.9, max_tokens=3036,
                                stop_token_ids=stop_token_ids, 
                                repetition_penalty=1.1)
            # outputs = self.llm.generate(all_inputs_vllm[0:2000], sampling_params=sampling_params, use_tqdm=True)
            outputs = []
            batch_size = 2
            all_inputs_vllm = all_inputs_vllm[0:200]
            with torch.cuda.stream(torch.cuda.Stream()):  # 使用独立CUDA流
                for i in range(0, len(all_inputs_vllm), batch_size):
                    batch = all_inputs_vllm[i:i + batch_size]
                    batch_outputs = self.llm.generate(batch, sampling_params=sampling_params, use_tqdm=True)
                    # Append the batch outputs to the outputs list
                    outputs.extend(batch_outputs)  # or outputs.append(batch_outputs) depending on the output format
                    print(len(outputs))
                    # Process the outputs
            print('outputs',outputs)
            for i in range(len(outputs)):
                # print(outputs[i])
                number_samples+=1
                output_text=outputs[i].outputs[0].text
                # print('prediced answer:',output_text)
                # print('ground truth answer:',data_item['answer'])
                # print(len(output_text),437)
                match = pattern.search(output_text)
                if match:
                    output_answer =  match.group(1).strip()
                    gt_answer  = pattern.search(all_inputs_vllm[i]['solution']).group(1).strip()
                    # print('gt_answer',gt_answer)
                    # print('prediced answer',output_answer)
                    if output_answer == gt_answer:
                        correct +=1
                    elif verify(output_answer, gt_answer) >0:
                        correct +=1
                accracy  = correct/number_samples
                self.model.to(original_model_device)
                self.ref_model.to(original_ref_model_device)
                torch.cuda.empty_cache()
                self.model.train()

        else:
            print("我是非主进程，Rank 是:", self.accelerator) # 非主进程 rank 大于 0

        return EvalLoopOutput(predictions=np.array([0]),label_ids=np.array([0]),metrics={'eval_accuracy':accracy},num_samples=num_samples)
    
    def process_batch_in_splits(inputs, model, batch_size, split_size):
        img_pixelper_length = int(inputs['pixel_values'].shape[0] / batch_size)
        logits_list = []

        for i in range(0, batch_size, split_size):
            # Calculate start and end indices for slicing
            start_input = i
            end_input = min(i + split_size, batch_size)
            start_pixel = int(i * img_pixelper_length)
            end_pixel = int(min(i + split_size, batch_size) * img_pixelper_length)

            # Create a dictionary of views without copying
            batch_slice = {
                'input_ids': inputs['input_ids'][start_input:end_input],
                'pixel_values': inputs['pixel_values'][start_pixel:end_pixel],
                'image_grid_thw': inputs['image_grid_thw'][start_input:end_input],
                'attention_mask': inputs['attention_mask'][start_input:end_input]
            }

            # Get the logits for the current slice
            logits = model(**batch_slice).logits
            logits_list.append(logits)

        # Concatenate the logits along the batch dimension
        logits = torch.cat(logits_list, dim=0)
        return logits
    def batchify_batch_collate_output(self,prompt_inputs, batch_size: int) -> BatchCollateOutput:
        batched_inputs = BatchCollateOutput(
            sft_format=[],
            input_ids=None,
            labels=None,
            images=None,
            attention_mask=None,
            images_seq_mask=None,
            images_spatial_crop=None,
            seq_lens=[]
        )

        for k in prompt_inputs.keys():
            v = prompt_inputs[k]
            if isinstance(v, torch.Tensor):
                batched_inputs[k] = v.repeat(batch_size, *[1] * (v.dim() - 1))
            elif isinstance(v, list):
                batched_inputs[k] = v * batch_size # Repeat list elements
            else:
                batched_inputs[k] = v # For other types, just keep it (though unlikely in BatchCollateOutput)
        return batched_inputs
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # self.accelerator.wait_for_everyone() # for waiting for eval loop

        device = self.accelerator.device
        conversation = eval(inputs[0]['prompt'])
        # for x in inputs:
        #     x['prompt'] = json.loads(x['prompt'])
        # prompts = [x["prompt"] for x in inputs]
        # prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        images = load_pil_images(conversation)
        print('images',images)
        print('conversation',conversation)
        prompt_inputs = self.processing_class.__call__(
        conversations=conversation,
        images=images,
        force_batchify=True,
        system_prompt="",
        inference_mode =False,
            ).to(device) 
        #这里是 sft_format=['system: A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>\n\nuser: <|User|>: <image>\n  In the 
        print('prompt_inputs',prompt_inputs)
        
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        # manual batch
        batch_size = self.num_generations

        batched_inputs = {
            k: v.repeat(batch_size, *[1] * (v.dim() - 1)) if isinstance(v, torch.Tensor) else v
            for k, v in prompt_inputs.items()
        }

        if self.max_prompt_length is not None:
            batched_inputs["input_ids"] = batched_inputs["input_ids"][:, -self.max_prompt_length:]
            batched_inputs["attention_mask"] = batched_inputs["attention_mask"][:, -self.max_prompt_length:]

        inputs_vllm = []

        for messages in prompts:
            prompt = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_data, _ = process_vision_info(messages)

            for i in range(batch_size):
                inputs_vllm.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": image_data
                    },
                })

        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                # remove_hooks(model)
                unwrapped_model = self.accelerator.unwrap_model(model)
                if is_compiled_module(unwrapped_model):
                    state_dict = unwrapped_model._orig_mod.state_dict()
                else:
                    state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights(state_dict.items())
            self._last_loaded_step = self.state.global_step
            # add_hooks(model)

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process

        
        all_inputs_vllm = gather_object(inputs_vllm)
        print(all_inputs_vllm[0:2])
        if self.accelerator.is_main_process:
            import time
            t0 = time.time()
            outputs = self.llm.generate(all_inputs_vllm, sampling_params=self.sampling_params, use_tqdm=False)
            completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]  
            print(len(completion_ids))
            print('generation main takes !!!!!',time.time()-t0,flush=True)
            
        else:
            completion_ids = [None] * len(all_inputs_vllm)

        # Broadcast the completions from the main process to all processes, ensuring each process receives its
        # print('generation takes !!!!!',time.time()-t0,flush=True)
        
        # corresponding slice.
        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        process_slice = slice(
            self.accelerator.process_index * len(prompts) * batch_size,
            (self.accelerator.process_index + 1) * len(prompts) * batch_size,
        )
        completion_ids = completion_ids[process_slice]

        # Pad the completions, and concatenate them with the prompts
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
        prompt_completion_ids = torch.cat([batched_inputs["input_ids"], completion_ids], dim=1)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([batched_inputs["attention_mask"], completion_mask], dim=1)

        def get_per_token_logps(model, **inputs):
            # logits = model(**inputs).logits  # (B, L, V)
            logits = model(**inputs).logits  # (B, L, V)
            
            # batch_size, seq_length = inputs['input_ids'].shape
            # split_size = batch_size // 2
            # logits_list = []
            # img_pixelper_length = int(inputs['pixel_values'].shape[0] / batch_size)
            # for i in range(0, batch_size, split_size):
            #     batch_slice = {'input_ids': inputs['input_ids'][i:i+split_size],'pixel_values':inputs['pixel_values'][int(i*img_pixelper_length):int((i+split_size)*img_pixelper_length)],'image_grid_thw':inputs['image_grid_thw'][i:i+split_size],'attention_mask':inputs['attention_mask'][i:i+split_size]}
            #     logits = model(**batch_slice).logits  # (split_size, L, V)
            #     logits_list.append(logits)
            # logits = torch.cat(logits_list, dim=0)
            # print('logits shape', logits.shape)
            # logits shape torch.Size([8, 2259, 151936])
            # logits shape torch.Size([8, 1179, 151936])
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = inputs['input_ids'][:,
                        1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        batched_inputs1 = batched_inputs.copy()

        # input_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        batched_inputs1["input_ids"] = prompt_completion_ids
        batched_inputs1["attention_mask"] = attention_mask
        per_token_logps = get_per_token_logps(model, **batched_inputs1)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)

        prompt_length = batched_inputs["input_ids"].size(1)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, **batched_inputs1)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, **batched_inputs1)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        torch.cuda.empty_cache()  # 清空缓存
        get_accelerator().empty_cache()

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
            self,
            model_name: Optional[str] = None,
            dataset_name: Optional[str] = None,
            tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            # wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            wandb_url=None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))