# Efficient-R1-VLLM

# 🚀 Efficient-R1-VLLM: Efficient RL-Tuned MoE Vision-Language Model For Reasoning  

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-cyan)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-orange)](https://pytorch.org/)
[![SGLang Optimized](https://img.shields.io/badge/SGLang-Rollout_Speedup-green)](https://github.com/sgl-project/sglang)

is the **first project** to apply **reward-based reinforcement learning (GRPO)** to finetune  Mixture-of-Experts (MoE) vision-language model,(**DeepSeek2-VL**,) for multimodal reasoning tasks. We focus on optimizing training efficiency via **SGLang-accelerated rollouts** while maintaining the model’s reasoning capabilities.

![Architecture](docs/moerl_pipeline.png)  
*Pipeline: Vision-Language Input → DeepSeek2-VL MoE → GRPO Reward Optimization → Reasoning Output*

## 🔥 Key Findings 
* **Enforcing Image Captions Significantly Improves Vision-Language Model Performance**
    * **Motivation:** Compared to LLMs, VLLMs incorporate visual components, enabling them to leverage image information during reasoning.
    * **Implementation:** Modify prompts and reinforcement learning reward feedback to require the model to output an image caption before the reasoning output. For example:

        ```
        <caption>A right-angled triangle is shown where one leg forms an obtuse angle at vertex A while another side runs parallel to its hypotenuse. Within this diagram, three points labeled P, E, and Q have been marked such that they correspondingly intersect lines AD and EC along segments DB and BE.</caption>
        <think>xxxxxxxxxxxxxxxxxxxxxxx...</think>
        </answer> xxxxxxxxxxxxxxxxxxxxxxxx...<answer>
        ```

    * **Results:** :  In the Qwen-7B-Instruct model with the GRPO algorithm, enforcing image captions led to significant performance improvements.
    * **Visualization:** : The blue curve in blow figure is the accuracy for  only the ```<think>``` and ```<answer>``` . The purple is the accuracy for enforcing ```<caption>``` .



![image](https://github.com/user-attachments/assets/5c8b851a-048e-49f4-9e08-cea50ddec35b)






## 🔥 Key Innovations  
1. **First RL-Tuned MoE  Vision-Language**  
   - Pioneer reinforcement learning adaptation of **DeepSeek2-VL-MoE** (8x experts) on complex vision-language datasets (e.g., ScienceQA, VCR).  
2. **SGLang-Optimized Rollouts**  
   - Achieve **1.7 xfaster trajectory sampling** by integrating [SGLang](https://github.com/sgl-project/sglang) with DeepSeek2-VL’s official codebase.  
3. **Embedded evaluation loop in trl framework**  

## 🚀 Quick Start  
code will come soon .
### Installation  
```bash  

cd MoERL-Vision  
pip install -r requirements.txt  # Requires CUDA 12.x and NVIDIA GPUs  

# Install SGLang for accelerated rollouts  
pip install "sglang[all]"
```

Our codebase builds upon [R1-Multimodal-Journey]([https://github.com/EvolvingLMMs-Lab/open-r1-multimodal](https://github.com/FanqingM/R1-Multimodal-Journey/)) and integrates open-source contributions from [vLLM](https://github.com/vllm-project/vllm), [Open-R1](https://github.com/huggingface/open-r1), and [trl](https://github.com/huggingface/trl). We also extend our gratitude to [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) for their open-source techniques and base models, which have enabled us to further our exploration.

**Core Contributors (equal contribution with Alphabetical sorting)**

Bai Bizhe



Thanks for the help from Professor Wenqi Shao and Qiaosheng Zhang for their help.

