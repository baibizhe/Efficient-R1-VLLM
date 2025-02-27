# Efficient-R1-VLLM

# 🚀 Efficient-R1-VLLMn: Efficient RL-Tuned MoE Vision-Language Model For Reasoning  

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-cyan)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-orange)](https://pytorch.org/)
[![SGLang Optimized](https://img.shields.io/badge/SGLang-Rollout_Speedup-green)](https://github.com/sgl-project/sglang)

is the **first project** to apply **reward-based reinforcement learning (GRPO)** to finetune  Mixture-of-Experts (MoE) vision-language model,(**DeepSeek2-VL**,) for multimodal reasoning tasks. We focus on optimizing training efficiency via **SGLang-accelerated rollouts** while maintaining the model’s reasoning capabilities.

![Architecture](docs/moerl_pipeline.png)  
*Pipeline: Vision-Language Input → DeepSeek2-VL MoE → GRPO Reward Optimization → Reasoning Output*

## 🔥 Key Innovations  
1. **First RL-Tuned MoE  Vision-Language**  
   - Pioneer reinforcement learning adaptation of **DeepSeek2-VL-MoE** (8x experts) on complex vision-language datasets (e.g., ScienceQA, VCR).  
2. **SGLang-Optimized Rollouts**  
   - Achieve **1.7 xfaster trajectory sampling** by integrating [SGLang](https://github.com/sgl-project/sglang) with DeepSeek2-VL’s official codebase.  
3. **Embed evaluation loop in trl framework**  

## 🚀 Quick Start  
code will come soon .
### Installation  
```bash  

cd MoERL-Vision  
pip install -r requirements.txt  # Requires CUDA 12.x and NVIDIA GPUs  

# Install SGLang for accelerated rollouts  
pip install "sglang[all]"  
