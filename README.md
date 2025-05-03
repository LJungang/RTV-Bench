<div align="center">
  <h1>$\mathcal{RTV}\text{-}Bench$: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video</h1> 
</div>

## 🔥 News
* **`2025.05.03`** 🌟 We are happy to release the RTV-Bench.

## 👀 RTV-Bench Overview

we introduce $\mathcal{RTV}\text{-}Bench$, a fine-grained benchmark for MLLM real-time video analysis, which contains **552** videos (167.2 hours) and **4,631** high-quality QA pairs. We evaluated leading MLLMs, including proprietary (GPT-4o, Gemini 2.0), open-source offline (Qwen2.5-VL, VideoLLaMA3), and open-source real-time (VITA-1.5, InternLM-XComposer2.5-OmniLive) models. Experiment results show open-source real-time models largely outperform offline ones but still trail top proprietary models. Our analysis also reveals that larger model size or higher frame sampling rates do not significantly boost $\mathcal{RTV}\text{-}Bench$ performance, sometimes causing slight decreases. This underscores the need for better model architectures optimized for video stream processing and long sequences to advance real-time video analysis with MLLMs.  $\mathcal{RTV}\text{-}Bench$ includes three key principles: 
* **Multi-Timestamp Question Answering (MTQA)**, where answers evolve with scene changes; 
* **Hierarchical Question Structure**, combining basic and advanced queries; and
* **Multi-dimensional Evaluation**, assessing the ability of continuous perception, understanding, and reasoning. 
$\mathcal{RTV}\text{-}Bench$ 

## Examples
<p align="center">
    <img src="./asset/1_examples.png" width="100%" height="100%" >
</p>

