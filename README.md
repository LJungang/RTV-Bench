<div align="center">
  <h2>
  <a href="https://github.com/LJungang/RTV-Bench">
    𝓡𝓣𝓥-𝓑𝓮𝓷𝓬𝓱: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video
  </a>
</h2>

</div>

<div align="center">
  <p align="center">
    &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2505.02064"><b>Paper</b></a>&nbsp&nbsp | &nbsp&nbsp🏠 <a href="https://ljungang.github.io/RTV-Bench/"><b>Project Page</b></a>&nbsp&nbsp | 🤗 <a href="https://huggingface.co/datasets/xunsh/RTV-Bench"><b>Hugging Face</b></a>&nbsp&nbsp | 🤖 <a href="https://www.modelscope.cn/datasets/Jungang/RTV-Bench"><b>Model Scope</b></a>&nbsp&nbsp
</p>
<p align="center">
  If our project helps you, please give us a star ⭐ on GitHub to support us.
  <br>
  <img src="https://img.shields.io/github/stars/LJungang/RTV-Bench?style=flat-square&color=E0E0E0&label=Stars" alt="GitHub stars">
</p>
</div>

## 📰 News
- **`2026-01-13`** 🌟 We updated the evaluation code for [**VideoChat-Online**](#videochat-online), and released new results for the model.
- **`2025-12-27`** 📚 We released an [open-source survey repo](https://github.com/LJungang/Awesome-Video-Reasoning-Landscape) ![](https://img.shields.io/github/stars/LJungang/Awesome-Video-Reasoning-Landscape?style=social)on the landscape of `video reasoning`, covering CoT-based, CoF-based, Interleaved, and Streaming paradigms.  
* **`2025-09-20`** 🎉 Our paper has been accepted by NeurIPS 2025, we will update our dataset and code for community as soon as possible~
* **`2025-06-27`** 🎉 We update core code for evaluation.
* **`2025-05-17`** 🎉 We have released the label json, which is named `QA.json`.
* **`2025-05-04`** 🎉  We released the paper [𝓡𝓣𝓥-𝓑𝓮𝓷𝓬𝓱: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](https://arxiv.org/abs/2505.02064).
* **`2025-05-03`** 🌟 We are happy to release the $\mathcal{RTV}\text{-}Bench$. You can find the $\mathcal{RTV}\text{-}Bench$ from [![hf_checkpoint](https://img.shields.io/badge/🤗-RTV--Bench-9C276A.svg?style=flat-square)](https://huggingface.co/datasets/xunsh/RTV-Bench) or [![ms_checkpoint](https://img.shields.io/badge/🤖-RTV--Bench-8A2BE2.svg?style=flat-square)](https://www.modelscope.cn/datasets/Jungang/RTV-Bench).

---

## 🔎 Overview

**RTV-Bench** is a fine-grained benchmark for **online/streaming video reasoning with Multimodal Large Language Models (MLLMs)**.  
It targets continuous perception, understanding, and reasoning over long, streaming videos.

RTV-Bench is built around three core ideas:
- **Multi-Timestamp Question Answering**: answers evolve as video content changes over time.
- **Hierarchical Question Design**: from basic perception to advanced reasoning.
- **Multi-Dimensional Evaluation**: assessing continuous perception, understanding, and reasoning jointly.

The benchmark contains **552 videos** and **4,608 high-quality QA pairs**, covering diverse real-world scenarios.

<p align="center">
  <img src="./asset/1_examples.png" width="100%">
</p>

**Video Categories and Distribution of Question Difficulty and Query Characteristics.** <p align="center"> <img src="./asset/2_dataset_stati.png" width="100%" height="100%" > (Left) RTV-Bench overs 3 key domains and 16 sub-class video types. (Center) Distribution of question difficulty levels across eight representative task types, measured by percentage-based performance ranges. (Right) Distribution of question queries by video length, categorized into Shallow, Moderate, and Deep levels. The bar heights indicate counts, while the line chart overlays query proportions for each duration bucket. </p>

---

## 🛠️ Evaluation

This section introduces the environment setup, data preparation, and evaluation pipeline for RTV-Bench, and presents a minimal working example based on `Qwen2.5-VL` for model inference and result evaluation.

---

### 1. Environment Setup

First, clone the repository and create a dedicated conda environment:
```bash
  git clone git@github.com:LJungang/RTV-Bench.git
  cd RTV-Bench

  conda create -n rtv-bench python=3.10
  conda activate rtv-bench
```
Install the required dependencies:
```bash
  pip install transformers==4.57.0
  pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128
  pip install qwen_vl_utils
  pip install accelerate
  pip install opencv-python==4.12.0.88
  pip install decord==0.6.0

  conda install -y -c conda-forge ffmpeg x264

```
---

### 2. Download RTV-Bench Dataset

Download the RTV-Bench dataset from Hugging Face:
```bash
  mkdir rtv-bench
  huggingface-cli download \
    --repo-type dataset \
    --resume-download \
    RTVBench/RTV-Bench \
    --local-dir ./rtv-bench \
    --local-dir-use-symlinks False
```

---

### 3. Download Model Checkpoints

Download the Qwen2.5-VL model checkpoints:
```bash
  mkdir ckpts
  huggingface-cli download \
    --repo-type model \
    --resume-download \
    Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir ./ckpts \
    --local-dir-use-symlinks False
```
**Note:** Access to the model repository may require a valid Hugging Face token.

---

### 4. Data Preparation

Some raw videos may have format or encoding issues. To ensure consistent decoding and stable evaluation, please transcode the raw videos using the provided script before running experiments:

```bash
  bash ./scripts/data_preparation/0_transcode_raw_videos_overwrite.sh
```

RTV-Bench evaluation operates on **video clips** rather than raw long videos.  
Prepare the clips by splitting raw videos according to the provided timestamps:
```bash
  bash scripts/data_preparation/1_prepare_rtv_clips.sh
```
This script preprocesses the raw videos and generates temporally aligned video clips used for inference and evaluation.

---

### 5. Model Inference

Run offline inference using **Qwen2.5-VL-7B-Instruct**:
```bash
  bash ./scripts/eval/infer_offline_qwen2_5_vl.sh
```
The script performs batch inference on the prepared video clips and saves model predictions for subsequent evaluation.

---

### 6. Computing Metric

Get the evaluation metric results

```bash
  python ./scripts/eval/compute_acc.py --inputs [your_json_file]
  e.g.
  python ./scripts/eval/compute_acc.py --inputs ./eval_results/qwen2.5-VL-*.json

  python ./scripts/eval/compute_score.py --inputs [your_json_file]
  e.g.
  python ./scripts/eval/compute_score.py --inputs ./eval_results/qwen2.5-VL-*.json
```
### 7. Quick Evaluation for Other Models

<details>

<summary> <a id="videochat-online"> <b>VideoChat-Online</b> </a></summary>

```shell
  mkdir baseline
  git clone git@github.com:MCG-NJU/VideoChat-Online.git ./baseline

  conda create -n vco-rtv python=3.9
  conda activate vco-rtv

  pip install -r ./baseline/VideoChat-Online/requirements.txt
  pip install av

  bash ./scripts/eval/infer_online_videochat_online.sh
```

</details>

*Note:* In our quick evaluation setup, FlashAttention-2 was not installed.

## 🔖 Evaluation Results

<p align="center">
  <img src="./asset/3_evaluation.png" width="100%">
</p>

---


## 👍 Acknowledgements

We sincerely thank the authors and maintainers of the following projects, whose open-source models, codebases, and released checkpoints have been instrumental to our research and evaluation pipeline:

- **[Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL)**: The most powerful vision-language model in the Qwen series to date.

- **[VideoChat-Online](https://github.com/MCG-NJU/VideoChat-Online)**: A robust and efficient model for online video understanding.

Their high-quality implementations and transparent releases provide a solid foundation for reproducible research in video-centric multimodal understanding and benchmarking.


## 📑 Citation
If you find $\mathcal{RTV}\text{-}Bench$ useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{xun2025rtv,
  title={RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video},
  author={Xun, Shuhang and Tao, Sicheng and Li, Jungang and Shi, Yibo and Lin, Zhixin and Zhu, Zhanhui and Yan, Yibo and Li, Hanqian and Zhang, Linghao and Wang, Shikang and Liu, Yixin and Zhang, Hanbo and Ma, Ying and Hu, Xuming},
  booktitle={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025},
  organization={NeurIPS}
}
```
