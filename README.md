



<div align="center">
  <h1>$\mathcal{RTV}\text{-}Bench$: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video</h1> 
        If our project helps you, please give us a star ⭐ on GitHub to support us. 🥸🥸


[![arXiv](https://img.shields.io/badge/arXiv-2505.02064-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2505.02064) [![hf_checkpoint](https://img.shields.io/badge/🤗-RTV--Bench-9C276A.svg)](https://huggingface.co/datasets/xunsh/RTV-Bench) [![ms_checkpoint](https://img.shields.io/badge/🤖-RTV--Bench-8A2BE2.svg)](https://www.modelscope.cn/datasets/Jungang/RTV-Bench)  
</div>

## 🔥 News
* **`2025-09-20`** 🎉🎉🎉 Our paper has been accepted by Nips 2025, we will update our dataset and code for community as soon as possible~
* **`2025-06-27`** 🎉 We update core code for evaluation.
* **`2025-05-17`** 🎉 We have released the label json, which is named `QA.json`.
* **`2025-05-04`** 🎉  We released the paper $\mathcal{RTV}\text{-}Bench$: [Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](https://arxiv.org/abs/2505.02064).
* **`2025-05-03`** 🌟 We are happy to release the $\mathcal{RTV}\text{-}Bench$. You can find the $\mathcal{RTV}\text{-}Bench$ from [![hf_checkpoint](https://img.shields.io/badge/🤗-RTV--Bench-9C276A.svg)](https://huggingface.co/datasets/xunsh/RTV-Bench) or [![ms_checkpoint](https://img.shields.io/badge/🤖-RTV--Bench-8A2BE2.svg)](https://www.modelscope.cn/datasets/Jungang/RTV-Bench).
<p align="center">
    <img src="./asset/1_examples.png" width="100%" height="100%" >
</p>

## TODO
- [x] Release the final label json.
- [x] Release the evaluation code.
- [ ] Construct a more comprehensive benchmark for real-time video analysis.
- [ ] ···
## 👀 $\mathcal{RTV}\text{-}Bench$ Overview

We introduce $\mathcal{RTV}\text{-}Bench$, a fine-grained benchmark for MLLM real-time video analysis, which contains **552** videos (167.2 hours) and **4,631** high-quality QA pairs. We evaluated leading MLLMs, including proprietary (_e.g._ GPT-4o, Gemini 2.0), open-source offline (_e.g._ Qwen2.5-VL, VideoLLaMA3), and open-source real-time (_e.g._ VITA-1.5, InternLM-XComposer2.5-OmniLive) models. Experiment results show open-source real-time models largely outperform offline ones but still trail top proprietary models. Our analysis also reveals that larger model size or higher frame sampling rates do not significantly boost $\mathcal{RTV}\text{-}Bench$ performance, sometimes causing slight decreases. This underscores the need for better model architectures optimized for video stream processing and long sequences to advance real-time video analysis with MLLMs.  $\mathcal{RTV}\text{-}Bench$ includes three key principles: 
* **Multi-Timestamp Question Answering (MTQA)**, where answers evolve with scene changes; 
* **Hierarchical Question Structure**, combining basic and advanced queries; and
* **Multi-dimensional Evaluation**, assessing the ability of continuous perception, understanding, and reasoning. 

**Video Categories and Distribution of Question Difficulty and Query Characteristics.** 
<p align="center">
    <img src="./asset/2_dataset_stati.png" width="100%" height="100%" >
    (Left) RTV-Bench overs 3 key domains and 16 sub-class video types. 
    (Center) Distribution of question difficulty levels across eight representative task types, measured by percentage-based performance ranges.
    (Right) Distribution of question queries by video length, categorized into Shallow, Moderate, and Deep levels. The bar heights indicate counts, while the line chart overlays query proportions for each duration bucket.
</p>

## 🔖Evaluation Results
<p align="center">
    <img src="./asset/3_evaluation.png" width="100%" height="100%">
</p>

## 🛠️Evaluation 
```shell
bash scripts/eval/eval_model.sh
```

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=LJungang/RTV-Bench&type=Date)](https://star-history.com/#LJungang/RTV-Bench&Date)

## 📑 Citation
If you find $\mathcal{RTV}\text{-}Bench$ useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{xun2025rtv,
  title={RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video},
  author={Xun, Shuhang and Tao, Sicheng and Li, Jungang and Shi, Yibo and Lin, Zhixin and Zhu, Zhanhui and Yan, Yibo and Li, Hanqian and Zhang, Linghao and Wang, Shikang and others},
  journal={arXiv preprint arXiv:2505.02064},
  year={2025}
}
```
