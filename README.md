# Simulating the Real World: Survey & Resources
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-pink.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-pink) [![Contribution Welcome](https://img.shields.io/badge/Contributions-welcome-pink)]()
![Stars](https://img.shields.io/github/stars/ALEEEHU/World-Simulator)

This repository is divided into two main sections:

> **Our Survey Paper Collection** - This section presents our survey, _"Simulating the Real World: A Unified Survey of Multimodal Generative Models"_, which systematically unify the study of 2D, video, 3D and 4D generation within a single framework.

> **Text2X Resources** – This section continues the original Awesome-Text2X-Resources, an open collection of state-of-the-art (SOTA) and novel Text-to-X (X can be everything) methods, including papers, codes, and datasets. The goal is to track the rapid progress in this field and provide researchers with up-to-date references.

⭐ If you find this repository useful for your research or work, a star is highly appreciated!

💗 This repository is continuously updated. If you find relevant papers, blog posts, videos, or other resources that should be included, feel free to submit a pull request (PR) or open an issue. Community contributions are always welcome!

<img src="./media/add_oil.png" width=15% align="right" />

## 🔔 News
- ✨ [13 Aug 2025] Updated our survey (Version 2, 25 pages) on arXiv.
- ✨ [6 Mar 2025] Updated our survey (Version 1) on arXiv.

## Table of Contents
- [Our Survey Paper Collection](#-our-survey-paper-collection)
  - [Abstract](#abstract)
    * [🔥 Cite Us!](#-citation)
    * [💡 New to this field? Check this!](#-getting-started-with-key-concepts)
  - [Paradigms](#paradigms)
    * [2D Generation](#2d-generation)
    * [Video Generation](#video-generation)
      * [Algorithms](#video-algorithms)
      * [Applications](#video-applications)
    * [3D Generation](#3d-generation)
      * [Algorithms](#3d-algorithms)
      * [Applications](#3d-applications)
    * [4D Generation](#4d-generation)
      * [Algorithms](#4d-algorithms)
      * [Applications](#4d-applications)
  - [Other Related Resources](#other-related-resources)
    * [WorldModel Benchmark](#world-model-benchmark)
    * [World Foundation Model Platform](#world-foundation-model-platform)
- [Text2X Resources](#-awesome-text2x-resources)
  - [Text to 4D](#text-to-4d)
    * [Accepted Papers](#-4d-accepted-papers)
    * [ArXiv Papers](#-4d-arxiv-papers)
    * [Additional Info](#previous-papers-and-other-awesome-repos)
  - [Text to Video](#text-to-video)
    * [Accepted Papers](#-t2v-accepted-papers)
    * [ArXiv Papers](#-t2v-arxiv-papers)
    * [Additional Info](#video-other-additional-info)
  - [Text to 3D Scene](#text-to-scene)
    * [Accepted Papers](#-3d-scene-accepted-papers)
    * [ArXiv Papers](#-3d-scene-arxiv-papers)
    * [Additional Info](#scene-other-additional-info)
  - [Text to Human Motion](#text-to-human-motion)
    * [Accepted Papers](#-motion-accepted-papers)
    * [ArXiv Papers](#-motion-arxiv-papers)
    * [Additional Info](#motion-other-additional-info)
  - [Text to 3D Human](#text-to-3d-human)
    * [Accepted Papers](#-human-accepted-papers)
    * [ArXiv Papers](#-human-arxiv-papers)
  - [Related Resources](#related-resources)
    * [Text to Other Tasks](#text-to-other-tasks)
    * [Survey and Awesome Repos](#survey-and-awesome-repos)

## 📜 Our Survey Paper Collection 
<p align=center> 𝐒𝐢𝐦𝐮𝐥𝐚𝐭𝐢𝐧𝐠 𝐭𝐡𝐞 𝐑𝐞𝐚𝐥 𝐖𝐨𝐫𝐥𝐝: 𝐀 𝐔𝐧𝐢𝐟𝐢𝐞𝐝 𝐒𝐮𝐫𝐯𝐞𝐲 𝐨𝐟 𝐌𝐮𝐥𝐭𝐢𝐦𝐨𝐝𝐚𝐥 𝐆𝐞𝐧𝐞𝐫𝐚𝐭𝐢𝐯𝐞 𝐌𝐨𝐝𝐞𝐥𝐬 </p>

<div align=center>

[![arXiv](https://img.shields.io/badge/arXiv-2503.04641-b31b1b.svg)](https://arxiv.org/abs/2503.04641)

</div>

<p align="center"> <img src="./media/teaser.png"> </p>

> ### Abstract
Understanding and replicating the real world is a critical challenge in Artificial General Intelligence (AGI) research. To achieve this, many existing approaches, such as world models, aim to capture the fundamental principles governing the physical world, enabling more accurate simulations and meaningful interactions. However, current methods often treat different modalities, including 2D (images), videos, 3D, and 4D representations, as independent domains, overlooking their interdependencies. Additionally, these methods typically focus on isolated dimensions of reality without systematically integrating their connections. In this survey, we present a unified survey for multimodal generative models that investigate the progression of data dimensionality in real-world simulation. Specifically, this survey starts from 2D generation (appearance), then moves to video (appearance+dynamics) and 3D generation (appearance+geometry), and finally culminates in 4D generation that integrate all dimensions. To the best of our knowledge, this is the first attempt to systematically unify the study of 2D, video, 3D and 4D generation within a single framework. To guide future research, we provide a comprehensive review of datasets, evaluation metrics and future directions, and fostering insights for newcomers. This survey serves as a bridge to advance the study of multimodal generative models and real-world simulation within a unified framework. 

> ### ⭐ Citation

If you find this paper and repo helpful for your research, please cite it below:

```bibtex

@article{hu2025simulating,
  title={Simulating the Real World: A Unified Survey of Multimodal Generative Models},
  author={Hu, Yuqi and Wang, Longguang and Liu, Xian and Chen, Ling-Hao and Guo, Yuwei and Shi, Yukai and Liu, Ce and Rao, Anyi and Wang, Zeyu and Xiong, Hui},
  journal={arXiv preprint arXiv:2503.04641},
  year={2025}
}

```

> ### 🧭 Getting Started with Key Concepts

> [!Note]
> If you are new to this field, you can find clear and concise definitions of essential technical terms and concepts, such as NeRF, 3DGS, SDS, and Diffusion Models in our [Glossary](./docs/glossary/glossary.md).

## Paradigms

> [!TIP]
> *Feel free to pull requests or contact us if you find any related papers that are not included here.* The process to submit a pull request is as follows:
- a. Fork the project into your own repository.
- b. Add the Title, Paper link, Conference, Project/GitHub link in `README.md` using the following format:
 ```
[Origin] **Paper Title** [[Paper](Paper Link)] [[GitHub](GitHub Link)] [[Project Page](Project Page Link)]
 ```
- c. Submit the pull request to this branch.

### 2D Generation

##### Text-to-Image Generation.
Here are some seminal papers and models.

* **Imagen**: [NeurIPS 2022] **Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding** [[Paper](https://arxiv.org/abs/2205.11487)] [[Project Page](https://imagen.research.google/)]
* **DALL-E**: [ICML 2021] **Zero-shot text-to-image generation** [[Paper](https://arxiv.org/abs/2102.12092)] [[GitHub](https://github.com/openai/DALL-E)]
* **DALL-E 2**: [arXiv 2022] **Hierarchical Text-Conditional Image Generation with CLIP Latents** [[Paper](https://arxiv.org/abs/2204.06125)]
* **DALL-E 3**: [[Platform Link](https://openai.com/index/dall-e-3/)]
* **DeepFloyd IF**: [[GitHub](https://github.com/deep-floyd/IF)]
* **Stable Diffusion**: [CVPR 2022] **High-Resolution Image Synthesis with Latent Diffusion Models** [[Paper](https://arxiv.org/abs/2112.10752)] [[GitHub](https://github.com/CompVis/latent-diffusion)]
* **SDXL**: [ICLR 2024 spotlight] **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis** [[Paper](https://arxiv.org/abs/2307.01952)] [[GitHub](https://github.com/Stability-AI/generative-models)]
* **FLUX.1**: [[Platform Link](https://blackforestlabs.ai/)]

----


### Video Generation
Text-to-video generation models adapt text-to-image frameworks to handle the additional dimension of dynamics in the real world. We classify these models into _three_ categories based on different generative machine learning architectures.

> ##### Survey
* [AIRC 2023] **A Survey of AI Text-to-Image and AI Text-to-Video Generators** [[Paper](https://arxiv.org/abs/2311.06329)] 
* [arXiv 2024] **Sora as an AGI World Model? A Complete Survey on Text-to-Video Generation** [[Paper](https://arxiv.org/abs/2403.05131)]

#### Video Algorithms

> ##### (1) VAE- and GAN-based Approaches.
VAE-based Approaches.
* **SV2P**: [ICLR 2018 Poster] **Stochastic Variational Video Prediction** [[Paper](https://arxiv.org/abs/1710.11252)] [[Project Page](https://sites.google.com/site/stochasticvideoprediction/)]
* [arXiv 2021] **FitVid: Overfitting in Pixel-Level Video Prediction** [[Paper](https://arxiv.org/abs/2403.05131)] [[GitHub](https://github.com/google-research/fitvid)] [[Project Page](https://sites.google.com/view/fitvidpaper)]

GAN-based Approaches.
* [CVPR 2018] **MoCoGAN: Decomposing Motion and Content for Video Generation** [[Paper](https://arxiv.org/abs/1707.04993)] [[GitHub](https://github.com/sergeytulyakov/mocogan)] 
* [CVPR 2022] **StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2** [[Paper](https://arxiv.org/abs/2112.14683)] [[GitHub](https://github.com/universome/stylegan-v)] [[Project Page](https://skor.sh/stylegan-v)]
* **DIGAN**: [ICLR 2022] **Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks** [[Paper](https://arxiv.org/abs/2202.10571)] [[GitHub](https://github.com/sihyun-yu/digan)] [[Project Page](https://sihyun.me/digan/)]
* [ICCV 2023] **StyleInV: A Temporal Style Modulated Inversion Network for Unconditional Video Generation** [[Paper](https://arxiv.org/abs/2308.16909)] [[GitHub](https://github.com/johannwyh/StyleInV)] [[Project Page](https://www.mmlab-ntu.com/project/styleinv/index.html)]

> ##### (2) Diffusion-based Approaches.
U-Net-based Architectures.
* [NeurIPS 2022] **Video Diffusion Models** [[Paper](https://arxiv.org/abs/2204.03458)] [[Project Page](https://video-diffusion.github.io/)] 
* [arXiv 2022] **Imagen Video: High Definition Video Generation with Diffusion Models** [[Paper](https://arxiv.org/abs/2210.02303)] [[Project Page](https://imagen.research.google/video/)]
* [arXiv 2022] **MagicVideo: Efficient Video Generation With Latent Diffusion Models** [[Paper](https://arxiv.org/abs/2211.11018)] [[Project Page](https://magicvideo.github.io/#)]
* [ICLR 2023 Poster] **Make-A-Video: Text-to-Video Generation without Text-Video Data** [[Paper](https://arxiv.org/abs/2209.14792)] [[Project Page](https://make-a-video.github.io/)]
* **GEN-1**: [ICCV 2023] **Structure and Content-Guided Video Synthesis with Diffusion Models** [[Paper](https://arxiv.org/abs/2302.03011)] [[Project Page](https://runwayml.com/research/gen-1)]
* **PYoCo**: [ICCV 2023] **Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models** [[Paper](https://arxiv.org/abs/2305.10474)] [[Project Page](https://research.nvidia.com/labs/dir/pyoco/)]
* [CVPR 2023] **Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models** [[Paper](https://arxiv.org/abs/2304.08818)] [[Project Page](https://research.nvidia.com/labs/toronto-ai/VideoLDM/)]
* [IJCV 2024] **Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation** [[Paper](https://arxiv.org/abs/2309.15818)] [[GitHub](https://github.com/showlab/Show-1)] [[Project Page](https://showlab.github.io/Show-1/)]
* [NeurIPS 2024] **VideoComposer: Compositional Video Synthesis with Motion Controllability** [[Paper](https://arxiv.org/abs/2306.02018)] [[GitHub](https://github.com/ali-vilab/videocomposer)] [[Project Page](https://videocomposer.github.io/)] 
* [ICLR 2024 Spotlight] **AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning** [[Paper](https://arxiv.org/abs/2307.04725)] [[GitHub](https://github.com/guoyww/AnimateDiff)] [[Project Page](https://animatediff.github.io/)] 
* [CVPR 2024] **Make Pixels Dance: High-Dynamic Video Generation** [[Paper](https://arxiv.org/abs/2311.10982)] [[Project Page](https://makepixelsdance.github.io/)]
* [ECCV 2024] **Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning** [[Paper](https://arxiv.org/abs/2311.10709)] [[Project Page](https://emu-video.metademolab.com/)]
* [SIGGRAPH Asia 2024] **Lumiere: A Space-Time Diffusion Model for Video Generation** [[Paper](https://arxiv.org/abs/2401.12945)] [[Project Page](https://lumiere-video.github.io/)]

Transformer-based Architectures.
* [ICLR 2024 Poster] **VDT: General-purpose Video Diffusion Transformers via Mask Modeling** [[Paper](https://arxiv.org/abs/2305.13311)] [[GitHub](https://github.com/RERV/VDT)] [[Project Page](https://vdt-2023.github.io/)]
* **W.A.L.T**: [ECCV 2024] **Photorealistic Video Generation with Diffusion Models** [[Paper](https://arxiv.org/abs/2312.06662)] [[Project Page](https://walt-video-diffusion.github.io/)]
* [CVPR 2024] **Snap Video: Scaled Spatiotemporal Transformers for Text-to-Video Synthesis** [[Paper](https://arxiv.org/abs/2402.14797)] [[Project Page](https://snap-research.github.io/snapvideo/)]
* [CVPR 2024] **GenTron: Diffusion Transformers for Image and Video Generation** [[Paper](https://arxiv.org/abs/2312.04557)] [[Project Page](https://www.shoufachen.com/gentron_website/)]
* [ICLR 2025 Poster] **CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer** [[Paper](https://arxiv.org/abs/2408.06072)] [[GitHub](https://github.com/THUDM/CogVideo)]
* [ICLR 2025 Spotlight] **Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers** [[Paper](https://arxiv.org/abs/2405.05945)] [[GitHub](https://github.com/Alpha-VLLM/Lumina-T2X)]

> ##### (3) Autoregressive-based Approaches.
* **VQ-GAN**: [CVPR 2021 Oral] **Taming Transformers for High-Resolution Image Synthesis** [[Paper](https://arxiv.org/abs/2012.09841)] [[GitHub](https://github.com/CompVis/taming-transformers)] 
* [CVPR 2023 Highlight] **MAGVIT: Masked Generative Video Transformer** [[Paper](https://arxiv.org/abs/2212.05199)] [[GitHub](https://github.com/google-research/magvit)] [[Project Page](https://magvit.cs.cmu.edu/)]
* [ICLR 2023 Poster] **CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers** [[Paper](https://arxiv.org/abs/2205.15868)] [[GitHub](https://github.com/THUDM/CogVideo)]
* [ICML 2024] **VideoPoet: A Large Language Model for Zero-Shot Video Generation** [[Paper](https://arxiv.org/abs/2312.14125)] [[Project Page](https://sites.research.google/videopoet/)]
* [ICLR 2024 Poster] **Language Model Beats Diffusion - Tokenizer is key to visual generation** [[Paper](https://arxiv.org/abs/2310.05737)] 
* [arXiv 2024] **Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation** [[Paper](https://arxiv.org/abs/2409.04410)] [[GitHub](https://github.com/TencentARC/SEED-Voken)]
* [arXiv 2024] **Emu3: Next-Token Prediction is All You Need** [[Paper](https://arxiv.org/abs/2409.18869)] [[GitHub](https://github.com/baaivision/Emu3)] [[Project Page](https://emu.baai.ac.cn/about)]
* [ICLR 2025 Poster] **Accelerating Auto-regressive Text-to-Image Generation with Training-free Speculative Jacobi Decoding** [[Paper](https://arxiv.org/abs/2410.01699)] [[GitHub](https://github.com/tyshiwo1/Accelerating-T2I-AR-with-SJD/)]

#### Video Applications
> ##### Video Editing.
* [ICCV 2023] **Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation** [[Paper](https://arxiv.org/abs/2212.11565)] [[GitHub](https://github.com/showlab/Tune-A-Video)] [[Project Page](https://tuneavideo.github.io/)]
* [ICCV 2023] **Pix2Video: Video Editing using Image Diffusion** [[Paper](https://arxiv.org/abs/2303.12688)] [[GitHub](https://github.com/duyguceylan/pix2video)] [[Project Page](https://duyguceylan.github.io/pix2video.github.io/)]
* [CVPR 2024] **VidToMe: Video Token Merging for Zero-Shot Video Editing** [[Paper](https://arxiv.org/abs/2312.10656)] [[GitHub](https://github.com/lixirui142/VidToMe)] [[Project Page](https://vidtome-diffusion.github.io/)]
* [CVPR 2024] **Video-P2P: Video Editing with Cross-attention Control** [[Paper](https://arxiv.org/abs/2303.04761)] [[GitHub](https://github.com/dvlab-research/Video-P2P)] [[Project Page](https://video-p2p.github.io/)]
* [CVPR 2024 Highlight] **CoDeF: Content Deformation Fields for Temporally Consistent Video Processing** [[Paper](https://arxiv.org/abs/2308.07926)] [[GitHub](https://github.com/ant-research/CoDeF)] [[Project Page](https://qiuyu96.github.io/CoDeF/)]
* [NeurIPS 2024] **Towards Consistent Video Editing with Text-to-Image Diffusion Models** [[Paper](https://arxiv.org/abs/2305.17431)]
* [ICLR 2024 Poster] **Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models** [[Paper](https://arxiv.org/abs/2310.01107)] [[GitHub](https://github.com/Ground-A-Video/Ground-A-Video)] [[Project Page](https://ground-a-video.github.io/)]
* [arXiv 2024] **UniEdit: A Unified Tuning-Free Framework for Video Motion and Appearance Editing** [[Paper](https://arxiv.org/abs/2402.13185)] [[GitHub](https://github.com/JianhongBai/UniEdit)] [[Project Page](https://jianhongbai.github.io/UniEdit/)]
* [TMLR 2024] **AnyV2V: A Tuning-Free Framework For Any Video-to-Video Editing Tasks** [[Paper](https://arxiv.org/abs/2403.14468)] [[GitHub](https://github.com/TIGER-AI-Lab/AnyV2V)] [[Project Page](https://tiger-ai-lab.github.io/AnyV2V/)]

> ##### Novel View Synthesis.
* [arXiv 2024] **ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis** [[Paper](https://arxiv.org/abs/2409.02048)] [[GitHub](https://github.com/Drexubery/ViewCrafter)] [[Project Page](https://drexubery.github.io/ViewCrafter/)]
* [CVPR 2024 Highlight] **ViVid-1-to-3: Novel View Synthesis with Video Diffusion Models** [[Paper](https://arxiv.org/abs/2312.01305)] [[GitHub](https://github.com/ubc-vision/vivid123)] [[Project Page](https://jgkwak95.github.io/ViVid-1-to-3/)]
* [ICLR 2025 Poster] **CameraCtrl: Enabling Camera Control for Video Diffusion Models** [[Paper](https://arxiv.org/abs/2404.02101)] [[GitHub](https://github.com/hehao13/CameraCtrl)] [[Project Page](https://hehao13.github.io/projects-CameraCtrl/)]
* [ICLR 2025 Poster] **NVS-Solver: Video Diffusion Model as Zero-Shot Novel View Synthesizer** [[Paper](https://arxiv.org/abs/2405.15364)] [[GitHub](https://github.com/ZHU-Zhiyu/NVS_Solver)] 

> ##### Human Animation in Videos.
* [ICCV 2019] **Everybody Dance Now** [[Paper](https://arxiv.org/abs/1808.07371)] [[GitHub](https://github.com/carolineec/EverybodyDanceNow)] [[Project Page](https://carolineec.github.io/everybody_dance_now/)]
* [ICCV 2019] **Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel View Synthesis** [[Paper](https://arxiv.org/abs/1909.12224)] [[GitHub](https://github.com/svip-lab/impersonator)] [[Project Page](https://svip-lab.github.io/project/impersonator.html)] [[Dataset](https://svip-lab.github.io/dataset/iPER_dataset.html)]
* [NeurIPS 2019] **First Order Motion Model for Image Animation** [[Paper](https://arxiv.org/abs/2003.00196)] [[GitHub](https://github.com/AliaksandrSiarohin/first-order-model)] [[Project Page](https://aliaksandrsiarohin.github.io/first-order-model-website/)]
* [ICCV 2023] **Adding Conditional Control to Text-to-Image Diffusion Models** [[Paper](https://arxiv.org/abs/2302.05543)] [[GitHub](https://github.com/lllyasviel/ControlNet)]
* [ICCV 2023] **HumanSD: A Native Skeleton-Guided Diffusion Model for Human Image Generation** [[Paper](https://arxiv.org/abs/2304.04269)] [[GitHub](https://github.com/IDEA-Research/HumanSD)] [[Project Page](https://idea-research.github.io/HumanSD/)]
* [CVPR 2023] **Learning Locally Editable Virtual Humans** [[Paper](https://arxiv.org/abs/2305.00121)] [[GitHub](https://github.com/custom-humans/editable-humans)] [[Project Page](https://custom-humans.github.io/)] [[Dataset](https://custom-humans.ait.ethz.ch/)]
* [CVPR 2023] **Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation** [[Paper](https://arxiv.org/abs/2311.17117)] [[GitHub](https://github.com/HumanAIGC/AnimateAnyone)] [[Project Page](https://humanaigc.github.io/animate-anyone/)]
* [CVPRW 2024] **LatentMan: Generating Consistent Animated Characters using Image Diffusion Models** [[Paper](https://arxiv.org/abs/2312.07133)] [[GitHub](https://github.com/abdo-eldesokey/latentman)] [[Project Page](https://abdo-eldesokey.github.io/latentman/)]
* [IJCAI 2024] **Zero-shot High-fidelity and Pose-controllable Character Animation** [[Paper](https://arxiv.org/abs/2404.13680)] 
* [arXiv 2024] **UniAnimate: Taming Unified Video Diffusion Models for Consistent Human Image Animation** [[Paper](https://arxiv.org/abs/2406.01188)] [[GitHub](https://github.com/ali-vilab/UniAnimate)] [[Project Page](https://unianimate.github.io/)]
* [arXiv 2024] **MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling** [[Paper](https://arxiv.org/abs/2409.16160)] [[GitHub](https://github.com/menyifang/MIMO)] [[Project Page](https://menyifang.github.io/projects/MIMO/index.html)]

----

### 3D Generation

#### 3D Algorithms
##### Text-to-3D Generation.
>##### Survey
* [arXiv 2023] **Generative AI meets 3D: A Survey on Text-to-3D in AIGC Era** [[Paper](https://arxiv.org/abs/2305.06131)]
* [arXiv 2024] **Advances in 3D Generation: A Survey** [[Paper](https://arxiv.org/abs/2401.17807)]
* [arXiv 2024] **A Survey On Text-to-3D Contents Generation In The Wild** [[Paper](https://arxiv.org/abs/2405.09431)]

>##### Feedforward Approaches.
* [arXiv 2022] **3D-LDM: Neural Implicit 3D Shape Generation with Latent Diffusion Models** [[Paper](https://arxiv.org/abs/2212.00842)] [[GitHub](https://www.3dldm.org/)] 
* [arXiv 2022] **Point-E: A System for Generating 3D Point Clouds from Complex Prompts** [[Paper](https://arxiv.org/abs/2212.08751)] [[GitHub](https://github.com/openai/point-e)] 
* [arXiv 2023] **Shap-E: Generating Conditional 3D Implicit Functions** [[Paper](https://arxiv.org/abs/2305.02463)] [[GitHub](https://github.com/openai/shap-e)] 
* [NeurIPS 2023] **Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation** [[Paper](https://arxiv.org/abs/2306.17115)] [[GitHub](https://github.com/NeuralCarver/Michelangelo)] [[Project Page](https://neuralcarver.github.io/michelangelo/)]
* [ICCV 2023] **ATT3D: Amortized Text-to-3D Object Synthesis** [[Paper](https://arxiv.org/abs/2306.07349)] [[Project Page](https://research.nvidia.com/labs/toronto-ai/ATT3D/)]
* [ICLR 2023 Spotlight] **MeshDiffusion: Score-based Generative 3D Mesh Modeling** [[Paper](https://arxiv.org/abs/2303.08133)] [[GitHub](https://github.com/lzzcd001/MeshDiffusion/)] [[Project Page](https://meshdiffusion.github.io/)]
* [CVPR 2023] **Diffusion-SDF: Text-to-Shape via Voxelized Diffusion** [[Paper](https://arxiv.org/abs/2212.03293)] [[GitHub](https://github.com/ttlmh/Diffusion-SDF)] [[Project Page](https://ttlmh.github.io/DiffusionSDF/)]
* [ICML 2024] **HyperFields:Towards Zero-Shot Generation of NeRFs from Text** [[Paper](https://arxiv.org/abs/2310.17075)] [[GitHub](https://github.com/threedle/hyperfields)] [[Project Page](https://threedle.github.io/hyperfields/)]
* [ECCV 2024] **LATTE3D: Large-scale Amortized Text-To-Enhanced3D Synthesis** [[Paper](https://arxiv.org/abs/2403.15385)] [[Project Page](https://research.nvidia.com/labs/toronto-ai/LATTE3D/)]
* [arXiv 2024] **AToM: Amortized Text-to-Mesh using 2D Diffusion** [[Paper](https://arxiv.org/abs/2402.00867)] [[GitHub](https://github.com/snap-research/AToM)] [[Project Page](https://snap-research.github.io/AToM/)]

>##### Optimization-based Approaches.
* [ICLR 2023 notable top 5%] **DreamFusion: Text-to-3D using 2D Diffusion** [[Paper](https://arxiv.org/abs/2209.14988)] [[Project Page](https://dreamfusion3d.github.io/)]
* [CVPR 2023 Highlight] **Magic3D: High-Resolution Text-to-3D Content Creation** [[Paper](https://arxiv.org/abs/2211.10440)] [[Project Page](https://research.nvidia.com/labs/dir/magic3d/)]
* [CVPR 2023] **Dream3D: Zero-Shot Text-to-3D Synthesis Using 3D Shape Prior and Text-to-Image Diffusion Models** [[Paper](https://arxiv.org/abs/2212.14704)] [[Project Page](https://bluestyle97.github.io/dream3d/)]
* [ICCV 2023] **Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation** [[Paper](https://arxiv.org/abs/2303.13873)] [[GitHub](https://github.com/Gorilla-Lab-SCUT/Fantasia3D)] [[Project Page](https://fantasia3d.github.io/)]
* [NeurIPS 2023 Spotlight] **ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation** [[Paper](https://arxiv.org/abs/2305.16213)] [[GitHub](https://github.com/thu-ml/prolificdreamer)] [[Project Page](https://ml.cs.tsinghua.edu.cn/prolificdreamer/)]
* [ICLR 2024 Poster] **MVDream: Multi-view Diffusion for 3D Generation** [[Paper](https://arxiv.org/abs/2308.16512)] [[GitHub](https://github.com/bytedance/MVDream)] [[Project Page](https://mv-dream.github.io/)]
* [ICLR 2024 Oral] **DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation** [[Paper](https://arxiv.org/abs/2309.16653)] [[GitHub](https://github.com/dreamgaussian/dreamgaussian)] [[Project Page](https://dreamgaussian.github.io/)]
* [CVPR 2024] **PI3D: Efficient Text-to-3D Generation with Pseudo-Image Diffusion** [[Paper](https://arxiv.org/abs/2312.09069)] 
* [CVPR 2024] **VP3D: Unleashing 2D Visual Prompt for Text-to-3D Generation** [[Paper](https://arxiv.org/abs/2403.17001)] [[Project Page](https://vp3d-cvpr24.github.io/)]
* [CVPR 2024] **GSGEN: Text-to-3D using Gaussian Splatting** [[Paper](https://arxiv.org/abs/2309.16585)]  [[GitHub](https://github.com/gsgen3d/gsgen)] [[Project Page](https://gsgen3d.github.io/)]
* [CVPR 2024] **GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models** [[Paper](https://arxiv.org/abs/2310.08529)]  [[GitHub](https://github.com/hustvl/GaussianDreamer)] [[Project Page](https://taoranyi.com/gaussiandreamer/)]
* [CVPR 2024] **Sculpt3D: Multi-View Consistent Text-to-3D Generation with Sparse 3D Prior** [[Paper](https://arxiv.org/abs/2403.09140)]  [[GitHub](https://github.com/StellarCheng/Scuplt_3d/tree/main)] [[Project Page](https://stellarcheng.github.io/Sculpt3D/)]

>##### MVS-based Approaches.
* [ICLR 2024 Poster] **Instant3D: Fast Text-to-3D with Sparse-view Generation and Large Reconstruction Model** [[Paper](https://arxiv.org/abs/2311.06214)] [[Project Page](https://jiahao.ai/instant3d/)]
* [CVPR 2024] **Direct2.5: Diverse Text-to-3D Generation via Multi-view 2.5D Diffusion** [[Paper](https://arxiv.org/abs/2311.15980)]  [[GitHub](https://github.com/apple/ml-direct2.5)] [[Project Page](https://nju-3dv.github.io/projects/direct25/)]
* [CVPR 2024] **Sherpa3D: Boosting High-Fidelity Text-to-3D Generation via Coarse 3D Prior** [[Paper](https://arxiv.org/abs/2312.06655)]  [[GitHub](https://github.com/liuff19/Sherpa3D)] [[Project Page](https://liuff19.github.io/Sherpa3D/)]

##### Image-to-3D Generation.
>##### Feedforward Approaches.
* [arXiv 2023] **3DGen: Triplane Latent Diffusion for Textured Mesh Generation** [[Paper](https://arxiv.org/abs/2303.05371)] 
* [NeurIPS 2023] **Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation** [[Paper](https://arxiv.org/abs/2306.17115)] [[GitHub](https://github.com/NeuralCarver/Michelangelo)] [[Project Page](https://neuralcarver.github.io/michelangelo/)]
* [NeurIPS 2024] **Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer** [[Paper](https://arxiv.org/abs/2405.14832)] [[GitHub](https://github.com/DreamTechAI/Direct3D)] [[Project Page](https://www.neural4d.com/research/direct3d)]
* [SIGGRAPH 2024 Best Paper Honorable Mention] **CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets** [[Paper](https://arxiv.org/abs/2406.13897)] [[GitHub](https://github.com/CLAY-3D/OpenCLAY)] [[Project Page](https://sites.google.com/view/clay-3dlm)]
* [arXiv 2024] **CraftsMan: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner** [[Paper](https://arxiv.org/abs/2405.14979)] [[GitHub](https://github.com/wyysf-98/CraftsMan3D)] [[Project Page](https://craftsman3d.github.io/)]
* [arXiv 2024] **Structured 3D Latents for Scalable and Versatile 3D Generation** [[Paper](https://arxiv.org/abs/2412.01506)] [[GitHub](https://github.com/Microsoft/TRELLIS)] [[Project Page](https://trellis3d.github.io/)]

>##### Optimization-based Approaches.
* [arXiv 2023] **Consistent123: Improve Consistency for One Image to 3D Object Synthesis** [[Paper](https://arxiv.org/abs/2310.08092)] [[Project Page](https://consistent-123.github.io/)]
* [arXiv 2023] **ImageDream: Image-Prompt Multi-view Diffusion for 3D Generation** [[Paper](https://arxiv.org/abs/2312.02201)] [[GitHub](https://github.com/bytedance/ImageDream)] [[Project Page](https://image-dream.github.io/)]
* [CVPR 2023] **RealFusion: 360° Reconstruction of Any Object from a Single Image** [[Paper](https://arxiv.org/abs/2302.10663)] [[GitHub](https://github.com/lukemelas/realfusion)] [[Project Page](https://lukemelas.github.io/realfusion/)]
* [ICCV 2023] **Zero-1-to-3: Zero-shot One Image to 3D Object** [[Paper](https://arxiv.org/abs/2303.11328)] [[GitHub](https://github.com/cvlab-columbia/zero123)] [[Project Page](https://zero123.cs.columbia.edu/)]
* [ICLR 2024 Poster] **Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors** [[Paper](https://arxiv.org/abs/2306.17843)] [[GitHub](https://github.com/guochengqian/Magic123)] [[Project Page](https://guochengqian.github.io/project/magic123/)]
* [ICLR 2024 Poster] **TOSS: High-quality Text-guided Novel View Synthesis from a Single Image** [[Paper](https://arxiv.org/abs/2310.10644)] [[GitHub](https://github.com/IDEA-Research/TOSS)] [[Project Page](https://toss3d.github.io/)]
* [ICLR 2024 Spotlight] **SyncDreamer: Generating Multiview-consistent Images from a Single-view Image** [[Paper](https://arxiv.org/abs/2309.03453)] [[GitHub](https://github.com/liuyuan-pal/SyncDreamer)] [[Project Page](https://liuyuan-pal.github.io/SyncDreamer/)]
* [CVPR 2024] **Wonder3D: Single Image to 3D using Cross-Domain Diffusion** [[Paper](https://arxiv.org/abs/2310.15008)]  [[GitHub](https://github.com/xxlong0/Wonder3D)] [[Project Page](https://www.xxlong.site/Wonder3D/)]
* [ICLR 2025] **IPDreamer: Appearance-Controllable 3D Object Generation with Complex Image Prompts** [[Paper](https://arxiv.org/pdf/2310.05375)] [[GitHub](https://github.com/zengbohan0217/IPDreamer)]

>##### MVS-based Approaches.
* [NeurIPS 2023] **One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization** [[Paper](https://arxiv.org/abs/2306.16928)] [[GitHub](https://github.com/One-2-3-45/One-2-3-45)] [[Project Page](https://one-2-3-45.github.io/)]
* [ECCV 2024] **CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model** [[Paper](https://arxiv.org/abs/2403.05034)] [[GitHub](https://github.com/thu-ml/CRM)] [[Project Page](https://ml.cs.tsinghua.edu.cn/~zhengyi/CRM/)]
* [arXiv 2024] **InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models** [[Paper](https://arxiv.org/abs/2404.07191)] [[GitHub](https://github.com/TencentARC/InstantMesh)]
* [ICLR 2024 Oral] **LRM: Large Reconstruction Model for Single Image to 3D** [[Paper](https://arxiv.org/abs/2311.04400)] [[Project Page](https://yiconghong.me/LRM/)]
* [NeurIPS 2024] **Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image** [[Paper](https://arxiv.org/abs/2405.20343)] [[GitHub](https://github.com/AiuniAI/Unique3D)] [[Project Page](https://wukailu.github.io/Unique3D/)]


##### Video-to-3D Generation.
* [CVPR 2024 Highlight] **ViVid-1-to-3: Novel View Synthesis with Video Diffusion Models** [[Paper](https://arxiv.org/abs/2312.01305)] [[GitHub](https://github.com/ubc-vision/vivid123)] [[Project Page](https://jgkwak95.github.io/ViVid-1-to-3/)]
* [ICML 2024] **IM-3D: Iterative Multiview Diffusion and Reconstruction for High-Quality 3D Generation** [[Paper](https://arxiv.org/abs/2402.08682)] [[Project Page](https://lukemelas.github.io/IM-3D/)]
* [arXiv 2024] **V3D: Video Diffusion Models are Effective 3D Generators** [[Paper](https://arxiv.org/abs/2403.06738)] [[GitHub](https://github.com/heheyas/V3D)] [[Project Page](https://heheyas.github.io/V3D/)]
* [ECCV 2024 Oral] **SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image Using Latent Video Diffusion** [[Paper](https://arxiv.org/abs/2403.12008)] [[Project Page](https://sv3d.github.io/)]
* [NeurIPS 2024 Oral] **CAT3D: Create Anything in 3D with Multi-View Diffusion Models** [[Paper](https://arxiv.org/abs/2405.10314)] [[Project Page](https://cat3d.github.io/)]

#### 3D Applications
>##### Avatar Generation.
* [CVPR 2023] **Zero-Shot Text-to-Parameter Translation for Game Character Auto-Creation** [[Paper](https://arxiv.org/abs/2303.01311)]
* [SIGGRAPH 2023] **DreamFace: Progressive Generation of Animatable 3D Faces under Text Guidance** [[Paper](https://arxiv.org/abs/2304.03117)] [[Project Page](https://sites.google.com/view/dreamface)]
* [NeurIPS 2023] **Headsculpt: Crafting 3d head avatars with text** [[Paper](https://arxiv.org/abs/2306.03038)] [[GitHub](https://github.com/BrandonHanx/HeadSculpt)] [[Project Page](https://brandonhan.uk/HeadSculpt/)]
* [NeurIPS 2023] **DreamWaltz: Make a Scene with Complex 3D Animatable Avatars** [[Paper](https://arxiv.org/abs/2305.12529)] [[GitHub](https://github.com/IDEA-Research/DreamWaltz)] [[Project Page](https://idea-research.github.io/DreamWaltz/)]
* [NeurIPS 2023 Spotlight] **DreamHuman: Animatable 3D Avatars from Text** [[Paper](https://arxiv.org/abs/2306.09329)] [[Project Page](https://dream-human.github.io/)]

>##### Scene Generation. 
* [ACM MM 2023] **RoomDreamer: Text-Driven 3D Indoor Scene Synthesis with Coherent Geometry and Texture** [[Paper](https://arxiv.org/abs/2305.11337)]
* [TVCG 2024] **Text2NeRF: Text-Driven 3D Scene Generation with Neural Radiance Fields** [[Paper](https://arxiv.org/abs/2305.11588)] [[GitHub](https://github.com/eckertzhang/Text2NeRF)] [[Project Page](https://eckertzhang.github.io/Text2NeRF.github.io/)]
* [ECCV 2024] **DreamScene: 3D Gaussian-based Text-to-3D Scene Generation via Formation Pattern Sampling** [[Paper](https://arxiv.org/abs/2404.03575)] [[GitHub](https://github.com/DreamScene-Project/DreamScene)] [[Project Page](https://dreamscene-project.github.io/)]
* [ECCV 2024] **DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting** [[Paper](https://arxiv.org/abs/2404.06903)] [[GitHub](https://github.com/ShijieZhou-UCLA/DreamScene360)] [[Project Page](https://dreamscene360.github.io/)]
* [arXiv 2024] **Urban Architect: Steerable 3D Urban Scene Generation with Layout Prior** [[Paper](https://arxiv.org/abs/2404.06780)] [[GitHub](https://github.com/UrbanArchitect/UrbanArchitect)] [[Project Page](https://urbanarchitect.github.io/)]
* [arXiv 2024] **CityCraft: A Real Crafter for 3D City Generation** [[Paper](https://arxiv.org/abs/2406.04983)] [[GitHub](https://github.com/djFatNerd/CityCraft)]

>##### 3D Editing. 
* [ECCV 2022] **Unified Implicit Neural Stylization** [[Paper](https://arxiv.org/abs/2204.01943)] [[GitHub](https://github.com/VITA-Group/INS)] [[Project Page](https://zhiwenfan.github.io/INS/)]
* [ECCV 2022] **ARF: Artistic Radiance Fields** [[Paper](https://arxiv.org/abs/2206.06360)] [[GitHub](https://github.com/Kai-46/ARF-svox2)] [[Project Page](https://www.cs.cornell.edu/projects/arf/)]
* [SIGGRAPH Asia 2022] **FDNeRF: Few-shot Dynamic Neural Radiance Fields for Face Reconstruction and Expression Editing** [[Paper](https://arxiv.org/abs/2208.05751)] [[GitHub](https://github.com/FDNeRF/FDNeRF)] [[Project Page](https://fdnerf.github.io/)]
* [CVPR 2022] **FENeRF: Face Editing in Neural Radiance Fields** [[Paper](https://arxiv.org/abs/2111.15490)] [[GitHub](https://github.com/MrTornado24/FENeRF)] [[Project Page](https://mrtornado24.github.io/FENeRF/)]
* [SIGGRAPH 2023] **TextDeformer: Geometry Manipulation using Text Guidance** [[Paper](https://arxiv.org/abs/2304.13348)] [[GitHub](https://github.com/threedle/TextDeformer)] [[Project Page](https://threedle.github.io/TextDeformer/)]
* [ICCV 2023] **ObjectSDF++: Improved Object-Compositional Neural Implicit Surfaces** [[Paper](https://arxiv.org/abs/2308.07868)] [[GitHub](https://github.com/QianyiWu/objectsdf_plus)] [[Project Page](https://wuqianyi.top/objectsdf++)] 
* [ICCV 2023 Oral] **Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions** [[Paper](https://arxiv.org/abs/2303.12789)] [[GitHub](https://github.com/ayaanzhaque/instruct-nerf2nerf)] [[Project Page](https://instruct-nerf2nerf.github.io/)] 

----

### 4D Generation

#### 4D Algorithms
>##### Feedforward Approaches.
* [CVPR 2024] **Control4D: Efficient 4D Portrait Editing with Text** [[Paper](https://arxiv.org/abs/2305.20082)] [[Project Page](https://control4darxiv.github.io/)]
* [NeurIPS 2024] **Animate3D: Animating Any 3D Model with Multi-view Video Diffusion** [[Paper](https://arxiv.org/abs/2407.11398)] [[GitHub](https://github.com/yanqinJiang/Animate3D)] [[Project Page](https://animate3d.github.io/)]
* [NeurIPS 2024] **Vidu4D: Single Generated Video to High-Fidelity 4D Reconstruction with Dynamic Gaussian Surfels** [[Paper](https://arxiv.org/abs/2405.16822)] [[GitHub](https://github.com/yikaiw/vidu4d)] [[Project Page](https://vidu4d-dgs.github.io/)]
* [NeurIPS 2024] **Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models** [[Paper](https://arxiv.org/abs/2405.16645)] [[GitHub](https://github.com/VITA-Group/Diffusion4D)] [[Project Page](https://vita-group.github.io/Diffusion4D/)] [[Dataset](https://huggingface.co/datasets/hw-liang/Diffusion4D)]
* [NeurIPS 2024] **L4GM: Large 4D Gaussian Reconstruction Model** [[Paper](https://arxiv.org/abs/2406.10324)] [[GitHub](https://github.com/nv-tlabs/L4GM-official)] [[Project Page](https://research.nvidia.com/labs/toronto-ai/l4gm/)] 

>##### Optimization-based Approaches.
* [arXiv 2023] **Text-To-4D Dynamic Scene Generation** [[Paper](https://arxiv.org/abs/2301.11280)] [[Project Page](https://make-a-video3d.github.io/)]
* [CVPR 2024] **4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling** [[Paper](https://arxiv.org/abs/2311.17984)] [[GitHub](https://github.com/sherwinbahmani/4dfy)] [[Project Page](https://sherwinbahmani.github.io/4dfy/)]
* [CVPR 2024] **A Unified Approach for Text- and Image-guided 4D Scene Generation** [[Paper](https://arxiv.org/abs/2311.16854)] [[GitHub](https://github.com/NVlabs/dream-in-4d)] [[Project Page](https://research.nvidia.com/labs/nxp/dream-in-4d/)]
* [CVPR 2024] **Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models** [[Paper](https://arxiv.org/abs/2312.13763)] [[Project Page](https://research.nvidia.com/labs/toronto-ai/AlignYourGaussians/)]
* [ECCV 2024] **TC4D: Trajectory-Conditioned Text-to-4D Generation** [[Paper](https://arxiv.org/abs/2403.17920)] [[GitHub](https://github.com/sherwinbahmani/tc4d)] [[Project Page](https://sherwinbahmani.github.io/tc4d/)]
* [ECCV 2024] **SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer** [[Paper](https://arxiv.org/abs/2404.03736)] [[GitHub](https://github.com/JarrentWu1031/SC4D)] [[Project Page](https://sc4d.github.io/)]
* [ECCV 2024] **STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians** [[Paper](https://arxiv.org/abs/2403.14939)] [[GitHub](https://github.com/zeng-yifei/STAG4D)] [[Project Page](https://nju-3dv.github.io/projects/STAG4D/)]
* [NeurIPS 2024] **4Real: Towards Photorealistic 4D Scene Generation via Video Diffusion Models** [[Paper](https://arxiv.org/abs/2406.07472)] [[Project Page](https://snap-research.github.io/4Real/)]
* [NeurIPS 2024] **Compositional 3D-aware Video Generation with LLM Director** [[Paper](https://arxiv.org/abs/2409.00558)] [[Project Page](https://www.microsoft.com/en-us/research/project/compositional-3d-aware-video-generation/)]
* [NeurIPS 2024] **DreamScene4D: Dynamic Multi-Object Scene Generation from Monocular Videos** [[Paper](https://arxiv.org/abs/2405.02280)] [[GitHub](https://github.com/dreamscene4d/dreamscene4d)] [[Project Page](https://dreamscene4d.github.io/)]
* [NeurIPS 2024] **DreamMesh4D: Video-to-4D Generation with Sparse-Controlled Gaussian-Mesh Hybrid Representation** [[Paper](https://arxiv.org/abs/2410.06756)] [[GitHub](https://github.com/WU-CVGL/DreamMesh4D)] [[Project Page](https://lizhiqi49.github.io/DreamMesh4D/)]
* [arXiv 2024] **Trans4D: Realistic Geometry-Aware Transition for Compositional Text-to-4D Synthesis** [[Paper](https://arxiv.org/pdf/2410.07155)] [[GitHub](https://github.com/YangLing0818/Trans4D)]

#### 4D Applications
>##### 4D Editing. 
* [CVPR 2024] **Control4D: Efficient 4D Portrait Editing with Text** [[Paper](https://arxiv.org/abs/2305.20082)] [[Project Page](https://control4darxiv.github.io/)]
* [CVPR 2024] **Instruct 4D-to-4D: Editing 4D Scenes as Pseudo-3D Scenes Using 2D Diffusion** [[Paper](https://arxiv.org/abs/2406.09402)] [[GitHub](https://github.com/Friedrich-M/Instruct-4D-to-4D/)] [[Project Page](https://immortalco.github.io/Instruct-4D-to-4D/)]

>##### Human Animation.
* [SIGGRAPH 2020] **Robust Motion In-betweening** [[Paper](https://arxiv.org/abs/2102.04942)]
* [CVPR 2022] **Generating Diverse and Natural 3D Human Motions from Text** [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.pdf)] [[GitHub](https://github.com/EricGuo5513/text-to-motion)] [[Project Page](https://ericguo5513.github.io/text-to-motion/)]
* [SCA 2023] **Motion In-Betweening with Phase Manifolds** [[Paper](https://arxiv.org/abs/2308.12751)] [[GitHub](https://github.com/paulstarke/PhaseBetweener)]
* [CVPR 2023] **T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations** [[Paper](https://arxiv.org/abs/2301.06052)] [[GitHub](https://github.com/Mael-zys/T2M-GPT)] [[Project Page](https://mael-zys.github.io/T2M-GPT/)]
* [ICLR 2023 notable top 25%] **Human Motion Diffusion Model** [[Paper](https://arxiv.org/abs/2209.14916)] [[GitHub](https://github.com/GuyTevet/motion-diffusion-model)] [[Project Page](https://guytevet.github.io/mdm-page/)]
* [NeurIPS 2023] **MotionGPT: Human Motion as a Foreign Language** [[Paper](https://arxiv.org/abs/2306.14795)] [[GitHub](https://github.com/OpenMotionLab/MotionGPT)] [[Project Page](https://motion-gpt.github.io/)]
* [ICML 2024] **HumanTOMATO: Text-aligned Whole-body Motion Generation** [[Paper](https://arxiv.org/abs/2310.12978)] [[GitHub](https://github.com/IDEA-Research/HumanTOMATO)] [[Project Page](https://lhchen.top/HumanTOMATO/)]
* [CVPR 2024] **MoMask: Generative Masked Modeling of 3D Human Motions** [[Paper](https://arxiv.org/abs/2312.00063)] [[GitHub](https://github.com/EricGuo5513/momask-codes)] [[Project Page](https://ericguo5513.github.io/momask/)]
* [CVPR 2024] **Lodge: A Coarse to Fine Diffusion Network for Long Dance Generation Guided by the Characteristic Dance Primitives** [[Paper](https://arxiv.org/abs/2403.10518)] [[GitHub](https://github.com/li-ronghui/LODGE)] [[Project Page](https://li-ronghui.github.io/lodge)]



## Other Related Resources

### World Model Benchmark
* [arXiv 2025] **WorldModelBench: Judging Video Generation Models As World Models** [[Paper](https://arxiv.org/abs/2502.20694)] [[GitHub](https://github.com/WorldModelBench-Team/WorldModelBench)] [[Project Page](https://worldmodelbench-team.github.io/)]

### World Foundation Model Platform
- [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) ([[GitHub](https://github.com/nvidia-cosmos)] [[Paper](https://arxiv.org/abs/2501.03575)]): NVIDIA Cosmos is a world foundation model platform for accelerating the development of physical AI systems.
  
	- [Cosmos-Transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1)：a world-to-world transfer model designed to bridge the perceptual divide between simulated and real-world environments.
   	- [Cosmos-Predict1](https://github.com/nvidia-cosmos/cosmos-predict1): a collection of general-purpose world foundation models for Physical AI that can be fine-tuned into customized world models for downstream applications.
   	- [Cosmos-Reason1](https://github.com/nvidia-cosmos/cosmos-reason1)： a model that understands the physical common sense and generate appropriate embodied decisions in natural language through long chain-of-thought reasoning processes.
 
- [Genie3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/), Google Deepmind, August 5th, 2025.

-------

[<u>🎯Back to Top - Our Survey Paper Collection</u>](#-our-survey-paper-collection)

## 🔥 Awesome Text2X Resources

An open collection of state-of-the-art (SOTA), novel **Text to X (X can be everything)** methods (papers, codes and datasets), intended to keep pace with the anticipated surge of research.

<div><div align="center">
	<img width="500" height="350" src="media/logo.svg" alt="Awesome"></div>



## Update Logs

* `2025.04.18` - update layout on section [Related Resources](#related-resources).

  
<details span>
<summary><b>2025 Update Logs:</b></summary>

* `2025.05.08` - update new layout.
* `2025.03.10` - [CVPR 2025 Accepted Papers](https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers)🎉
* `2025.02.28` - update several papers status "CVPR 2025" to accepted papers, congrats to all 🎉
* `2025.01.23` - update several papers status "ICLR 2025" to accepted papers, congrats to all 🎉
* `2025.01.09` - update layout.
  
</details>

<details close>
<summary><b>Previous 2024 Update Logs:</b></summary>

* `2024.12.21` adjusted the layouts of several sections and _Happy Winter Solstice_ ⚪🥣.
* `2024.09.26` - update several papers status "NeurIPS 2024" to accepted papers, congrats to all 🎉
* `2024.09.03` - add one new section 'text to model'.
* `2024.06.30` - add one new section 'text to video'.	
* `2024.07.02` - update several papers status "ECCV 2024" to accepted papers, congrats to all 🎉
* `2024.06.21` - add one hot Topic about _AIGC 4D Generation_ on the section of __Suvery and Awesome Repos__.
* `2024.06.17` - an awesome repo for CVPR2024 [Link](https://github.com/52CV/CVPR-2024-Papers) 👍🏻
* `2024.04.05` adjusted the layout and added accepted lists and ArXiv lists to each section.
* `2024.04.05` - an awesome repo for CVPR2024 on 3DGS and NeRF [Link](https://github.com/Yubel426/NeRF-3DGS-at-CVPR-2024) 👍🏻
* `2024.03.25` - add one new survey paper of 3D GS into the section of "Survey and Awesome Repos--Topic 1: 3D Gaussian Splatting".
* `2024.03.12` - add a new section "Dynamic Gaussian Splatting", including Neural Deformable 3D Gaussians, 4D Gaussians, Dynamic 3D Gaussians.
* `2024.03.11` - CVPR 2024 Accpeted Papers [Link](https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers) 
* update some papers accepted by CVPR 2024! Congratulations🎉
  
</details>

## Text to 4D
(Also, Image/Video to 4D)

### 🎉 4D Accepted Papers
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **Optimizing 4D Gaussians for Dynamic Scene Video from Single Landscape Images**  | ICLR 2025 |          [Link](https://arxiv.org/abs/2504.05458)          | [Link](https://github.com/cvsp-lab/ICLR2025_3D-MOM)  | [Link](https://paper.pnu-cvsp.com/ICLR2025_3D-MOM/)  |
| 2025 | **GS-DiT: Advancing Video Generation with Pseudo 4D Gaussian Fields through Efficient Dense 3D Point Tracking**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2501.02690)          | [Link](https://github.com/wkbian/GS-DiT)  | [Link](https://wkbian.github.io/Projects/GS-DiT/)  |
| 2025 | **Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos**  | CVPR 2025 Oral |          [Link](https://arxiv.org/abs/2412.09621)          | [Link](https://github.com/Stereo4d/stereo4d-code)  | [Link](https://stereo4d.github.io/)  |
| 2025 | **Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video**  | CVPR 2025 Highlight |          [Link](https://arxiv.org/abs/2503.21761v1)          | [Link](https://github.com/Davidyao99/uni4d/tree/main)  | [Link](https://davidyao99.github.io/uni4d/)  |
| 2025 | **4D-Fly: Fast 4D Reconstruction from a Single Monocular Video**  | CVPR 2025 |          [Link](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_4D-Fly_Fast_4D_Reconstruction_from_a_Single_Monocular_Video_CVPR_2025_paper.pdf)          | Coming Soon!  | [Link](https://diankun-wu.github.io/4D-Fly/)  |
| 2025 | **GenMOJO: Robust Multi-Object 4D Generation for In-the-wild Videos**  | CVPR 2025 |          [Link](https://www.arxiv.org/abs/2506.12716)          | [Link](https://github.com/genmojo/GenMOJO)  | [Link](https://genmojo.github.io/)  |
| 2025 | **Articulated Kinematics Distillation from Video Diffusion Models**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2504.01204)          | --  | [Link](https://research.nvidia.com/labs/dir/akd/)  |
| 2025 | **Free4D: Tuning-free 4D Scene Generation with Spatial-Temporal Consistency**  | ICCV 2025 |          [Link](https://arxiv.org/abs/2503.20785)          | [Link](https://github.com/TQTQliu/Free4D)  | [Link](https://free4d.github.io/)  |
| 2025 | **St4RTrack: Simultaneous 4D Reconstruction and Tracking in the World**  | ICCV 2025 |          [Link](https://arxiv.org/abs/2504.13152)          | [Link](https://github.com/HavenFeng/St4RTrack)  | [Link](https://st4rtrack.github.io/)  |
| 2025 | **VLM4D: Towards Spatiotemporal Awareness in Vision Language Models**  | ICCV 2025 |          [Link](https://arxiv.org/pdf/2508.02095)          | [Link](https://github.com/ShijieZhou-UCLA/VLM4D)  | [Link](https://vlm4d.github.io/)  |
| 2025 | **Express4D: Expressive, Friendly, and Extensible 4D Facial Motion Generation Benchmark**  | ICCV DataCV Workshop 2025 |          [Link](https://arxiv.org/abs/2508.12438)          | [Link](https://github.com/jaron1990/Express4D/)  | [Link](https://jaron1990.github.io/Express4D/)  |
| 2025 | **CityDreamer4D: Compositional Generative Model of Unbounded 4D Cities**  | TPAMI 2025 |          [Link](https://arxiv.org/abs/2501.08983)          | [Link](https://github.com/hzxie/CityDreamer4D?tab=readme-ov-file)  | [Link](https://www.infinitescript.com/project/city-dreamer-4d/)  |
| 2025 | **TesserAct: Learning 4D Embodied World Models**  | ICCV 2025 |   [Link](https://arxiv.org/abs/2504.20995)          | [Link](https://github.com/UMass-Embodied-AGI/TesserAct)  | [Link](https://tesseractworld.github.io/)  |
| 2025 | **T2Bs: Text-to-Character Blendshapes via Video Generation**  | ICCV 2025 |   [Link](https://arxiv.org/abs/2509.10678)          | --  | [Link](https://snap-research.github.io/T2Bs/)  |
| 2025 | **Geo4D: Leveraging Video Generators for Geometric 4D Scene Reconstruction**  | ICCV 2025 Highlight |          [Link](https://arxiv.org/abs/2504.07961)          | [Link](https://github.com/jzr99/Geo4D)  | [Link](https://geo4d.github.io/)  |
| 2025 | **SV4D 2.0: Enhancing Spatio-Temporal Consistency in Multi-View Video Diffusion for High-Quality 4D Generation**  | ICCV 2025 |          [Link](https://arxiv.org/abs/2503.16396)          | [Link](https://github.com/Stability-AI/generative-models)  | [Link](https://sv4d20.github.io/)  |
| 2025 | **HoloTime: Taming Video Diffusion Models for Panoramic 4D Scene Generation**  | ACM MM 2025 |          [Link](https://arxiv.org/abs/2504.21650)          | [Link](https://github.com/PKU-YuanGroup/HoloTime)  | [Link](https://zhouhyocean.github.io/holotime/)  |
| 2025 | **Stable Part Diffusion 4D: Multi-View RGB and Kinematic Parts Video Generation**  | NeurIPS 2025 |          [Link](https://arxiv.org/abs/2509.10687)          | Coming Soon! | [Link](https://stablepartdiffusion4d.github.io/)  |
| 2025 | **In-2-4D: Inbetweening from Two Single-View Images to 4D Generation**  | SIGGRAPH ASIA 2025 |          [Link](https://arxiv.org/abs/2504.08366)          | [Link](https://github.com/sauradip/In-2-4D)  | [Link](https://in-2-4d.github.io/)  |
| 2025 | **Track, Inpaint, Resplat: Subject-driven 3D and 4D Generation with Progressive Texture Infilling**  | NeurIPS 2025 |          [Link](https://arxiv.org/abs/2510.23605)          | -- | [Link](https://zsh2000.github.io/track-inpaint-resplat.github.io/)  |

<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

@inproceedings{jinoptimizing,
  title={Optimizing 4D Gaussians for Dynamic Scene Video from Single Landscape Images},
  author={Jin, In-Hwan and Choo, Haesoo and Jeong, Seong-Hun and Heemoon, Park and Kim, Junghwan and Kwon, Oh-joon and Kong, Kyeongbo},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}

@article{bian2025gsdit,
  title={GS-DiT: Advancing Video Generation with Pseudo 4D Gaussian Fields through Efficient Dense 3D Point Tracking},
  author={Bian, Weikang and Huang, Zhaoyang and Shi, Xiaoyu and and Li, Yijin and Wang, Fu-Yun and Li, Hongsheng},
  journal={arXiv preprint arXiv:2501.02690},
  year={2025}
}

@article{jin2024stereo4d,
  title={Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos}, 
  author={Jin, Linyi and Tucker, Richard and Li, Zhengqi and Fouhey, David and Snavely, Noah and Holynski, Aleksander},
  journal={CVPR},
  year={2025},
}

@article{yao2025uni4d,
  title={Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video},
  author={Yao, David Yifan and Zhai, Albert J and Wang, Shenlong},
  journal={arXiv preprint arXiv:2503.21761},
  year={2025}
}

@inproceedings{wu20254d,
  title={4D-Fly: Fast 4D Reconstruction from a Single Monocular Video},
  author={Wu, Diankun and Liu, Fangfu and Hung, Yi-Hsin and Qian, Yue and Zhan, Xiaohang and Duan, Yueqi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={16663--16673},
  year={2025}
}

@inproceedings{chu2025robust,
  title={Robust Multi-Object 4D Generation for In-the-wild Videos},
  author={Chu, Wen-Hsuan and Ke, Lei and Liu, Jianmeng and Huo, Mingxiao and Tokmakov, Pavel and Fragkiadaki, Katerina},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={22067--22077},
  year={2025}
}

@inproceedings{li2025articulated,
  title={Articulated Kinematics Distillation from Video Diffusion Models},
  author={Li, Xuan and Ma, Qianli and Lin, Tsung-Yi and Chen, Yongxin and Jiang, Chenfanfu and Liu, Ming-Yu and Xiang, Donglai},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={17571--17581},
  year={2025}
}

@article{liu2025free4d,
     title={Free4D: Tuning-free 4D Scene Generation with Spatial-Temporal Consistency},
     author={Liu, Tianqi and Huang, Zihao and Chen, Zhaoxi and Wang, Guangcong and Hu, Shoukang and Shen, liao and Sun, Huiqiang and Cao, Zhiguo and Li, Wei and Liu, Ziwei},
     journal={arXiv preprint arXiv:2503.20785},
     year={2025}
}

@inproceedings{st4rtrack2025,
  title={St4RTrack: Simultaneous 4D Reconstruction and Tracking in the World},
  author={Feng*, Haiwen and Zhang*, Junyi and Wang, Qianqian and Ye, Yufei and Yu, Pengcheng and Black, Michael J. and Darrell, Trevor and Kanazawa, Angjoo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}

@inproceedings{zhou2025vlm4d,
    title={VLM4D: Towards Spatiotemporal Awareness in Vision Language Models},
    author={Zhou, Shijie and Vilesov, Alexander and He, Xuehai and Wan, Ziyu and Zhang, Shuwang and Nagachandra, Aditya and Chang, Di and Chen, Dongdong and Wang, Eric Xin and Kadambi, Achuta},
    booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
    year={2025}
}

@misc{aloni2025express4dexpressivefriendlyextensible,
title={Express4D: Expressive, Friendly, and Extensible 4D Facial Motion Generation Benchmark}, 
author={Yaron Aloni and Rotem Shalev-Arkushin and Yonatan Shafir and Guy Tevet and Ohad Fried and Amit Haim Bermano},
year={2025},
eprint={2508.12438},
archivePrefix={arXiv},
primaryClass={cs.GR},
url={https://arxiv.org/abs/2508.12438}
}

@article{xie2025citydreamer4d,
  title={CityDreamer4D: Compositional generative model of unbounded 4D cities},
  author={Xie, Haozhe and Chen, Zhaoxi and Hong, Fangzhou and Liu, Ziwei},
  journal={arXiv e-prints},
  pages={arXiv--2501},
  year={2025}
}

@article{zhen2025tesseract,
  title={TesserAct: learning 4D embodied world models},
  author={Zhen, Haoyu and Sun, Qiao and Zhang, Hongxin and Li, Junyan and Zhou, Siyuan and Du, Yilun and Gan, Chuang},
  journal={arXiv preprint arXiv:2504.20995},
  year={2025}
}

@misc{luo2025t2bstexttocharacterblendshapesvideo,
      title={T2Bs: Text-to-Character Blendshapes via Video Generation}, 
      author={Jiahao Luo and Chaoyang Wang and Michael Vasilkovsky and Vladislav Shakhrai and Di Liu and Peiye Zhuang and Sergey Tulyakov and Peter Wonka and Hsin-Ying Lee and James Davis and Jian Wang},
      year={2025},
      eprint={2509.10678},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2509.10678}, 
}

@article{jiang2025geo4d,
  title={Geo4d: Leveraging video generators for geometric 4d scene reconstruction},
  author={Jiang, Zeren and Zheng, Chuanxia and Laina, Iro and Larlus, Diane and Vedaldi, Andrea},
  journal={arXiv preprint arXiv:2504.07961},
  year={2025}
}

@article{yao2025sv4d,
  title={Sv4d 2.0: Enhancing spatio-temporal consistency in multi-view video diffusion for high-quality 4d generation},
  author={Yao, Chun-Han and Xie, Yiming and Voleti, Vikram and Jiang, Huaizu and Jampani, Varun},
  journal={arXiv preprint arXiv:2503.16396},
  year={2025}
}

@article{zhou2025holotime,
  title={HoloTime: Taming Video Diffusion Models for Panoramic 4D Scene Generation},
  author={Zhou, Haiyang and Yu, Wangbo and Guan, Jiawen and Cheng, Xinhua and Tian, Yonghong and Yuan, Li},
  journal={arXiv preprint arXiv:2504.21650},
  year={2025}
}

@article{zhang2025stable,
  title={Stable Part Diffusion 4D: Multi-View RGB and Kinematic Parts Video Generation},
  author={Zhang, Hao and Yao, Chun-Han and Donn{\'e}, Simon and Ahuja, Narendra and Jampani, Varun},
  journal={arXiv preprint arXiv:2509.10687},
  year={2025}
}

@article{nag20252,
  title={In-2-4d: Inbetweening from two single-view images to 4d generation},
  author={Nag, Sauradip and Cohen-Or, Daniel and Zhang, Hao and Mahdavi-Amiri, Ali},
  journal={arXiv preprint arXiv:2504.08366},
  year={2025}
}

@inproceedings{zheng2025trackinpaintresplat,
  title={Track, Inpaint, Resplat: Subject-driven 3D and 4D Generation with Progressive Texture Infilling},
  author={Zheng, Shuhong and Mirzaei, Ashkan and Gilitschenski, Igor},
  booktitle={NeurIPS},
  year={2025}
}

```
</details>

-------

### 💡 4D ArXiv Papers

#### 1. AR4D: Autoregressive 4D Generation from Monocular Videos
Hanxin Zhu, Tianyu He, Xiqian Yu, Junliang Guo, Zhibo Chen, Jiang Bian (University of Science and Technology of China, Microsoft Research Asia)
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in generative models have ignited substantial interest in dynamic 3D content creation (\ie, 4D generation). Existing approaches primarily rely on Score Distillation Sampling (SDS) to infer novel-view videos, typically leading to issues such as limited diversity, spatial-temporal inconsistency and poor prompt alignment, due to the inherent randomness of SDS. To tackle these problems, we propose AR4D, a novel paradigm for SDS-free 4D generation. Specifically, our paradigm consists of three stages. To begin with, for a monocular video that is either generated or captured, we first utilize pre-trained expert models to create a 3D representation of the first frame, which is further fine-tuned to serve as the canonical space. Subsequently, motivated by the fact that videos happen naturally in an autoregressive manner, we propose to generate each frame's 3D representation based on its previous frame's representation, as this autoregressive generation manner can facilitate more accurate geometry and motion estimation. Meanwhile, to prevent overfitting during this process, we introduce a progressive view sampling strategy, utilizing priors from pre-trained large-scale 3D reconstruction models. To avoid appearance drift introduced by autoregressive generation, we further incorporate a refinement stage based on a global deformation field and the geometry of each frame's 3D representation. Extensive experiments have demonstrated that AR4D can achieve state-of-the-art 4D generation without SDS, delivering greater diversity, improved spatial-temporal consistency, and better alignment with input prompts.
</details>

#### 2. WideRange4D: Enabling High-Quality 4D Reconstruction with Wide-Range Movements and Scenes
Ling Yang, Kaixin Zhu, Juanxi Tian, Bohan Zeng, Mingbao Lin, Hongjuan Pei, Wentao Zhang, Shuicheng Yan 

(Peking University, University of the Chinese Academy of Sciences, National University of Singapore)
<details span>
<summary><b>Abstract</b></summary>
With the rapid development of 3D reconstruction technology, research in 4D reconstruction is also advancing, existing 4D reconstruction methods can generate high-quality 4D scenes. However, due to the challenges in acquiring multi-view video data, the current 4D reconstruction benchmarks mainly display actions performed in place, such as dancing, within limited scenarios. In practical scenarios, many scenes involve wide-range spatial movements, highlighting the limitations of existing 4D reconstruction datasets. Additionally, existing 4D reconstruction methods rely on deformation fields to estimate the dynamics of 3D objects, but deformation fields struggle with wide-range spatial movements, which limits the ability to achieve high-quality 4D scene reconstruction with wide-range spatial movements. In this paper, we focus on 4D scene reconstruction with significant object spatial movements and propose a novel 4D reconstruction benchmark, WideRange4D. This benchmark includes rich 4D scene data with large spatial variations, allowing for a more comprehensive evaluation of the generation capabilities of 4D generation methods. Furthermore, we introduce a new 4D reconstruction method, Progress4D, which generates stable and high-quality 4D results across various complex 4D scene reconstruction tasks. We conduct both quantitative and qualitative comparison experiments on WideRange4D, showing that our Progress4D outperforms existing state-of-the-art 4D reconstruction methods. 
</details>

#### 3. TwoSquared: 4D Generation from 2D Image Pairs
Lu Sang, Zehranaz Canfes, Dongliang Cao, Riccardo Marin, Florian Bernard, Daniel Cremers

(Technical University of Munich, Munich Center of Machine Learning, University of Bonn)
<details span>
<summary><b>Abstract</b></summary>
Despite the astonishing progress in generative AI, 4D dynamic object generation remains an open challenge. With limited high-quality training data and heavy computing requirements, the combination of hallucinating unseen geometry together with unseen movement poses great challenges to generative models. In this work, we propose TwoSquared as a method to obtain a 4D physically plausible sequence starting from only two 2D RGB images corresponding to the beginning and end of the action. Instead of directly solving the 4D generation problem, TwoSquared decomposes the problem into two steps: 1) an image-to-3D module generation based on the existing generative model trained on high-quality 3D assets, and 2) a physically inspired deformation module to predict intermediate movements. To this end, our method does not require templates or object-class-specific prior knowledge and can take in-the-wild images as input. In our experiments, we demonstrate that TwoSquared is capable of producing texture-consistent and geometry-consistent 4D sequences only given 2D images.
</details>

#### 4. DeepVerse: 4D Autoregressive Video Generation as a World Model
Junyi Chen, Haoyi Zhu, Xianglong He, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Zhoujie Fu, Jiangmiao Pang, Tong He

(Shanghai Jiao Tong University, Shanghai AI Lab, University of Science and Technology of China, Tsinghua University, Zhejiang University, Fudan University, Nanyang Technology University)
<details span>
<summary><b>Abstract</b></summary>
World models serve as essential building blocks toward Artificial General Intelligence (AGI), enabling intelligent agents to predict future states and plan actions by simulating complex physical interactions. However, existing interactive models primarily predict visual observations, thereby neglecting crucial hidden states like geometric structures and spatial coherence. This leads to rapid error accumulation and temporal inconsistency. To address these limitations, we introduce DeepVerse, a novel 4D interactive world model explicitly incorporating geometric predictions from previous timesteps into current predictions conditioned on actions. Experiments demonstrate that by incorporating explicit geometric constraints, DeepVerse captures richer spatio-temporal relationships and underlying physical dynamics. This capability significantly reduces drift and enhances temporal consistency, enabling the model to reliably generate extended future sequences and achieve substantial improvements in prediction accuracy, visual realism, and scene rationality. Furthermore, our method provides an effective solution for geometry-aware memory retrieval, effectively preserving long-term spatial consistency. We validate the effectiveness of DeepVerse across diverse scenarios, establishing its capacity for high-fidelity, long-horizon predictions grounded in geometry-aware dynamics.
</details>

#### 5. Sonic4D: Spatial Audio Generation for Immersive 4D Scene Exploration
Siyi Xie, Hanxin Zhu, Tianyu He, Xin Li, Zhibo Chen (University of Science and Technology of China)

<details span>
<summary><b>Abstract</b></summary>
Recent advancements in 4D generation have demonstrated its remarkable capability in synthesizing photorealistic renderings of dynamic 3D scenes. However, despite achieving impressive visual performance, almost all existing methods overlook the generation of spatial audio aligned with the corresponding 4D scenes, posing a significant limitation to truly immersive audiovisual experiences. To mitigate this issue, we propose Sonic4D, a novel framework that enables spatial audio generation for immersive exploration of 4D scenes. Specifically, our method is composed of three stages: 1) To capture both the dynamic visual content and raw auditory information from a monocular video, we first employ pre-trained expert models to generate the 4D scene and its corresponding monaural audio. 2) Subsequently, to transform the monaural audio into spatial audio, we localize and track the sound sources within the 4D scene, where their 3D spatial coordinates at different timestamps are estimated via a pixel-level visual grounding strategy. 3) Based on the estimated sound source locations, we further synthesize plausible spatial audio that varies across different viewpoints and timestamps using physics-based simulation. Extensive experiments have demonstrated that our proposed method generates realistic spatial audio consistent with the synthesized 4D scene in a training-free manner, significantly enhancing the immersive experience for users.
</details>

#### 6. 4Real-Video-V2: Fused View-Time Attention and Feedforward Reconstruction for 4D Scene Generation
Chaoyang Wang, Ashkan Mirzaei, Vidit Goel, Willi Menapace, Aliaksandr Siarohin, Avalon Vinella, Michael Vasilkovsky, Ivan Skorokhodov, Vladislav Shakhrai, Sergey Korolev, Sergey Tulyakov, Peter Wonka

(Snap Inc., KAUST)

<details span>
<summary><b>Abstract</b></summary>
We propose the first framework capable of computing a 4D spatio-temporal grid of video frames and 3D Gaussian particles for each time step using a feed-forward architecture. Our architecture has two main components, a 4D video model and a 4D reconstruction model. In the first part, we analyze current 4D video diffusion architectures that perform spatial and temporal attention either sequentially or in parallel within a two-stream design. We highlight the limitations of existing approaches and introduce a novel fused architecture that performs spatial and temporal attention within a single layer. The key to our method is a sparse attention pattern, where tokens attend to others in the same frame, at the same timestamp, or from the same viewpoint. In the second part, we extend existing 3D reconstruction algorithms by introducing a Gaussian head, a camera token replacement algorithm, and additional dynamic layers and training. Overall, we establish a new state of the art for 4D generation, improving both visual quality and reconstruction capability.
</details>

#### 7. BulletGen: Improving 4D Reconstruction with Bullet-Time Generation
Denys Rozumnyi, Jonathon Luiten, Numair Khan, Johannes Schönberger, Peter Kontschieder (Meta Reality Labs)

<details span>
<summary><b>Abstract</b></summary>
Transforming casually captured, monocular videos into fully immersive dynamic experiences is a highly ill-posed task, and comes with significant challenges, e.g., reconstructing unseen regions, and dealing with the ambiguity in monocular depth estimation. In this work we introduce BulletGen, an approach that takes advantage of generative models to correct errors and complete missing information in a Gaussian-based dynamic scene representation. This is done by aligning the output of a diffusion-based video generation model with the 4D reconstruction at a single frozen "bullet-time" step. The generated frames are then used to supervise the optimization of the 4D Gaussian model. Our method seamlessly blends generative content with both static and dynamic scene components, achieving state-of-the-art results on both novel-view synthesis, and 2D/3D tracking tasks.
</details>

#### 8. 4D-LRM: Large Space-Time Reconstruction Model From and To Any View at Any Time
Ziqiao Ma, Xuweiyi Chen, Shoubin Yu, Sai Bi, Kai Zhang, Chen Ziwen, Sihan Xu, Jianing Yang, Zexiang Xu, Kalyan Sunkavalli, Mohit Bansal, Joyce Chai, Hao Tan

(Adobe Research, University of Michigan, UNC Chapel Hill, University of Virginia, Oregon State University)

<details span>
<summary><b>Abstract</b></summary>
Can we scale 4D pretraining to learn general space-time representations that reconstruct an object from a few views at some times to any view at any time? We provide an affirmative answer with 4D-LRM, the first large-scale 4D reconstruction model that takes input from unconstrained views and timestamps and renders arbitrary novel view-time combinations. Unlike prior 4D approaches, e.g., optimization-based, geometry-based, or generative, that struggle with efficiency, generalization, or faithfulness, 4D-LRM learns a unified space-time representation and directly predicts per-pixel 4D Gaussian primitives from posed image tokens across time, enabling fast, high-quality rendering at, in principle, infinite frame rate. Our results demonstrate that scaling spatiotemporal pretraining enables accurate and efficient 4D reconstruction. We show that 4D-LRM generalizes to novel objects, interpolates across time, and handles diverse camera setups. It reconstructs 24-frame sequences in one forward pass with less than 1.5 seconds on a single A100 GPU.
</details>

#### 9. MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second
Chenguo Lin, Yuchen Lin, Panwang Pan, Yifan Yu, Honglei Yan, Katerina Fragkiadaki, Yadong Mu (Peking University, ByteDance, Carnegie Mellon University)

<details span>
<summary><b>Abstract</b></summary>
We present MoVieS, a novel feed-forward model that synthesizes 4D dynamic novel views from monocular videos in one second. MoVieS represents dynamic 3D scenes using pixel-aligned grids of Gaussian primitives, explicitly supervising their time-varying motion. This allows, for the first time, the unified modeling of appearance, geometry and motion, and enables view synthesis, reconstruction and 3D point tracking within a single learning-based framework. By bridging novel view synthesis with dynamic geometry reconstruction, MoVieS enables large-scale training on diverse datasets with minimal dependence on task-specific supervision. As a result, it also naturally supports a wide range of zero-shot applications, such as scene flow estimation and moving object segmentation. Extensive experiments validate the effectiveness and efficiency of MoVieS across multiple tasks, achieving competitive performance while offering several orders of magnitude speedups.
</details>

#### 10. 4DNeX: Feed-Forward 4D Generative Modeling Made Easy
Zhaoxi Chen, Tianqi Liu, Long Zhuo, Jiawei Ren, Zeng Tao, He Zhu, Fangzhou Hong, Liang Pan, Ziwei Liu 

(Nanyang Technological University, Shanghai AI Laboratory)

<details span>
<summary><b>Abstract</b></summary>
We present 4DNeX, the first feed-forward framework for generating 4D (i.e., dynamic 3D) scene representations from a single image. In contrast to existing methods that rely on computationally intensive optimization or require multi-frame video inputs, 4DNeX enables efficient, end-to-end image-to-4D generation by fine-tuning a pretrained video diffusion model. Specifically, 1) to alleviate the scarcity of 4D data, we construct 4DNeX-10M, a large-scale dataset with high-quality 4D annotations generated using advanced reconstruction approaches. 2) we introduce a unified 6D video representation that jointly models RGB and XYZ sequences, facilitating structured learning of both appearance and geometry. 3) we propose a set of simple yet effective adaptation strategies to repurpose pretrained video diffusion models for 4D modeling. 4DNeX produces high-quality dynamic point clouds that enable novel-view video synthesis. Extensive experiments demonstrate that 4DNeX outperforms existing 4D generation methods in efficiency and generalizability, offering a scalable solution for image-to-4D modeling and laying the foundation for generative 4D world models that simulate dynamic scene evolution.
</details>

#### 11. OmniWorld: A Multi-Domain and Multi-Modal Dataset for 4D World Modeling
Yang Zhou, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Haoyu Guo, Zizun Li, Kaijing Ma, Xinyue Li, Yating Wang, Haoyi Zhu, Mingyu Liu, Dingning Liu, Jiange Yang, Zhoujie Fu, Junyi Chen, Chunhua Shen, Jiangmiao Pang, Kaipeng Zhang, Tong He

(Shanghai AI Laboratory, ZJU)

<details span>
<summary><b>Abstract</b></summary>
The field of 4D world modeling - aiming to jointly capture spatial geometry and temporal dynamics - has witnessed remarkable progress in recent years, driven by advances in large-scale generative models and multimodal learning. However, the development of truly general 4D world models remains fundamentally constrained by the availability of high-quality data. Existing datasets and benchmarks often lack the dynamic complexity, multi-domain diversity, and spatial-temporal annotations required to support key tasks such as 4D geometric reconstruction, future prediction, and camera-control video generation. To address this gap, we introduce OmniWorld, a large-scale, multi-domain, multi-modal dataset specifically designed for 4D world modeling. OmniWorld consists of a newly collected OmniWorld-Game dataset and several curated public datasets spanning diverse domains. Compared with existing synthetic datasets, OmniWorld-Game provides richer modality coverage, larger scale, and more realistic dynamic interactions. Based on this dataset, we establish a challenging benchmark that exposes the limitations of current state-of-the-art (SOTA) approaches in modeling complex 4D environments. Moreover, fine-tuning existing SOTA methods on OmniWorld leads to significant performance gains across 4D reconstruction and video generation tasks, strongly validating OmniWorld as a powerful resource for training and evaluation. We envision OmniWorld as a catalyst for accelerating the development of general-purpose 4D world models, ultimately advancing machines' holistic understanding of the physical world.
</details>

#### 12. Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation
Sherwin Bahmani, Tianchang Shen, Jiawei Ren, Jiahui Huang, Yifeng Jiang, Haithem Turki, Andrea Tagliasacchi, David B. Lindell, Zan Gojcic, Sanja Fidler, Huan Ling, Jun Gao, Xuanchi Ren

(NVIDIA, University of Toronto, Vector Institute, Simon Fraser University)

<details span>
<summary><b>Abstract</b></summary>
The ability to generate virtual environments is crucial for applications ranging from gaming to physical AI domains such as robotics, autonomous driving, and industrial AI. Current learning-based 3D reconstruction methods rely on the availability of captured real-world multi-view data, which is not always readily available. Recent advancements in video diffusion models have shown remarkable imagination capabilities, yet their 2D nature limits the applications to simulation where a robot needs to navigate and interact with the environment. In this paper, we propose a self-distillation framework that aims to distill the implicit 3D knowledge in the video diffusion models into an explicit 3D Gaussian Splatting (3DGS) representation, eliminating the need for multi-view training data. Specifically, we augment the typical RGB decoder with a 3DGS decoder, which is supervised by the output of the RGB decoder. In this approach, the 3DGS decoder can be purely trained with synthetic data generated by video diffusion models. At inference time, our model can synthesize 3D scenes from either a text prompt or a single image for real-time rendering. Our framework further extends to dynamic 3D scene generation from a monocular input video. Experimental results show that our framework achieves state-of-the-art performance in static and dynamic 3D scene generation.
</details>

#### 13. ShapeGen4D: Towards High Quality 4D Shape Generation from Videos
Jiraphon Yenphraphai, Ashkan Mirzaei, Jianqi Chen, Jiaxu Zou, Sergey Tulyakov, Raymond A. Yeh, Peter Wonka, Chaoyang Wang

(Snap, Purdue University, KAUST)

<details span>
<summary><b>Abstract</b></summary>
Video-conditioned 4D shape generation aims to recover time-varying 3D geometry and view-consistent appearance directly from an input video. In this work, we introduce a native video-to-4D shape generation framework that synthesizes a single dynamic 3D representation end-to-end from the video. Our framework introduces three key components based on large-scale pre-trained 3D models: (i) a temporal attention that conditions generation on all frames while producing a time-indexed dynamic representation; (ii) a time-aware point sampling and 4D latent anchoring that promote temporally consistent geometry and texture; and (iii) noise sharing across frames to enhance temporal stability. Our method accurately captures non-rigid motion, volume changes, and even topological transitions without per-frame optimization. Across diverse in-the-wild videos, our method improves robustness and perceptual fidelity and reduces failure modes compared with the baselines.
</details>

#### 14. SEE4D: Pose-Free 4D Generation via Auto-Regressive Video Inpainting
Dongyue Lu, Ao Liang, Tianxin Huang, Xiao Fu, Yuyang Zhao, Baorui Ma, Liang Pan, Wei Yin, Lingdong Kong, Wei Tsang Ooi, Ziwei Liu

(NUS, HKU, CUHK, Tsinghua, Shanghai AI Laboratory, Horizon Robotics, NTU)

<details span>
<summary><b>Abstract</b></summary>
Immersive applications call for synthesizing spatiotemporal 4D content from casual videos without costly 3D supervision. Existing video-to-4D methods typically rely on manually annotated camera poses, which are labor-intensive and brittle for in-the-wild footage. Recent warp-then-inpaint approaches mitigate the need for pose labels by warping input frames along a novel camera trajectory and using an inpainting model to fill missing regions, thereby depicting the 4D scene from diverse viewpoints. However, this trajectory-to-trajectory formulation often entangles camera motion with scene dynamics and complicates both modeling and inference. We introduce SEE4D, a pose-free, trajectory-to-camera framework that replaces explicit trajectory prediction with rendering to a bank of fixed virtual cameras, thereby separating camera control from scene modeling. A view-conditional video inpainting model is trained to learn a robust geometry prior by denoising realistically synthesized warped images and to inpaint occluded or missing regions across virtual viewpoints, eliminating the need for explicit 3D annotations. Building on this inpainting core, we design a spatiotemporal autoregressive inference pipeline that traverses virtual-camera splines and extends videos with overlapping windows, enabling coherent generation at bounded per-step complexity. We validate See4D on cross-view video generation and sparse reconstruction benchmarks. Across quantitative metrics and qualitative assessments, our method achieves superior generalization and improved performance relative to pose- or trajectory-conditioned baselines, advancing practical 4D world modeling from casual videos.
</details>

#### 15. Object-Aware 4D Human Motion Generation
Shurui Gui, Deep Anil Patel, Xiner Li, Martin Renqiang Min

(Texas A&M University, NEC Laboratories America)

<details span>
<summary><b>Abstract</b></summary>
Recent advances in video diffusion models have enabled the generation of high-quality videos. However, these videos still suffer from unrealistic deformations, semantic violations, and physical inconsistencies that are largely rooted in the absence of 3D physical priors. To address these challenges, we propose an object-aware 4D human motion generation framework grounded in 3D Gaussian representations and motion diffusion priors. With pre-generated 3D humans and objects, our method, Motion Score Distilled Interaction (MSDI), employs the spatial and prompt semantic information in large language models (LLMs) and motion priors through the proposed Motion Diffusion Score Distillation Sampling (MSDS). The combination of MSDS and LLMs enables our spatial-aware motion optimization, which distills score gradients from pre-trained motion diffusion models, to refine human motion while respecting object and semantic constraints. Unlike prior methods requiring joint training on limited interaction datasets, our zero-shot approach avoids retraining and generalizes to out-of-distribution object aware human motions. Experiments demonstrate that our framework produces natural and physically plausible human motions that respect 3D spatial context, offering a scalable solution for realistic 4D generation.
</details>

-----

</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **AR4D: Autoregressive 4D Generation from Monocular Videos**  | 3 Jan 2025 |          [Link](https://arxiv.org/abs/2501.01722)          | --  | [Link](https://hanxinzhu-lab.github.io/AR4D/)  |
| 2025 | **WideRange4D: Enabling High-Quality 4D Reconstruction with Wide-Range Movements and Scenes**  | 17 Mar 2025 |          [Link](https://arxiv.org/abs/2503.13435)          | [Link](https://github.com/Gen-Verse/WideRange4D)  | [Dataset Page](https://huggingface.co/datasets/Gen-Verse/WideRange4D)  |
| 2025 | **TwoSquared: 4D Generation from 2D Image Pairs**  | 17 Apr 2025 |          [Link](https://arxiv.org/abs/2504.12825)          | [Link](https://github.com/Sangluisme/TwoSquared)  | [Link](https://sangluisme.github.io/TwoSquared/)  |
| 2025 | **DeepVerse: 4D Autoregressive Video Generation as a World Model**  | 1 Jun 2025 |          [Link](https://www.arxiv.org/abs/2506.01103)          | [Link](https://github.com/SOTAMak1r/DeepVerse)  | [Link](https://sotamak1r.github.io/deepverse/)  |
| 2025 | **Sonic4D: Spatial Audio Generation for Immersive 4D Scene Exploration**  | 18 Jun 2025 |          [Link](https://arxiv.org/abs/2506.15759)          | [Link](https://github.com/X-Drunker/Sonic4D-project-page)  | [Link](https://x-drunker.github.io/Sonic4D-project-page/)  |
| 2025 | **4Real-Video-V2: Fused View-Time Attention and Feedforward Reconstruction for 4D Scene Generation**  | 18 Jun 2025 |      [Link](https://arxiv.org/abs/2506.18839)     | --  | [Link](https://snap-research.github.io/4Real-Video-V2/)  |
| 2025 | **BulletGen: Improving 4D Reconstruction with Bullet-Time Generation**  | 23 Jun 2025 |      [Link](https://arxiv.org/abs/2506.18601)   | --  | --  |
| 2025 | **4D-LRM: Large Space-Time Reconstruction Model From and To Any View at Any Time**  | 23 Jun 2025 |          [Link](https://arxiv.org/abs/2506.18890)          | [Link](https://github.com/Mars-tin/4D-LRM)  | [Link](https://4dlrm.github.io/)  |
| 2025 | **MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second**  | 14 Jul 2025 |          [Link](https://arxiv.org/abs/2507.10065)          | [Link](https://github.com/chenguolin/MoVieS)  | [Link](https://chenguolin.github.io/projects/MoVieS/)  |
| 2025 | **4DNeX: Feed-Forward 4D Generative Modeling Made Easy**  | 18 Aug 2025 |          [Link](https://arxiv.org/abs/2508.13154)          | [Link](https://github.com/3DTopia/4DNeX)  | [Link](https://4dnex.github.io/)  |
| 2025 | **OmniWorld: A Multi-Domain and Multi-Modal Dataset for 4D World Modeling**  | 15 Sep 2025 |          [Link](https://arxiv.org/abs/2509.12201)          | [Link](https://github.com/yangzhou24/OmniWorld)  | [Link](https://yangzhou24.github.io/OmniWorld/)  |
| 2025 | **Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation**  | 23 Sep 2025 |          [Link](https://arxiv.org/abs/2509.19296)          | [Link](https://github.com/nv-tlabs/lyra)  | [Link](https://research.nvidia.com/labs/toronto-ai/lyra/)  |
| 2025 | **ShapeGen4D: Towards High Quality 4D Shape Generation from Videos**  | 7 Oct 2025 |          [Link](https://arxiv.org/abs/2510.06208)          | -- | [Link](https://shapegen4d.github.io/)  |
| 2025 | **SEE4D: Pose-Free 4D Generation via Auto-Regressive Video Inpainting**  | 30 Oct 2025 |          [Link](https://arxiv.org/abs/2510.26796)          | -- | [Link](https://see-4d.github.io/#4d-gen)  |
| 2025 | **Object-Aware 4D Human Motion Generation**  | 31 Oct 2025 |          [Link](https://arxiv.org/abs/2511.00248)          | -- | -- |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@misc{zhu2025ar4dautoregressive4dgeneration,
      title={AR4D: Autoregressive 4D Generation from Monocular Videos}, 
      author={Hanxin Zhu and Tianyu He and Xiqian Yu and Junliang Guo and Zhibo Chen and Jiang Bian},
      year={2025},
      eprint={2501.01722},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.01722}, 
}

@article{yang2025widerange4d,
  title={WideRange4D: Enabling High-Quality 4D Reconstruction with Wide-Range Movements and Scenes},
  author={Yang, Ling and Zhu, Kaixin and Tian, Juanxi and Zeng, Bohan and Lin, Mingbao and Pei, Hongjuan and Zhang, Wentao and Yan, Shuichen},
  journal={arXiv preprint arXiv:2503.13435},
  year={2025}
}

@article{sang2025twosquared,
  title={TwoSquared: 4D Generation from 2D Image Pairs},
  author={Sang, Lu and Canfes, Zehranaz and Cao, Dongliang and Marin, Riccardo and Bernard, Florian and Cremers, Daniel},
  journal={arXiv preprint arXiv:2504.12825},
  year={2025}
}

@article{zhou2025holotime,
  title={HoloTime: Taming Video Diffusion Models for Panoramic 4D Scene Generation},
  author={Zhou, Haiyang and Yu, Wangbo and Guan, Jiawen and Cheng, Xinhua and Tian, Yonghong and Yuan, Li},
  journal={arXiv preprint arXiv:2504.21650},
  year={2025}
}

@misc{chen2025deepverse4dautoregressivevideo,
      title={DeepVerse: 4D Autoregressive Video Generation as a World Model}, 
      author={Junyi Chen and Haoyi Zhu and Xianglong He and Yifan Wang and Jianjun Zhou and Wenzheng Chang and Yang Zhou and Zizun Li and Zhoujie Fu and Jiangmiao Pang and Tong He},
      year={2025},
      eprint={2506.01103},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.01103}, 
}

@misc{xie2025sonic4dspatialaudiogeneration,
      title={Sonic4D: Spatial Audio Generation for Immersive 4D Scene Exploration}, 
      author={Siyi Xie and Hanxin Zhu and Tianyu He and Xin Li and Zhibo Chen},
      year={2025},
      eprint={2506.15759},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2506.15759}, 
}

@misc{wang20254realvideov2fusedviewtimeattention,
      title={4Real-Video-V2: Fused View-Time Attention and Feedforward Reconstruction for 4D Scene Generation}, 
      author={Chaoyang Wang and Ashkan Mirzaei and Vidit Goel and Willi Menapace and Aliaksandr Siarohin and Avalon Vinella and Michael Vasilkovsky and Ivan Skorokhodov and Vladislav Shakhrai and Sergey Korolev and Sergey Tulyakov and Peter Wonka},
      year={2025},
      eprint={2506.18839},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.18839}, 
}

@misc{rozumnyi2025bulletgenimproving4dreconstruction,
      title={BulletGen: Improving 4D Reconstruction with Bullet-Time Generation}, 
      author={Denys Rozumnyi and Jonathon Luiten and Numair Khan and Johannes Schönberger and Peter Kontschieder},
      year={2025},
      eprint={2506.18601},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2506.18601}, 
}

@article{ma20254dlrm,
  title={4D-LRM: Large Space-Time Reconstruction Model From and To Any View at Any Time}, 
  author={Ziqiao Ma and Xuweiyi Chen and Shoubin Yu and Sai Bi and Kai Zhang and Ziwen Chen and Sihan Xu and Jianing Yang and Zexiang Xu and Kalyan Sunkavalli and Mohit Bansal and Joyce Chai and Hao Tan},
  year={2025},
  journal={arXiv:2506.18890},
}

@misc{lin2025moviesmotionaware4ddynamic,
      title={MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second}, 
      author={Chenguo Lin and Yuchen Lin and Panwang Pan and Yifan Yu and Honglei Yan and Katerina Fragkiadaki and Yadong Mu},
      year={2025},
      eprint={2507.10065},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.10065}, 
}

@article{chen20254dnex,
    title={4DNeX: Feed-Forward 4D Generative Modeling Made Easy},
    author={Chen, Zhaoxi and Liu, Tianqi and Zhuo, Long and Ren, Jiawei and Tao, Zeng and Zhu, He and Hong, Fangzhou and Pan, Liang and Liu, Ziwei},
    journal={arXiv preprint arXiv:2508.13154},
    year={2025}
}

@misc{zhou2025omniworld,
      title={OmniWorld: A Multi-Domain and Multi-Modal Dataset for 4D World Modeling}, 
      author={Yang Zhou and Yifan Wang and Jianjun Zhou and Wenzheng Chang and Haoyu Guo and Zizun Li and Kaijing Ma and Xinyue Li and Yating Wang and Haoyi Zhu and Mingyu Liu and Dingning Liu and Jiange Yang and Zhoujie Fu and Junyi Chen and Chunhua Shen and Jiangmiao Pang and Kaipeng Zhang and Tong He},
      year={2025},
      eprint={2509.12201},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.12201}, 
}

@article{bahmani2025lyra,
  title={Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation},
  author={Bahmani, Sherwin and Shen, Tianchang and Ren, Jiawei and Huang, Jiahui and Jiang, Yifeng and Turki, Haithem and Tagliasacchi, Andrea and Lindell, David B and Gojcic, Zan and Fidler, Sanja and others},
  journal={arXiv preprint arXiv:2509.19296},
  year={2025}
}

@misc{yenphraphai2025shapegen4dhighquality4d,
      title={ShapeGen4D: Towards High Quality 4D Shape Generation from Videos}, 
      author={Jiraphon Yenphraphai and Ashkan Mirzaei and Jianqi Chen and Jiaxu Zou and Sergey Tulyakov and Raymond A. Yeh and Peter Wonka and Chaoyang Wang},
      year={2025},
      eprint={2510.06208},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.06208}, 
}

@article{lu2025see4d,
  title={SEE4D: Pose-Free 4D Generation via Auto-Regressive Video Inpainting},
  author={Lu, Dongyue and Liang, Ao and Huang, Tianxin and Fu, Xiao and Zhao, Yuyang and Ma, Baorui and Pan, Liang and Yin, Wei and Kong, Lingdong and Ooi, Wei Tsang and others},
  journal={arXiv preprint arXiv:2510.26796},
  year={2025}
}

@article{gui2025object,
  title={Object-Aware 4D Human Motion Generation},
  author={Gui, Shurui and Patel, Deep Anil and Li, Xiner and Min, Martin Renqiang},
  journal={arXiv preprint arXiv:2511.00248},
  year={2025}
}

```
</details>

---

### Previous Papers and Other Awesome Repos

#### Year 2023
In 2023, tasks classified as text/Image to 4D and video to 4D generally involve producing four-dimensional data from text/Image or video input. For more details, please check the [2023 4D Papers](./docs/4d/4d_2023.md), including 6 accepted papers and 3 arXiv papers.

#### Year 2024
For more details, please check the [2024 4D Papers](./docs/4d/4d_2024.md), including 24 accepted papers and 10 arXiv papers.

<details close>
<summary>Awesome Repos</summary>

> ##### Survey
* [11 Sep 2025]**3D and 4D World Modeling: A Survey** [[Paper](https://arxiv.org/abs/2509.07996)][[GitHub](https://github.com/worldbench/survey)]
* [22 Oct 2025]**Advances in 4D Representation: Geometry, Motion, and Interaction** [[Paper](https://arxiv.org/abs/2510.19255)][[Project Page](https://mingrui-zhao.github.io/4DRep-GMI/)]

</details>

--------------


## Text to Video

### 🎉 T2V Accepted Papers
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **TransPixar: Advancing Text-to-Video Generation with Transparency**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2501.03006)          | [Link](https://github.com/wileewang/TransPixar)  | [Link](https://wileewang.github.io/TransPixar/)  |
| 2025 | **BlobGEN-Vid: Compositional Text-to-Video Generation with Blob Video Representations**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2501.07647)          | -- | [Link](https://blobgen-vid2.github.io/)  |
| 2025 | **Identity-Preserving Text-to-Video Generation by Frequency Decomposition**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2411.17440)          | [Link](https://github.com/PKU-YuanGroup/ConsisID) | [Link](https://pku-yuangroup.github.io/ConsisID/)  |
| 2025 | **One-Minute Video Generation with Test-Time Training**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2504.05298v1)          | [Link](https://github.com/test-time-training/ttt-video-dit) | [Link](https://test-time-training.github.io/video-dit/)  |
| 2025 | **The Devil is in the Prompts: Retrieval-Augmented Prompt Optimization for Text-to-Video Generation**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2504.11739)          | [Link](https://github.com/Vchitect/RAPO) | [Link](https://whynothaha.github.io/Prompt_optimizer/RAPO.html)  |
| 2025 | **SnapGen-V: Generating a Five-Second Video within Five Seconds on a Mobile Device**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2412.10494)          | -- | [Link](https://snap-research.github.io/snapgen-v/)  |
| 2025 | **Multi-subject Open-set Personalization in Video Generation**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2501.06187)          | [Link](https://github.com/snap-research/MSRVTT-Personalization) | [Link](https://snap-research.github.io/open-set-video-personalization/)  |
| 2025 | **WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions**  | ICCV 2025 |          [Link](https://arxiv.org/abs/2505.18151)          | -- | [Link](https://kyleleey.github.io/WonderPlay/)  |
| 2025 | **Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation**  | Siggraph Asia 2025 |          [Link](https://arxiv.org/abs/2504.14899)          | [Link](https://github.com/ewrfcas/Uni3C) | [Link](https://ewrfcas.github.io/Uni3C/)  |
| 2025 | **Scaling RL to Long Videos**  | NeurIPS 2025 |      [Link](https://arxiv.org/abs/2507.07966)      | [Link](https://github.com/NVlabs/Long-RL) | [YouTube Video](https://www.youtube.com/watch?v=ykbblK2jiEg) |
| 2025 | **Video Killed the Energy Budget: Characterizing the Latency and Power Regimes of Open Text-to-Video Models**  |  NeurIPS 2025 NextVid Workshop |      [Link](https://arxiv.org/abs/2509.19222)      | -- | -- |

<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

@misc{wang2025transpixar,
     title={TransPixar: Advancing Text-to-Video Generation with Transparency}, 
     author={Luozhou Wang and Yijun Li and Zhifei Chen and Jui-Hsien Wang and Zhifei Zhang and He Zhang and Zhe Lin and Yingcong Chen},
     year={2025},
     eprint={2501.03006},
     archivePrefix={arXiv},
     primaryClass={cs.CV},
     url={https://arxiv.org/abs/2501.03006}, 
}

@article{feng2025blobgen,
  title={BlobGEN-Vid: Compositional Text-to-Video Generation with Blob Video Representations},
  author={Feng, Weixi and Liu, Chao and Liu, Sifei and Wang, William Yang and Vahdat, Arash and Nie, Weili},
  journal={arXiv preprint arXiv:2501.07647},
  year={2025}
}

@article{yuan2024identity,
  title={Identity-Preserving Text-to-Video Generation by Frequency Decomposition},
  author={Yuan, Shenghai and Huang, Jinfa and He, Xianyi and Ge, Yunyuan and Shi, Yujun and Chen, Liuhan and Luo, Jiebo and Yuan, Li},
  journal={arXiv preprint arXiv:2411.17440},
  year={2024}
}

@misc{dalal2025oneminutevideogenerationtesttime,
      title={One-Minute Video Generation with Test-Time Training}, 
      author={Karan Dalal and Daniel Koceja and Gashon Hussein and Jiarui Xu and Yue Zhao and Youjin Song and Shihao Han and Ka Chun Cheung and Jan Kautz and Carlos Guestrin and Tatsunori Hashimoto and Sanmi Koyejo and Yejin Choi and Yu Sun and Xiaolong Wang},
      year={2025},
      eprint={2504.05298},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.05298}, 
}

@article{gao2025devil,
  title={The Devil is in the Prompts: Retrieval-Augmented Prompt Optimization for Text-to-Video Generation},
  author={Gao, Bingjie and Gao, Xinyu and Wu, Xiaoxue and Zhou, Yujie and Qiao, Yu and Niu, Li and Chen, Xinyuan and Wang, Yaohui},
  journal={arXiv preprint arXiv:2504.11739},
  year={2025}
}

@inproceedings{wu2025snapgen,
  title={Snapgen-v: Generating a five-second video within five seconds on a mobile device},
  author={Wu, Yushu and Zhang, Zhixing and Li, Yanyu and Xu, Yanwu and Kag, Anil and Sui, Yang and Coskun, Huseyin and Ma, Ke and Lebedev, Aleksei and Hu, Ju and others},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={2479--2490},
  year={2025}
}

@article{chen2025multi,
  title={Multi-subject Open-set Personalization in Video Generation},
  author={Chen, Tsai-Shien and Siarohin, Aliaksandr and Menapace, Willi and Fang, Yuwei and Lee, Kwot Sin and Skorokhodov, Ivan and Aberman, Kfir and Zhu, Jun-Yan and Yang, Ming-Hsuan and Tulyakov, Sergey},
  journal={arXiv preprint arXiv:2501.06187},
  year={2025}
}

@article{li2025wonderplay,
  title={WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions},
  author={Li, Zizhang and Yu, Hong-Xing and Liu, Wei and Yang, Yin and Herrmann, Charles and Wetzstein, Gordon and Wu, Jiajun},
  journal={arXiv preprint arXiv:2505.18151},
  year={2025}
}

@article{cao2025uni3c,
  title={Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation},
  author={Cao, Chenjie and Zhou, Jingkai and Li, Shikai and Liang, Jingyun and Yu, Chaohui and Wang, Fan and Xue, Xiangyang and Fu, Yanwei},
  journal={arXiv preprint arXiv:2504.14899},
  year={2025}
}

@article{chen2025scaling,
  title={Scaling rl to long videos},
  author={Chen, Yukang and Huang, Wei and Shi, Baifeng and Hu, Qinghao and Ye, Hanrong and Zhu, Ligeng and Liu, Zhijian and Molchanov, Pavlo and Kautz, Jan and Qi, Xiaojuan and others},
  journal={arXiv preprint arXiv:2507.07966},
  year={2025}
}

@article{delavande2025video,
  title={Video Killed the Energy Budget: Characterizing the Latency and Power Regimes of Open Text-to-Video Models},
  author={Delavande, Julien and Pierrard, Regis and Luccioni, Sasha},
  journal={arXiv preprint arXiv:2509.19222},
  year={2025}
}

```
</details>

-------

### 💡 T2V ArXiv Papers

#### 1. We'll Fix it in Post: Improving Text-to-Video Generation with Neuro-Symbolic Feedback
Minkyu Choi, S P Sharan, Harsh Goel, Sahil Shah, Sandeep Chinchali

(The University of Texas at Austin)
<details span>
<summary><b>Abstract</b></summary>
Current text-to-video (T2V) generation models are increasingly popular due to their ability to produce coherent videos from textual prompts. However, these models often struggle to generate semantically and temporally consistent videos when dealing with longer, more complex prompts involving multiple objects or sequential events. Additionally, the high computational costs associated with training or fine-tuning make direct improvements impractical. To overcome these limitations, we introduce NeuS-E, a novel zero-training video refinement pipeline that leverages neuro-symbolic feedback to automatically enhance video generation, achieving superior alignment with the prompts. Our approach first derives the neuro-symbolic feedback by analyzing a formal video representation and pinpoints semantically inconsistent events, objects, and their corresponding frames. This feedback then guides targeted edits to the original video. Extensive empirical evaluations on both open-source and proprietary T2V models demonstrate that NeuS-E significantly enhances temporal and logical alignment across diverse prompts by almost 40%.
</details>

#### 2. HunyuanCustom: A Multimodal-Driven Architecture for Customized Video Generation
Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan Zhou, Qin Lin, Qinglin Lu (Tencent Hunyuan)
<details span>
<summary><b>Abstract</b></summary>
Customized video generation aims to produce videos featuring specific subjects under flexible user-defined conditions, yet existing methods often struggle with identity consistency and limited input modalities. In this paper, we propose HunyuanCustom, a multi-modal customized video generation framework that emphasizes subject consistency while supporting image, audio, video, and text conditions. Built upon HunyuanVideo, our model first addresses the image-text conditioned generation task by introducing a text-image fusion module based on LLaVA for enhanced multi-modal understanding, along with an image ID enhancement module that leverages temporal concatenation to reinforce identity features across frames. To enable audio- and video-conditioned generation, we further propose modality-specific condition injection mechanisms: an AudioNet module that achieves hierarchical alignment via spatial cross-attention, and a video-driven injection module that integrates latent-compressed conditional video through a patchify-based feature-alignment network. Extensive experiments on single- and multi-subject scenarios demonstrate that HunyuanCustom significantly outperforms state-of-the-art open- and closed-source methods in terms of ID consistency, realism, and text-video alignment. Moreover, we validate its robustness across downstream tasks, including audio and video-driven customized video generation. Our results highlight the effectiveness of multi-modal conditioning and identity-preserving strategies in advancing controllable video generation.
</details>

#### 3. M4V: Multi-Modal Mamba for Text-to-Video Generation
Jiancheng Huang, Gengwei Zhang, Zequn Jie, Siyu Jiao, Yinlong Qian, Ling Chen, Yunchao Wei, Lin Ma

(Meituan, University of Techcnology Sydney, Beijing Jiaotong University)
<details span>
<summary><b>Abstract</b></summary>
Text-to-video generation has significantly enriched content creation and holds the potential to evolve into powerful world simulators. However, modeling the vast spatiotemporal space remains computationally demanding, particularly when employing Transformers, which incur quadratic complexity in sequence processing and thus limit practical applications. Recent advancements in linear-time sequence modeling, particularly the Mamba architecture, offer a more efficient alternative. Nevertheless, its plain design limits its direct applicability to multi-modal and spatiotemporal video generation tasks. To address these challenges, we introduce M4V, a Multi-Modal Mamba framework for text-to-video generation. Specifically, we propose a multi-modal diffusion Mamba (MM-DiM) block that enables seamless integration of multi-modal information and spatiotemporal modeling through a multi-modal token re-composition design. As a result, the Mamba blocks in M4V reduce FLOPs by 45% compared to the attention-based alternative when generating videos at 7681280 resolution. Additionally, to mitigate the visual quality degradation in long-context autoregressive generation processes, we introduce a reward learning strategy that further enhances per-frame visual realism. Extensive experiments on text-to-video benchmarks demonstrate M4V's ability to produce high-quality videos while significantly lowering computational costs. 
</details>

#### 4. Omni-Video: Democratizing Unified Video Understanding and Generation
Zhiyu Tan, Hao Yang, Luozheng Qin, Jia Gong, Mengping Yang, Hao Li (Fudan University, Shanghai Academy of Artificial Intelligence for Science)

<details span>
<summary><b>Abstract</b></summary>
Notable breakthroughs in unified understanding and generation modeling have led to remarkable advancements in image understanding, reasoning, production and editing, yet current foundational models predominantly focus on processing images, creating a gap in the development of unified models for video understanding and generation. This report presents Omni-Video, an efficient and effective unified framework for video understanding, generation, as well as instruction-based editing. Our key insight is to teach existing multimodal large language models (MLLMs) to produce continuous visual clues that are used as the input of diffusion decoders, which produce high-quality videos conditioned on these visual clues. To fully unlock the potential of our system for unified video modeling, we integrate several technical improvements: 1) a lightweight architectural design that respectively attaches a vision head on the top of MLLMs and a adapter before the input of diffusion decoders, the former produce visual tokens for the latter, which adapts these visual tokens to the conditional space of diffusion decoders; and 2) an efficient multi-stage training scheme that facilitates a fast connection between MLLMs and diffusion decoders with limited data and computational resources. We empirically demonstrate that our model exhibits satisfactory generalization abilities across video generation, editing and understanding tasks.
</details>

#### 5. FIFA: Unified Faithfulness Evaluation Framework for Text-to-Video and Video-to-Text Generation
Liqiang Jing, Viet Lai, Seunghyun Yoon, Trung Bui, Xinya Du (University of Texas at Dallas, Adobe Research)

<details span>
<summary><b>Abstract</b></summary>
Video Multimodal Large Language Models (VideoMLLMs) have achieved remarkable progress in both Video-to-Text and Text-to-Video tasks. However, they often suffer fro hallucinations, generating content that contradicts the visual input. Existing evaluation methods are limited to one task (e.g., V2T) and also fail to assess hallucinations in open-ended, free-form responses. To address this gap, we propose FIFA, a unified FaIthFulness evAluation framework that extracts comprehensive descriptive facts, models their semantic dependencies via a Spatio-Temporal Semantic Dependency Graph, and verifies them using VideoQA models. We further introduce Post-Correction, a tool-based correction framework that revises hallucinated content. Extensive experiments demonstrate that FIFA aligns more closely with human judgment than existing evaluation methods, and that Post-Correction effectively improves factual consistency in both text and video generation.
</details>

#### 6. Lumos-1: On Autoregressive Video Generation from a Unified Model Perspective
Hangjie Yuan, Weihua Chen, Jun Cen, Hu Yu, Jingyun Liang, Shuning Chang, Zhihui Lin, Tao Feng, Pengwei Liu, Jiazheng Xing, Hao Luo, Jiasheng Tang, Fan Wang, Yi Yang

(DAMO Academy Alibaba Group, Hupan Lab, Zhejiang University, Tsinghua University)

<details span>
<summary><b>Abstract</b></summary>
Autoregressive large language models (LLMs) have unified a vast range of language tasks, inspiring preliminary efforts in autoregressive video generation. Existing autoregressive video generators either diverge from standard LLM architectures, depend on bulky external text encoders, or incur prohibitive latency due to next-token decoding. In this paper, we introduce Lumos-1, an autoregressive video generator that retains the LLM architecture with minimal architectural modifications. To inject spatiotemporal correlations in LLMs, we identify the efficacy of incorporating 3D RoPE and diagnose its imbalanced frequency spectrum ranges. Therefore, we propose MM-RoPE, a RoPE scheme that preserves the original textual RoPE while providing comprehensive frequency spectra and scaled 3D positions for modeling multimodal spatiotemporal data. Moreover, Lumos-1 resorts to a token dependency strategy that obeys intra-frame bidirectionality and inter-frame temporal causality. Based on this dependency strategy, we identify the issue of frame-wise loss imbalance caused by spatial information redundancy and solve it by proposing Autoregressive Discrete Diffusion Forcing (AR-DF). AR-DF introduces temporal tube masking during training with a compatible inference-time masking policy to avoid quality degradation. By using memory-efficient training techniques, we pre-train Lumos-1 on only 48 GPUs, achieving performance comparable to EMU3 on GenEval, COSMOS-Video2World on VBench-I2V, and OpenSoraPlan on VBench-T2V.
</details>

#### 7. "PhyWorldBench": A Comprehensive Evaluation of Physical Realism in Text-to-Video Models
Jing Gu, Xian Liu, Yu Zeng, Ashwin Nagarajan, Fangrui Zhu, Daniel Hong, Yue Fan, Qianqi Yan, Kaiwen Zhou, Ming-Yu Liu, Xin Eric Wang

(University of California Santa Cruz, NVIDIA Research, Northeastern University)

<details span>
<summary><b>Abstract</b></summary>
Video generation models have achieved remarkable progress in creating high-quality, photorealistic content. However, their ability to accurately simulate physical phenomena remains a critical and unresolved challenge. This paper presents PhyWorldBench, a comprehensive benchmark designed to evaluate video generation models based on their adherence to the laws of physics. The benchmark covers multiple levels of physical phenomena, ranging from fundamental principles like object motion and energy conservation to more complex scenarios involving rigid body interactions and human or animal motion. Additionally, we introduce a novel ""Anti-Physics"" category, where prompts intentionally violate real-world physics, enabling the assessment of whether models can follow such instructions while maintaining logical consistency. Besides large-scale human evaluation, we also design a simple yet effective method that could utilize current MLLM to evaluate the physics realism in a zero-shot fashion. We evaluate 12 state-of-the-art text-to-video generation models, including five open-source and five proprietary models, with a detailed comparison and analysis. we identify pivotal challenges models face in adhering to real-world physics. Through systematic testing of their outputs across 1,050 curated prompts-spanning fundamental, composite, and anti-physics scenarios-we identify pivotal challenges these models face in adhering to real-world physics. We then rigorously examine their performance on diverse physical phenomena with varying prompt types, deriving targeted recommendations for crafting prompts that enhance fidelity to physical principles.
</details>

#### 8. Can Your Model Separate Yolks with a Water Bottle? Benchmarking Physical Commonsense Understanding in Video Generation Models
Enes Sanli, Baris Sarper Tezcan, Aykut Erdem, Erkut Erdem (Koç University, Hacettepe University)

<details span>
<summary><b>Abstract</b></summary>
Recent progress in text-to-video (T2V) generation has enabled the synthesis of visually compelling and temporally coherent videos from natural language. However, these models often fall short in basic physical commonsense, producing outputs that violate intuitive expectations around causality, object behavior, and tool use. Addressing this gap, we present PhysVidBench, a benchmark designed to evaluate the physical reasoning capabilities of T2V systems. The benchmark includes 383 carefully curated prompts, emphasizing tool use, material properties, and procedural interactions, and domains where physical plausibility is crucial. For each prompt, we generate videos using diverse state-of-the-art models and adopt a three-stage evaluation pipeline: (1) formulate grounded physics questions from the prompt, (2) caption the generated video with a vision-language model, and (3) task a language model to answer several physics-involved questions using only the caption. This indirect strategy circumvents common hallucination issues in direct video-based evaluation. By highlighting affordances and tool-mediated actions, areas overlooked in current T2V evaluations, PhysVidBench provides a structured, interpretable framework for assessing physical commonsense in generative video models.
</details>

#### 9. LongVie: Multimodal-Guided Controllable Ultra-Long Video Generation
Jianxiong Gao, Zhaoxi Chen, Xian Liu, Jianfeng Feng, Chenyang Si, Yanwei Fu, Yu Qiao, Ziwei Liu

(Nanjing University, Fudan University, Nanyang Technological University, Nvidia, Shanghai Artificial Intelligence Laboratory)

<details span>
<summary><b>Abstract</b></summary>
Controllable ultra-long video generation is a fundamental yet challenging task. Although existing methods are effective for short clips, they struggle to scale due to issues such as temporal inconsistency and visual degradation. In this paper, we initially investigate and identify three key factors: separate noise initialization, independent control signal normalization, and the limitations of single-modality guidance. To address these issues, we propose LongVie, an end-to-end autoregressive framework for controllable long video generation. LongVie introduces two core designs to ensure temporal consistency: 1) a unified noise initialization strategy that maintains consistent generation across clips, and 2) global control signal normalization that enforces alignment in the control space throughout the entire video. To mitigate visual degradation, LongVie employs 3) a multi-modal control framework that integrates both dense (e.g., depth maps) and sparse (e.g., keypoints) control signals, complemented by 4) a degradation-aware training strategy that adaptively balances modality contributions over time to preserve visual quality. We also introduce LongVGenBench, a comprehensive benchmark consisting of 100 high-resolution videos spanning diverse real-world and synthetic environments, each lasting over one minute. Extensive experiments show that LongVie achieves state-of-the-art performance in long-range controllability, consistency, and quality.
</details>

#### 10. Yan: Foundational Interactive Video Generation
Yan Team (Tencent)

<details span>
<summary><b>Abstract</b></summary>
We present Yan, a foundational framework for interactive video generation, covering the entire pipeline from simulation and generation to editing. Specifically, Yan comprises three core modules. AAA-level Simulation: We design a highly-compressed, low-latency 3D-VAE coupled with a KV-cache-based shift-window denoising inference process, achieving real-time 1080P/60FPS interactive simulation. Multi-Modal Generation: We introduce a hierarchical autoregressive caption method that injects game-specific knowledge into open-domain multi-modal video diffusion models (VDMs), then transforming the VDM into a frame-wise, action-controllable, real-time infinite interactive video generator. Notably, when the textual and visual prompts are sourced from different domains, the model demonstrates strong generalization, allowing it to blend and compose the style and mechanics across domains flexibly according to user prompts. Multi-Granularity Editing: We propose a hybrid model that explicitly disentangles interactive mechanics simulation from visual rendering, enabling multi-granularity video content editing during interaction through text. Collectively, Yan offers an integration of these modules, pushing interactive video generation beyond isolated capabilities toward a comprehensive AI-driven interactive creation paradigm, paving the way for the next generation of creative tools, media, and entertainment. 
</details>

#### 11. LongLive: Real-time Interactive Long Video Generation
Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, Song Han, Yukang Chen

(Nvidia, MIT, HKUST(GZ), HKU, THU)

<details span>
<summary><b>Abstract</b></summary>
We present LongLive, a frame-level autoregressive (AR) framework for real-time and interactive long video generation. Long video generation presents challenges in both efficiency and quality. Diffusion and Diffusion-Forcing models can produce high-quality videos but suffer from low efficiency due to bidirectional attention. Causal attention AR models support KV caching for faster inference, but often degrade in quality on long videos due to memory challenges during long-video training. In addition, beyond static prompt-based generation, interactive capabilities, such as streaming prompt inputs, are critical for dynamic content creation, enabling users to guide narratives in real time. This interactive requirement significantly increases complexity, especially in ensuring visual consistency and semantic coherence during prompt transitions. To address these challenges, LongLive adopts a causal, frame-level AR design that integrates a KV-recache mechanism that refreshes cached states with new prompts for smooth, adherent switches; streaming long tuning to enable long video training and to align training and inference (train-long-test-long); and short window attention paired with a frame-level attention sink, shorten as frame sink, preserving long-range consistency while enabling faster generation. With these key designs, LongLive fine-tunes a 1.3B-parameter short-clip model to minute-long generation in just 32 GPU-days. At inference, LongLive sustains 20.7 FPS on a single NVIDIA H100, achieves strong performance on VBench in both short and long videos. LongLive supports up to 240-second videos on a single H100 GPU. LongLive further supports INT8-quantized inference with only marginal quality loss.
</details>

#### 12. Character Mixing for Video Generation
Tingting Liao, Chongjian Ge, Guangyi Liu, Hao Li, Yi Zhou (Mohamed bin Zayed University of Artificial Intelligence)

<details span>
<summary><b>Abstract</b></summary>
Imagine Mr. Bean stepping into Tom and Jerry—can we generate videos where characters interact naturally across different worlds? We study inter-character interaction in text-to-video generation, where the key challenge is to preserve each character’s identity and behaviors while enabling coherent cross-context interaction. This is difficult because characters may never have coexisted and
because mixing styles often causes style delusion, where realistic characters appear cartoonish or vice versa. We introduce a framework that tackles these issues with Cross-Character Embedding (CCE), which learns identity and behavioral logic across multimodal sources, and Cross-Character Augmentation (CCA), which enriches training with synthetic co-existence and mixed-style data. Together, these techniques allow natural interactions between previously uncoexistent characters without losing stylistic fidelity. Experiments on a curated benchmark of cartoons and live-action series with 10 characters show clear improvements in identity preservation, interaction quality, and robustness to style delusion, enabling new forms of generative storytelling.
</details>

#### 13. NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos
Hongyu Li, Lingfeng Sun, Yafei Hu, Duy Ta, Jennifer Barry, George Konidaris, Jiahui Fu (Robotics and AI Institute, Brown University)

<details span>
<summary><b>Abstract</b></summary>
Enabling robots to execute novel manipulation tasks zero-shot is a central goal in robotics. Most existing methods assume in-distribution tasks or rely on fine-tuning with embodiment-matched data, limiting transfer across platforms. We present NovaFlow, an autonomous manipulation framework that converts a task description into an actionable plan for a target robot without any demonstrations. Given a task description, NovaFlow synthesizes a video using a video generation model and distills it into 3D actionable object flow using off-the-shelf perception modules. From the object flow, it computes relative poses for rigid objects and realizes them as robot actions via grasp proposals and trajectory optimization. For deformable objects, this flow serves as a tracking objective for model-based planning with a particle-based dynamics model. By decoupling task understanding from low-level control, NovaFlow naturally transfers across embodiments. We validate on rigid, articulated, and deformable object manipulation tasks using a table-top Franka arm and a Spot quadrupedal mobile robot, and achieve effective zero-shot execution without demonstrations or embodiment-specific training.
</details>

#### 14. SeqBench: Benchmarking Sequential Narrative Generation in Text-to-Video Models
Zhengxu Tang, Zizheng Wang, Luning Wang, Zitao Shuai, Chenhao Zhang, Siyu Qian, Yirui Wu, Bohao Wang, Haosong Rao, Zhenyu Yang, Chenwei Wu 

(University of Michigan, Northeastern University, University of Washington, Harvard University, Beijing Jiaotong University, Zhejiang University, University of Rochester)

<details span>
<summary><b>Abstract</b></summary>
Text-to-video (T2V) generation models have made significant progress in creating visually appealing videos. However, they struggle with generating coherent sequential narratives that require logical progression through multiple events. Existing T2V benchmarks primarily focus on visual quality metrics but fail to evaluate narrative coherence over extended sequences. To bridge this gap, we present SeqBench, a comprehensive benchmark for evaluating sequential narrative coherence in T2V generation. SeqBench includes a carefully designed dataset of 320 prompts spanning various narrative complexities, with 2,560 human-annotated videos generated from 8 state-of-the-art T2V models. Additionally, we design a Dynamic Temporal Graphs (DTG)-based automatic evaluation metric, which can efficiently capture long-range dependencies and temporal ordering while maintaining computational efficiency. Our DTG-based metric demonstrates a strong correlation with human annotations. Through systematic evaluation using SeqBench, we reveal critical limitations in current T2V models: failure to maintain consistent object states across multi-action sequences, physically implausible results in multi-object scenarios, and difficulties in preserving realistic timing and ordering relationships between sequential actions. SeqBench provides the first systematic framework for evaluating narrative coherence in T2V generation and offers concrete insights for improving sequential reasoning capabilities in future models.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **We'll Fix it in Post: Improving Text-to-Video Generation with Neuro-Symbolic Feedback**  | 25 Apr 2025 |          [Link](https://arxiv.org/abs/2504.17180)          | -- | --  |
| 2025 | **HunyuanCustom: A Multimodal-Driven Architecture for Customized Video Generation**  | 7 May 2025 |          [Link](https://arxiv.org/abs/2505.04512)          | [Link](https://github.com/Tencent/HunyuanCustom) | [Link](https://hunyuancustom.github.io/)  |
| 2025 | **M4V: Multi-Modal Mamba for Text-to-Video Generation**  | 12 Jun 2025 |          [Link](https://arxiv.org/abs/2506.10915)          | [Link](https://github.com/huangjch526/M4V) | [Link](https://huangjch526.github.io/M4V_project/)  |
| 2025 | **Omni-Video: Democratizing Unified Video Understanding and Generation**  | 9 Jul 2025 |          [Link](https://arxiv.org/abs/2507.06119)          | [Link](https://github.com/SAIS-FUXI/Omni-Video) | [Link](https://howellyoung-s.github.io/OmniVideo_project/)  |
| 2025 | **FIFA: Unified Faithfulness Evaluation Framework for Text-to-Video and Video-to-Text Generation**  | 9 Jul 2025 |          [Link](https://arxiv.org/abs/2507.06523)          | [Link](https://github.com/du-nlp-lab/FIFA) | -- |
| 2025 | **Lumos-1: On Autoregressive Video Generation from a Unified Model Perspective**  | 11 Jul 2025 |      [Link](https://arxiv.org/abs/2507.08801)      | [Link](https://github.com/alibaba-damo-academy/Lumos) | -- |
| 2025 | **"PhyWorldBench": A Comprehensive Evaluation of Physical Realism in Text-to-Video Models**  | 17 Jul 2025 |      [Link](https://www.arxiv.org/pdf/2507.13428)      | -- | -- |
| 2025 | **Can Your Model Separate Yolks with a Water Bottle? Benchmarking Physical Commonsense Understanding in Video Generation Models**  | 21 Jul 2025 |      [Link](https://arxiv.org/abs/2507.15824)      | [Link](https://github.com/ensanli/PhysVidBenchCode) | [Link](https://cyberiada.github.io/PhysVidBench/) |
| 2025 | **LongVie: Multimodal-Guided Controllable Ultra-Long Video Generation**  | 5 Aug 2025 |      [Link](https://arxiv.org/abs/2508.03694)      | [Link](https://github.com/Vchitect/LongVie) | [Link](https://vchitect.github.io/LongVie-project/) |
| 2025 | **Yan: Foundational Interactive Video Generation**  | 12 Aug 2025 |      [Link](https://www.arxiv.org/abs/2508.08601)      | -- | [Link](https://greatx3.github.io/Yan/) |
| 2025 | **LongLive: Real-time Interactive Long Video Generation**  | 26 Sep 2025 |      [Link](https://arxiv.org/abs/2509.22622)      | [Link](https://github.com/NVlabs/LongLive) | [Link](https://nvlabs.github.io/LongLive/) |
| 2025 | **Character Mixing for Video Generation**  | 6 Oct 2025 |      [Link](https://arxiv.org/abs/2510.05093)      | [Link](https://github.com/TingtingLiao/mimix) | [Link](https://tingtingliao.github.io/mimix/) |
| 2025 | **NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos**  | 9 Oct 2025  |      [Link](https://arxiv.org/abs/2510.08568)      | Coming Soon! | [Link](https://novaflow.lhy.xyz/) |
| 2025 | **SeqBench: Benchmarking Sequential Narrative Generation in Text-to-Video Models**  | 14 Oct 2025  |      [Link](https://arxiv.org/abs/2510.13042)      | -- | [Link](https://videobench.github.io/SeqBench.github.io/) |


<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@article{choi2025we,
  title={We'll Fix it in Post: Improving Text-to-Video Generation with Neuro-Symbolic Feedback},
  author={Choi, Minkyu and Sharan, SP and Goel, Harsh and Shah, Sahil and Chinchali, Sandeep},
  journal={arXiv preprint arXiv:2504.17180},
  year={2025}
}

@article{hu2025hunyuancustom,
  title={HunyuanCustom: A Multimodal-Driven Architecture for Customized Video Generation},
  author={Hu, Teng and Yu, Zhentao and Zhou, Zhengguang and Liang, Sen and Zhou, Yuan and Lin, Qin and Lu, Qinglin},
  journal={arXiv preprint arXiv:2505.04512},
  year={2025}
}

@article{huang2025m4v,
  title={M4V: Multi-Modal Mamba for Text-to-Video Generation},
  author={Huang, Jiancheng and Zhang, Gengwei and Jie, Zequn and Jiao, Siyu and Qian, Yinlong and Chen, Ling and Wei, Yunchao and Ma, Lin},
  journal={arXiv preprint arXiv:2506.10915},
  year={2025}
}

@article{tan2025omni,
  title={Omni-Video: Democratizing Unified Video Understanding and Generation},
  author={Tan, Zhiyu and Yang, Hao and Qin, Luozheng and Gong, Jia and Yang, Mengping and Li, Hao},
  journal={arXiv preprint arXiv:2507.06119},
  year={2025}
}

@misc{jing2025fifaunifiedfaithfulnessevaluation,
      title={FIFA: Unified Faithfulness Evaluation Framework for Text-to-Video and Video-to-Text Generation}, 
      author={Liqiang Jing and Viet Lai and Seunghyun Yoon and Trung Bui and Xinya Du},
      year={2025},
      eprint={2507.06523},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.06523}, 
}

@article{Yuan2025Lumos-1,
  title={Lumos-1: On Autoregressive Video Generation from a Unified Model Perspective},
  author={Yuan, Hangjie and Chen, Weihua and Cen, Jun and Yu, Hu and Liang, Jingyun and Chang, Shuning and Lin, Zhihui and Feng, Tao and Liu, Pengwei and Xing, Jiazheng and Luo, Hao and Tang, Jiasheng and Wang, Fan and Yang, Yi},
  journal={arXiv preprint arXiv:2507.08801},
  year={2025}
}

@article{gu2025phyworldbench,
  title={"PhyWorldBench": A Comprehensive Evaluation of Physical Realism in Text-to-Video Models},
  author={Gu, Jing and Liu, Xian and Zeng, Yu and Nagarajan, Ashwin and Zhu, Fangrui and Hong, Daniel and Fan, Yue and Yan, Qianqi and Zhou, Kaiwen and Liu, Ming-Yu and others},
  journal={arXiv preprint arXiv:2507.13428},
  year={2025}
}

@misc{sanli2025modelseparateyolkswater,
      title={Can Your Model Separate Yolks with a Water Bottle? Benchmarking Physical Commonsense Understanding in Video Generation Models}, 
      author={Enes Sanli and Baris Sarper Tezcan and Aykut Erdem and Erkut Erdem},
      year={2025},
      eprint={2507.15824},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.15824}, 
}

@misc{gao2025longviemultimodalguidedcontrollableultralong,
      title={LongVie: Multimodal-Guided Controllable Ultra-Long Video Generation}, 
      author={Jianxiong Gao and Zhaoxi Chen and Xian Liu and Jianfeng Feng and Chenyang Si and Yanwei Fu and Yu Qiao and Ziwei Liu},
      year={2025},
      eprint={2508.03694},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.03694}, 
}

@article{yan,
  title   = {Yan: Foundational Interactive Video Generation},
  author  = {Yan Team},
  url     = {https://greatx3.github.io/Yan/},
  year    = {2025}
}

@article{yang2025longlive,
    title={LongLive: Real-time Interactive Long Video Generation},
    author={Shuai Yang and Wei Huang and Ruihang Chu and Yicheng Xiao and Yuyang Zhao and Xianbang Wang and Muyang Li and Enze Xie and Yingcong Chen and Yao Lu and Song Hanand Yukang Chen},
    year={2025},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@article{mimix2025,
  title   = {Character Mixing for Video Generation},
  author  = {Tingting Liao, Chongjian Ge, Guangyi Liu, Hao Li, Yi Zhou},
  year    = {2025}
  eprint  = {2510.05093}, 
  note    = {arXiv preprint}
}

@article{li2025novaflow,
  title={NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos},
  author={Li, Hongyu and Sun, Lingfeng and Hu, Yafei and Ta, Duy and Barry, Jennifer and Konidaris, George and Fu, Jiahui},
  journal={arXiv preprint arXiv:2510.08568},
  year={2025}
}

@article{tang2025seqbench,
  title={SeqBench: Benchmarking Sequential Narrative Generation in Text-to-Video Models},
  author={Tang, Zhengxu and Wang, Zizheng and Wang, Luning and Shuai, Zitao and Zhang, Chenhao and Qian, Siyu and Wu, Yirui and Wang, Bohao and Rao, Haosong and Yang, Zhenyu and others},
  journal={arXiv preprint arXiv:2510.13042},
  year={2025}
}
```
</details>


---

### Video Other Additional Info

### Previous Papers

#### Year 2024
For more details, please check the [2024 T2V Papers](./docs/video/t2v_2024.md), including 21 accepted papers and 6 arXiv papers.

- OSS video generation models: [Mochi 1](https://github.com/genmoai/models) preview is an open state-of-the-art video generation model with high-fidelity motion and strong prompt adherence.
- Survey: The Dawn of Video Generation: Preliminary Explorations with SORA-like Models, [arXiv](https://arxiv.org/abs/2410.05227), [Project Page](https://ailab-cvc.github.io/VideoGen-Eval/), [GitHub Repo](https://github.com/AILab-CVC/VideoGen-Eval)

### 📚 Dataset Works

#### 1. VidGen-1M: A Large-Scale Dataset for Text-to-video Generation
Zhiyu Tan, Xiaomeng Yang, Luozheng Qin, Hao Li

(Fudan University, ShangHai Academy of AI for Science)
<details span>
<summary><b>Abstract</b></summary>
The quality of video-text pairs fundamentally determines the upper bound of text-to-video models. Currently, the datasets used for training these models suffer from significant shortcomings, including low temporal consistency, poor-quality captions, substandard video quality, and imbalanced data distribution. The prevailing video curation process, which depends on image models for tagging and manual rule-based curation, leads to a high computational load and leaves behind unclean data. As a result, there is a lack of appropriate training datasets for text-to-video models. To address this problem, we present VidGen-1M, a superior training dataset for text-to-video models. Produced through a coarse-to-fine curation strategy, this dataset guarantees high-quality videos and detailed captions with excellent temporal consistency. When used to train the video generation model, this dataset has led to experimental results that surpass those obtained with other models.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2024 | **VidGen-1M: A Large-Scale Dataset for Text-to-video Generation**  | 5 Aug 2024  |          [Link](https://arxiv.org/abs/2408.02629)          | [Link](https://github.com/SAIS-FUXI/VidGen) | [Link](https://sais-fuxi.github.io/projects/vidgen-1m/)  |

<details close>
<summary>References</summary>

```
%axiv papers

@article{tan2024vidgen,
  title={VidGen-1M: A Large-Scale Dataset for Text-to-video Generation},
  author={Tan, Zhiyu and Yang, Xiaomeng, and Qin, Luozheng and Li Hao},
  booktitle={arXiv preprint arxiv:2408.02629},
  year={2024}
}


```
</details>

--------------

## Text to Scene

### 🎉 3D Scene Accepted Papers
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **Scene Splatter: Momentum 3D Scene Generation from Single Image with Video Diffusion Model**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2504.02764)          | [Link](https://github.com/shengjun-zhang/Scene-Splatter)  | [Link](https://shengjun-zhang.github.io/SceneSplatter/)  |
| 2025 | **ScenePainter: Semantically Consistent Perpetual 3D Scene Generation with Concept Relation Alignment**  | ICCV 2025 |          [Link](https://arxiv.org/abs/2507.19058)          | [Link](https://github.com/xiac20/ScenePainter)  | [Link](https://xiac20.github.io/ScenePainter/)  |
| 2025 | **Bolt3D: Generating 3D Scenes in Seconds**  | ICCV 2025 |          [Link](https://arxiv.org/abs/2503.14445)          | --  | [Link](https://szymanowiczs.github.io/bolt3d)  |
| 2025 | **Generating Physically Stable and Buildable Brick Structures from Text**  | ICCV 2025 Best Paper |          [Link](https://arxiv.org/abs/2505.05469)          | [Link](https://github.com/AvaLovelace1/BrickGPT/)  | [Link](https://avalovelace1.github.io/BrickGPT/)  |
| 2025 | **WorldExplorer: Towards Generating Fully Navigable 3D Scenes**  | SIGGRAPH Asia 2025 |          [Link](https://arxiv.org/abs/2506.01799)          | [Link](https://github.com/mschneider456/worldexplorer)  | [Link](https://mschneider456.github.io/world-explorer/)  |
| 2025 | **VideoFrom3D: 3D Scene Video Generation via Complementary Image and Video Diffusion Models**  | SIGGRAPH Asia 2025 |          [Link](https://arxiv.org/abs/2509.17985)          | [Link](https://github.com/KIMGEONUNG/VideoFrom3D)  | [Link](https://kimgeonung.github.io/VideoFrom3D/)  |
| 2025 | **SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent**  | NeurIPS 2025 |          [Link](https://arxiv.org/abs/2509.20414)          | [Link](https://github.com/Scene-Weaver/SceneWeaver)  | [Link](https://scene-weaver.github.io/)  |

<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

@article{Scene Splatter,
        title   = {Scene Splatter: Momentum 3D Scene Generation from Single Image with Video Diffusion Model},
        author  = {Zhang, Shengjun and Li, Jinzhao and Fei, Xin and Liu, Hao and Duan, Yueqi},
        journal = {IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)},
        year    = {2025},
}

@article{xia2025scenepainter,
  title={ScenePainter: Semantically Consistent Perpetual 3D Scene Generation with Concept Relation Alignment},
  author={Xia, Chong and Zhang, Shengjun and Liu, Fangfu and Liu, Chang and Hirunyaratsameewong, Khodchaphun and Duan, Yueqi},
  journal={arXiv preprint arXiv:2507.19058},
  year={2025}
}

@article{szymanowicz2025bolt3d,
title={Bolt3D: Generating 3D Scenes in Seconds},
author={Szymanowicz, Stanislaw and Zhang, Jason Y. and Srinivasan, Pratul
     and Gao, Ruiqi and Brussee, Arthur and Holynski, Aleksander and
     Martin-Brualla, Ricardo and Barron, Jonathan T. and Henzler, Philipp},
journal={International Conference on Computer Vision},
year={2025}
}

@inproceedings{pun2025brickgpt,
    title     = {Generating Physically Stable and Buildable Brick Structures from Text},
    author    = {Pun, Ava and Deng, Kangle and Liu, Ruixuan and Ramanan, Deva and Liu, Changliu and Zhu, Jun-Yan},
    booktitle = {ICCV},
    year      = {2025}
}

@article{schneider2025worldexplorer,
  title={WorldExplorer: Towards Generating Fully Navigable 3D Scenes},
  author={Schneider, Manuel-Andreas and H{\"o}llein, Lukas and Nie{\ss}ner, Matthias},
  journal={arXiv preprint arXiv:2506.01799},
  year={2025}
}

@article{kim2025videofrom3d,
  title={VideoFrom3D: 3D Scene Video Generation via Complementary Image and Video Diffusion Models},
  author={Kim, Geonung and Han, Janghyeok and Cho, Sunghyun},
  journal={arXiv preprint arXiv:2509.17985},
  year={2025}
}

@article{yang2025sceneweaver,
  title={SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent},
  author={Yang, Yandan and Jia, Baoxiong and Zhang, Shujie and Huang, Siyuan},
  journal={arXiv preprint arXiv:2509.20414},
  year={2025}
}

```
</details>

-------

### 💡 3D Scene ArXiv Papers

#### 1. LAYOUTDREAMER: Physics-guided Layout for Text-to-3D Compositional Scene Generation
Yang Zhou, Zongjin He, Qixuan Li, Chao Wang (ShangHai University)
<details span>
<summary><b>Abstract</b></summary>
Recently, the field of text-guided 3D scene generation has garnered significant attention. High-quality generation that aligns with physical realism and high controllability is crucial for practical 3D scene applications. However, existing methods face fundamental limitations: (i) difficulty capturing complex relationships between multiple objects described in the text, (ii) inability to generate physically plausible scene layouts, and (iii) lack of controllability and extensibility in compositional scenes. In this paper, we introduce LayoutDreamer, a framework that leverages 3D Gaussian Splatting (3DGS) to facilitate high-quality, physically consistent compositional scene generation guided by text. Specifically, given a text prompt, we convert it into a directed scene graph and adaptively adjust the density and layout of the initial compositional 3D Gaussians. Subsequently, dynamic camera adjustments are made based on the training focal point to ensure entity-level generation quality. Finally, by extracting directed dependencies from the scene graph, we tailor physical and layout energy to ensure both realism and flexibility. Comprehensive experiments demonstrate that LayoutDreamer outperforms other compositional scene generation quality and semantic alignment methods. Specifically, it achieves state-of-the-art (SOTA) performance in the multiple objects generation metric of T3Bench.
</details>

#### 2. WORLDMEM: Long-term Consistent World Simulation with Memory
Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, Xingang Pan

(Nanyang Technological University, Peking University, Shanghai AI Laboratry)
<details span>
<summary><b>Abstract</b></summary>
World simulation has gained increasing popularity due to its ability to model virtual environments and predict the consequences of actions. However, the limited temporal context window often leads to failures in maintaining long-term consistency, particularly in preserving 3D spatial consistency. In this work, we present WorldMem, a framework that enhances scene generation with a memory bank consisting of memory units that store memory frames and states (e.g., poses and timestamps). By employing a memory attention mechanism that effectively extracts relevant information from these memory frames based on their states, our method is capable of accurately reconstructing previously observed scenes, even under significant viewpoint or temporal gaps. Furthermore, by incorporating timestamps into the states, our framework not only models a static world but also captures its dynamic evolution over time, enabling both perception and interaction within the simulated world. Extensive experiments in both virtual and real scenarios validate the effectiveness of our approach.
</details>

#### 3. HiScene: Creating Hierarchical 3D Scenes with Isometric View Generation
Wenqi Dong, Bangbang Yang, Zesong Yang, Yuan Li, Tao Hu, Hujun Bao, Yuewen Ma, Zhaopeng Cui

(Zhejiang University, ByteDance)
<details span>
<summary><b>Abstract</b></summary>
Scene-level 3D generation represents a critical frontier in multimedia and computer graphics, yet existing approaches either suffer from limited object categories or lack editing flexibility for interactive applications. In this paper, we present HiScene, a novel hierarchical framework that bridges the gap between 2D image generation and 3D object generation and delivers high-fidelity scenes with compositional identities and aesthetic scene content. Our key insight is treating scenes as hierarchical "objects" under isometric views, where a room functions as a complex object that can be further decomposed into manipulatable items. This hierarchical approach enables us to generate 3D content that aligns with 2D representations while maintaining compositional structure. To ensure completeness and spatial alignment of each decomposed instance, we develop a video-diffusion-based amodal completion technique that effectively handles occlusions and shadows between objects, and introduce shape prior injection to ensure spatial coherence within the scene. Experimental results demonstrate that our method produces more natural object arrangements and complete object instances suitable for interactive applications, while maintaining physical plausibility and alignment with user inputs.
</details>

#### 4. 3DTown: Constructing a 3D Town from a Single Image
Kaizhi Zheng, Ruijian Zhang, Jing Gu, Jie Yang, Xin Eric Wang

(University of California Santa Cruz, Columbia University, Cybever AI)
<details span>
<summary><b>Abstract</b></summary>
Acquiring detailed 3D scenes typically demands costly equipment, multi-view data, or labor-intensive modeling. Therefore, a lightweight alternative, generating complex 3D scenes from a single top-down image, plays an essential role in real-world applications. While recent 3D generative models have achieved remarkable results at the object level, their extension to full-scene generation often leads to inconsistent geometry, layout hallucinations, and low-quality meshes. In this work, we introduce 3DTown, a training-free framework designed to synthesize realistic and coherent 3D scenes from a single top-down view. Our method is grounded in two principles: region-based generation to improve image-to-3D alignment and resolution, and spatial-aware 3D inpainting to ensure global scene coherence and high-quality geometry generation. Specifically, we decompose the input image into overlapping regions and generate each using a pretrained 3D object generator, followed by a masked rectified flow inpainting process that fills in missing geometry while maintaining structural continuity. This modular design allows us to overcome resolution bottlenecks and preserve spatial structure without requiring 3D supervision or fine-tuning. Extensive experiments across diverse scenes show that 3DTown outperforms state-of-the-art baselines, including Trellis, Hunyuan3D-2, and TripoSG, in terms of geometry quality, spatial coherence, and texture fidelity. Our results demonstrate that high-quality 3D town generation is achievable from a single image using a principled, training-free approach.
</details>

#### 5. Agentic 3D Scene Generation with Spatially Contextualized VLMs
Xinhang Liu, Yu-Wing Tai, Chi-Keung Tang (HKUST, Dartmouth College)

<details span>
<summary><b>Abstract</b></summary>
Despite recent advances in multimodal content generation enabled by vision-language models (VLMs), their ability to reason about and generate structured 3D scenes remains largely underexplored. This limitation constrains their utility in spatially grounded tasks such as embodied AI, immersive simulations, and interactive 3D applications. We introduce a new paradigm that enables VLMs to generate, understand, and edit complex 3D environments by injecting a continually evolving spatial context. Constructed from multimodal input, this context consists of three components: a scene portrait that provides a high-level semantic blueprint, a semantically labeled point cloud capturing object-level geometry, and a scene hypergraph that encodes rich spatial relationships, including unary, binary, and higher-order constraints. Together, these components provide the VLM with a structured, geometry-aware working memory that integrates its inherent multimodal reasoning capabilities with structured 3D understanding for effective spatial reasoning. Building on this foundation, we develop an agentic 3D scene generation pipeline in which the VLM iteratively reads from and updates the spatial context. The pipeline features high-quality asset generation with geometric restoration, environment setup with automatic verification, and ergonomic adjustment guided by the scene hypergraph. Experiments show that our framework can handle diverse and challenging inputs, achieving a level of generalization not observed in prior work. Further results demonstrate that injecting spatial context enables VLMs to perform downstream tasks such as interactive scene editing and path planning, suggesting strong potential for spatially intelligent systems in computer graphics, 3D vision, and embodied applications.
</details>

#### 6. Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation
Tianyu Huang, Wangguandong Zheng, Tengfei Wang, Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li, Rynson W.H. Lau, Wangmeng Zuo, Chunchao Guo

(Harbin Institute of Technology, Southeast University, Tencent Hunyuan, City University of Hong Kong)

<details span>
<summary><b>Abstract</b></summary>
Real-world applications like video gaming and virtual reality often demand the ability to model 3D scenes that users can explore along custom camera trajectories. While significant progress has been made in generating 3D objects from text or images, creating long-range, 3D-consistent, explorable 3D scenes remains a complex and challenging problem. In this work, we present Voyager, a novel video diffusion framework that generates world-consistent 3D point-cloud sequences from a single image with user-defined camera path. Unlike existing approaches, Voyager achieves end-to-end scene generation and reconstruction with inherent consistency across frames, eliminating the need for 3D reconstruction pipelines (e.g., structure-from-motion or multi-view stereo). Our method integrates three key components: 1) World-Consistent Video Diffusion: A unified architecture that jointly generates aligned RGB and depth video sequences, conditioned on existing world observation to ensure global coherence 2) Long-Range World Exploration: An efficient world cache with point culling and an auto-regressive inference with smooth video sampling for iterative scene extension with context-aware consistency, and 3) Scalable Data Engine: A video reconstruction pipeline that automates camera pose estimation and metric depth prediction for arbitrary videos, enabling large-scale, diverse training data curation without manual 3D annotations. Collectively, these designs result in a clear improvement over existing methods in visual quality and geometric accuracy, with versatile applications.
</details>

#### 7. ReSpace: Text-Driven 3D Scene Synthesis and Editing with Preference Alignment
Martin JJ. Bucher, Iro Armeni (Stanford University)

<details span>
<summary><b>Abstract</b></summary>
Scene synthesis and editing has emerged as a promising direction in computer graphics. Current trained approaches for 3D indoor scenes either oversimplify object semantics through one-hot class encodings (e.g., 'chair' or 'table'), require masked diffusion for editing, ignore room boundaries, or rely on floor plan renderings that fail to capture complex layouts. In contrast, LLM-based methods enable richer semantics via natural language (e.g., 'modern studio with light wood furniture') but do not support editing, remain limited to rectangular layouts or rely on weak spatial reasoning from implicit world models. We introduce ReSpace, a generative framework for text-driven 3D indoor scene synthesis and editing using autoregressive language models. Our approach features a compact structured scene representation with explicit room boundaries that frames scene editing as a next-token prediction task. We leverage a dual-stage training approach combining supervised fine-tuning and preference alignment, enabling a specially trained language model for object addition that accounts for user instructions, spatial geometry, object semantics, and scene-level composition. For scene editing, we employ a zero-shot LLM to handle object removal and prompts for addition. We further introduce a novel voxelization-based evaluation that captures fine-grained geometry beyond 3D bounding boxes. Experimental results surpass state-of-the-art on object addition while maintaining competitive results on full scene synthesis.
</details>

#### 8. EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence
Xinjie Wang, Liu Liu, Yu Cao, Ruiqi Wu, Wenkang Qin, Dehui Wang, Wei Sui, Zhizhong Su

(Horizon Robotics, GigaAI, D-Robotics, Shanghai Jiao Tong University, Nankai University)

<details span>
<summary><b>Abstract</b></summary>
Constructing a physically realistic and accurately scaled simulated 3D world is crucial for the training and evaluation of embodied intelligence tasks. The diversity, realism, low cost accessibility and affordability of 3D data assets are critical for achieving generalization and scalability in embodied AI. However, most current embodied intelligence tasks still rely heavily on traditional 3D computer graphics assets manually created and annotated, which suffer from high production costs and limited realism. These limitations significantly hinder the scalability of data driven approaches. We present EmbodiedGen, a foundational platform for interactive 3D world generation. It enables the scalable generation of high-quality, controllable and photorealistic 3D assets with accurate physical properties and real-world scale in the Unified Robotics Description Format (URDF) at low cost. These assets can be directly imported into various physics simulation engines for fine-grained physical control, supporting downstream tasks in training and evaluation. EmbodiedGen is an easy-to-use, full-featured toolkit composed of six key modules: Image-to-3D, Text-to-3D, Texture Generation, Articulated Object Generation, Scene Generation and Layout Generation. EmbodiedGen generates diverse and interactive 3D worlds composed of generative 3D assets, leveraging generative AI to address the challenges of generalization and evaluation to the needs of embodied intelligence related research. 
</details>

#### 9. ImmerseGen: Agent-Guided Immersive World Generation with Alpha-Textured Proxies
Jinyan Yuan, Bangbang Yang, Keke Wang, Panwang Pan, Lin Ma, Xuehai Zhang, Xiao Liu, Zhaopeng Cui, Yuewen Ma

(PICO Bytedance, State Key Laboratory of CAD&CG Zhejiang University)

<details span>
<summary><b>Abstract</b></summary>
Automatic creation of 3D scenes for immersive VR presence has been a significant research focus for decades. However, existing methods often rely on either high-poly mesh modeling with post-hoc simplification or massive 3D Gaussians, resulting in a complex pipeline or limited visual realism. In this paper, we demonstrate that such exhaustive modeling is unnecessary for achieving compelling immersive experience. We introduce ImmerseGen, a novel agent-guided framework for compact and photorealistic world modeling. ImmerseGen represents scenes as hierarchical compositions of lightweight geometric proxies, i.e., simplified terrain and billboard meshes, and generates photorealistic appearance by synthesizing RGBA textures onto these proxies. Specifically, we propose terrain-conditioned texturing for user-centric base world synthesis, and RGBA asset texturing for midground and foreground scenery. This reformulation offers several advantages: (i) it simplifies modeling by enabling agents to guide generative models in producing coherent textures that integrate seamlessly with the scene; (ii) it bypasses complex geometry creation and decimation by directly synthesizing photorealistic textures on proxies, preserving visual quality without degradation; (iii) it enables compact representations suitable for real-time rendering on mobile VR headsets. To automate scene creation from text prompts, we introduce VLM-based modeling agents enhanced with semantic grid-based analysis for improved spatial reasoning and accurate asset placement. ImmerseGen further enriches scenes with dynamic effects and ambient audio to support multisensory immersion. Experiments on scene generation and live VR showcases demonstrate that ImmerseGen achieves superior photorealism, spatial coherence and rendering efficiency compared to prior methods.
</details>

#### 10. DreamAnywhere: Object-Centric Panoramic 3D Scene Generation
Edoardo Alberto Dominici, Jozef Hladky, Floor Verhoeven, Lukas Radl, Thomas Deixelberger, Stefan Ainetter, Philipp Drescher, Stefan Hauswiesner, Arno Coomans, Giacomo Nazzaro, Konstantinos Vardis, Markus Steinberger

(Huawei Technologies, Graz University of Technology)

<details span>
<summary><b>Abstract</b></summary>
Recent advances in text-to-3D scene generation have demonstrated significant potential to transform content creation across multiple industries. Although the research community has made impressive progress in addressing the challenges of this complex task, existing methods often generate environments that are only front-facing, lack visual fidelity, exhibit limited scene understanding, and are typically fine-tuned for either indoor or outdoor settings. In this work, we address these issues and propose DreamAnywhere, a modular system for the fast generation and prototyping of 3D scenes. Our system synthesizes a 360° panoramic image from text, decomposes it into background and objects, constructs a complete 3D representation through hybrid inpainting, and lifts object masks to detailed 3D objects that are placed in the virtual environment. DreamAnywhere supports immersive navigation and intuitive object-level editing, making it ideal for scene exploration, visual mock-ups, and rapid prototyping -- all with minimal manual modeling. These features make our system particularly suitable for low-budget movie production, enabling quick iteration on scene layout and visual tone without the overhead of traditional 3D workflows. Our modular pipeline is highly customizable as it allows components to be replaced independently. Compared to current state-of-the-art text and image-based 3D scene generation approaches, DreamAnywhere shows significant improvements in coherence in novel view synthesis and achieves competitive image quality, demonstrating its effectiveness across diverse and challenging scenarios. A comprehensive user study demonstrates a clear preference for our method over existing approaches, validating both its technical robustness and practical usefulness.
</details>

#### 11. Towards Geometric and Textural Consistency 3D Scene Generation via Single Image-guided Model Generation and Layout Optimization
Xiang Tang, Ruotong Li, Xiaopeng Fan

(Harbin Institute of Technology Shenzhen, Peng Cheng Laboratory, Harbin Institute of Technology, Harbin Institute of Technology Suzhou Research Institute)

<details span>
<summary><b>Abstract</b></summary>
In recent years, 3D generation has made great strides in both academia and industry. However, generating 3D scenes from a single RGB image remains a significant challenge, as current approaches often struggle to ensure both object generation quality and scene coherence in multi-object scenarios. To overcome these limitations, we propose a novel three-stage framework for 3D scene generation with explicit geometric representations and high-quality textural details via single image-guided model generation and spatial layout optimization. Our method begins with an image instance segmentation and inpainting phase, which recovers missing details of occluded objects in the input images, thereby achieving complete generation of foreground 3D assets. Subsequently, our approach captures the spatial geometry of reference image by constructing pseudo-stereo viewpoint for camera parameter estimation and scene depth inference, while employing a model selection strategy to ensure optimal alignment between the 3D assets generated in the previous step and the input. Finally, through model parameterization and minimization of the Chamfer distance between point clouds in 3D and 2D space, our approach optimizes layout parameters to produce an explicit 3D scene representation that maintains precise alignment with input guidance image. Extensive experiments on multi-object scene image sets have demonstrated that our approach not only outperforms state-of-the-art methods in terms of geometric accuracy and texture fidelity of individual generated 3D models, but also has significant advantages in scene layout synthesis.
</details>

#### 12. HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels
Tencent Hunyuan

<details span>
<summary><b>Abstract</b></summary>
Creating immersive and playable 3D worlds from texts or images remains a fundamental challenge in computer vision and graphics. Existing world generation approaches typically fall into two categories: video-based methods that offer rich diversity but lack 3D consistency and rendering efficiency, and 3D-based methods that provide geometric consistency but struggle with limited training data and memory-inefficient representations. To address these limitations, we present HunyuanWorld 1.0, a novel framework that combines the best of both worlds for generating immersive, explorable, and interactive 3D scenes from text and image conditions. Our approach features three key advantages: 1) 360° immersive experiences via panoramic world proxies; 2) mesh export capabilities for seamless compatibility with existing computer graphics pipelines; 3) disentangled object representations for augmented interactivity. The core of our framework is a semantically layered 3D mesh representation that leverages panoramic images as 360° world proxies for semantic-aware world decomposition and reconstruction, enabling the generation of diverse 3D worlds. Extensive experiments demonstrate that our method achieves state-of-the-art performance in generating coherent, explorable, and interactive 3D worlds while enabling versatile applications in virtual reality, physical simulation, game development, and interactive content creation.
</details>

#### 13. Matrix-3D: Omnidirectional Explorable 3D World Generation
Zhongqi Yang, Wenhang Ge, Yuqi Li, Jiaqi Chen, Haoyuan Li, Mengyin An, Fei Kang, Hua Xue, Baixin Xu, Yuyang Yin, Eric Li, Yang Liu, Yikai Wang, Hao-Xiang Guo, Yahui Zhou

(Skywork AI, Hong Kong University of Science and Technology (Guangzhou), Institute of Computing Technology Chinese Academy of Sciences, School of Artificial Intelligence Beijing Normal University)

<details span>
<summary><b>Abstract</b></summary>
Explorable 3D world generation from a single image or text prompt forms a cornerstone of spatial intelligence. Recent works utilize video model to achieve wide-scope and generalizable 3D world generation. However, existing approaches often suffer from limited reconstruction scope and suboptimal visual quality. In this work, we propose Matrix-3D, a framework that utilize panoramic representation for wide-coverage omnidirectional explorable 3D world generation that combines conditional video generation and panoramic 3D reconstruction. We first train a trajectory-guided panoramic video diffusion model that employs scene mesh renders as condition, to enable high-quality and geometrically consistent scene video generation. To enable 3D world generation, we introduce two methods that lift the 2D content to 3D world, ensuring efficiency and effectiveness. To lift the panorama scene video to 3D world, we propose two separate pipelines — a feed-forward large reconstruction model for rapid 3D scene reconstruction and an optimization-based pipeline for accurate and detailed 3D scene reconstruction. For efficiency, we introduce a feed-forward panoramic 3D reconstruction model that projects video latents and camera poses to predict omni-directional 3D Gaussian Splatting attributes. To facilitate convergence, we adopt a two-stage training strategy and supervise the model using rendered panoramic novel views. For effectiveness, we also propose a optimization-based reconstruction method. However, no existing panoramic video dataset provides associated camera poses. To facilitate effective training, we also introduce the Matrix-Pano dataset — the first large-scale synthetic collection comprising 116,759 high-quality static panoramic video sequences with various annotations. Extensive experiments demonstrate the effectiveness of our proposed framework, which achieves state-of-the-art performance in panoramic video generation and 3D world generation.
</details>

#### 14. SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass
Yanxu Meng, Haoning Wu, Ya Zhang, Weidi Xie (Shanghai Jiao Tong University)

<details span>
<summary><b>Abstract</b></summary>
3D content generation has recently attracted significant research interest due to its applications in VR/AR and embodied AI. In this work, we address the challenging task of synthesizing multiple 3D assets within a single scene image. Concretely, our contributions are fourfold: (i) we present SceneGen, a novel framework that takes a scene image and corresponding object masks as input, simultaneously producing multiple 3D assets with geometry and texture. Notably, SceneGen operates with no need for optimization or asset retrieval; (ii) we introduce a novel feature aggregation module that integrates local and global scene information from visual and geometric encoders within the feature extraction module. Coupled with a position head, this enables the generation of 3D assets and their relative spatial positions in a single feedforward pass; (iii) we demonstrate SceneGen's direct extensibility to multi-image input scenarios. Despite being trained solely on single-image inputs, our architectural design enables improved generation performance with multi-image inputs; and (iv) extensive quantitative and qualitative evaluations confirm the efficiency and robust generation abilities of our approach. We believe this paradigm offers a novel solution for high-quality 3D content generation, potentially advancing its practical applications in downstream tasks.
</details>

#### 15. FlashWorld: High-quality 3D Scene Generation within Seconds
Xinyang Li, Tengfei Wang, Zixiao Gu, Shengchuan Zhang, Chunchao Guo, Liujuan Cao 

(MAC Lab Xiamen University, Tencent, Yes Lab Fudan University)

<details span>
<summary><b>Abstract</b></summary>
We propose FlashWorld, a generative model that produces 3D scenes from a single image or text prompt in seconds, 10~100 faster than previous works while possessing superior rendering quality. Our approach shifts from the conventional multi-view-oriented (MV-oriented) paradigm, which generates multi-view images for subsequent 3D reconstruction, to a 3D-oriented approach where the model directly produces 3D Gaussian representations during multi-view generation. While ensuring 3D consistency, 3D-oriented method typically suffers poor visual quality. FlashWorld includes a dual-mode pre-training phase followed by a cross-mode post-training phase, effectively integrating the strengths of both paradigms. Specifically, leveraging the prior from a video diffusion model, we first pre-train a dual-mode multi-view diffusion model, which jointly supports MV-oriented and 3D-oriented generation modes. To bridge the quality gap in 3D-oriented generation, we further propose a cross-mode post-training distillation by matching distribution from consistent 3D-oriented mode to high-quality MV-oriented mode. This not only enhances visual quality while maintaining 3D consistency, but also reduces the required denoising steps for inference. Also, we propose a strategy to leverage massive single-view images and text prompts during this process to enhance the model's generalization to out-of-distribution inputs. Extensive experiments demonstrate the superiority and efficiency of our method.
</details>

#### 16. WorldGrow: Generating Infinite 3D World
Sikuang Li, Chen Yang, Jiemin Fang, Taoran Yi, Jia Lu, Jiazhong Cen, Lingxi Xie, Wei Shen, Qi Tian

(SJTU, Huawei Inc., Huazhong University of Science and Technology)

<details span>
<summary><b>Abstract</b></summary>
We tackle the challenge of generating the infinitely extendable 3D world -- large, continuous environments with coherent geometry and realistic appearance. Existing methods face key challenges: 2D-lifting approaches suffer from geometric and appearance inconsistencies across views, 3D implicit representations are hard to scale up, and current 3D foundation models are mostly object-centric, limiting their applicability to scene-level generation. Our key insight is leveraging strong generation priors from pre-trained 3D models for structured scene block generation. To this end, we propose WorldGrow, a hierarchical framework for unbounded 3D scene synthesis. Our method features three core components: (1) a data curation pipeline that extracts high-quality scene blocks for training, making the 3D structured latent representations suitable for scene generation; (2) a 3D block inpainting mechanism that enables context-aware scene extension; and (3) a coarse-to-fine generation strategy that ensures both global layout plausibility and local geometric/textural fidelity. Evaluated on the large-scale 3D-FRONT dataset, WorldGrow achieves SOTA performance in geometry reconstruction, while uniquely supporting infinite scene generation with photorealistic and structurally consistent outputs. These results highlight its capability for constructing large-scale virtual environments and potential for building future world models.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **LAYOUTDREAMER: Physics-guided Layout for Text-to-3D Compositional Scene Generation**  | 4 Feb 2025 |          [Link](https://arxiv.org/abs/2502.01949)          | --  | --  |
| 2025 | **WORLDMEM: Long-term Consistent World Simulation with Memory**  | 16 Apr 2025 |          [Link](https://arxiv.org/abs/2504.12369)          | [Link](https://github.com/xizaoqu/WorldMem)  | [Link](https://xizaoqu.github.io/worldmem/)  |
| 2025 | **HiScene: Creating Hierarchical 3D Scenes with Isometric View Generation**  | 17 Apr 2025 |          [Link](https://arxiv.org/abs/2504.13072)          | --  | [Link](https://zju3dv.github.io/hiscene/)  |
| 2025 | **3DTown: Constructing a 3D Town from a Single Image**  | 21 May 2025 |          [Link](https://arxiv.org/abs/2505.15765)          | --  | [Link](https://eric-ai-lab.github.io/3dtown.github.io/)  |
| 2025 | **Agentic 3D Scene Generation with Spatially Contextualized VLMs**  | 26 May 2025 |          [Link](https://arxiv.org/abs/2505.20129)          | --  | [Link](https://spatctxvlm.github.io/project_page/)  |
| 2025 | **Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation**  | 4 Jun 2025 |          [Link](https://arxiv.org/abs/2506.04225)          | [Link](https://github.com/Voyager-World/Voyager)  | [Link](https://voyager-world.github.io/)  |
| 2025 | **ReSpace: Text-Driven 3D Scene Synthesis and Editing with Preference Alignment**  | 10 Jun 2025 |          [Link](https://arxiv.org/pdf/2506.02459)          | [Link](https://github.com/GradientSpaces/respace)  | [Link](https://respace.mnbucher.com/)  |
| 2025 | **EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence**  | 16 Jun 2025 |          [Link](https://arxiv.org/abs/2506.10600)          | [Link](https://github.com/HorizonRobotics/EmbodiedGen)  | [Link](https://horizonrobotics.github.io/robot_lab/embodied_gen/index.html)  |
| 2025 | **ImmerseGen: Agent-Guided Immersive World Generation with Alpha-Textured Proxies**  | 18 Jun 2025 |          [Link](https://www.arxiv.org/abs/2506.14315)          | Coming Soon! | [Link](https://immersegen.github.io/)  |
| 2025 | **DreamAnywhere: Object-Centric Panoramic 3D Scene Generation**  | 25 Jun 2025 |          [Link](https://arxiv.org/abs/2506.20367)          | -- | --  |
| 2025 | **Towards Geometric and Textural Consistency 3D Scene Generation via Single Image-guided Model Generation and Layout Optimization**  | 20 Jul 2025 |          [Link](https://arxiv.org/abs/2507.14841)          | [Link](https://github.com/xdlbw/sing3d) | [Link](https://xdlbw.github.io/sing3d/)  |
| 2025 | **HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels**  | 26 Jul 2025 |          [Technical Report](https://3d-models.hunyuan.tencent.com/world/HY_World_1_technical_report.pdf)          | [Link](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) | [Link](https://3d-models.hunyuan.tencent.com/world/)  |
| 2025 | **Matrix-3D: Omnidirectional Explorable 3D World Generation**  | 12 Aug 2025 |          [Technical Report](https://github.com/SkyworkAI/Matrix-3D/blob/main/asset/report.pdf)          | [Link](https://github.com/SkyworkAI/Matrix-3D) | [Link](https://matrix-3d.github.io/)  |
| 2025 | **SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass**  | 21 Aug 2025 |          [Link](https://arxiv.org/abs/2508.15769)          | [Link](https://github.com/Mengmouxu/SceneGen) | [Link](https://mengmouxu.github.io/SceneGen/)  |
| 2025 | **FlashWorld: High-quality 3D Scene Generation within Seconds**  | 15 Oct 2025 |          [Link](https://arxiv.org/abs/2510.13678)          | [Link](https://github.com/imlixinyang/FlashWorld) | [Link](https://imlixinyang.github.io/FlashWorld-Project-Page/)  |
| 2025 | **WorldGrow: Generating Infinite 3D World**  | 24 Oct 2025 |          [Link](https://arxiv.org/abs/2510.21682)          | [Link](https://github.com/world-grow/WorldGrow) | [Link](https://world-grow.github.io/)  |


<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@article{zhou2025layoutdreamer,
  title={LAYOUTDREAMER: Physics-guided Layout for Text-to-3D Compositional Scene Generation},
  author={Zhou, Yang and He, Zongjin and Li, Qixuan and Wang, Chao},
  journal={arXiv preprint arXiv:2502.01949},
  year={2025}
}

@misc{xiao2025worldmemlongtermconsistentworld,
      title={WORLDMEM: Long-term Consistent World Simulation with Memory}, 
      author={Zeqi Xiao and Yushi Lan and Yifan Zhou and Wenqi Ouyang and Shuai Yang and Yanhong Zeng and Xingang Pan},
      year={2025},
      eprint={2504.12369},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.12369}, 
}

@article{dong2025hiscene,
      title   = {HiScene: Creating Hierarchical 3D Scenes with Isometric View Generation},
      author  = {Dong, Wenqi and Yang, Bangbang and Yang, Zesong and Li, Yuan and Hu, Tao and Bao, Hujun and Ma, Yuewen and Cui, Zhaopeng},
      journal = {arXiv preprint arXiv:2504.13072},
      year    = {2025},
}

@misc{zheng2025constructing3dtownsingle,
      title={Constructing a 3D Town from a Single Image}, 
      author={Kaizhi Zheng and Ruijian Zhang and Jing Gu and Jie Yang and Xin Eric Wang},
      year={2025},
      eprint={2505.15765},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.15765}, 
}

@misc{liu2025agentic3dscenegeneration,
      title={Agentic 3D Scene Generation with Spatially Contextualized VLMs}, 
      author={Xinhang Liu and Yu-Wing Tai and Chi-Keung Tang},
      year={2025},
      eprint={2505.20129},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.20129}, 
}

@article{huang2025voyager,
  title={Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation},
  author={Huang, Tianyu and Zheng, Wangguandong and Wang, Tengfei and Liu, Yuhao and Wang, Zhenwei and Wu, Junta and Jiang, Jie and Li, Hui and Lau, Rynson WH and Zuo, Wangmeng and others},
  journal={arXiv preprint arXiv:2506.04225},
  year={2025}
}

@article{bucher2025respace,
  title={ReSpace: Text-Driven 3D Scene Synthesis and Editing with Preference Alignment},
  author={Bucher, Martin JJ and Armeni, Iro},
  journal={arXiv preprint arXiv:2506.02459},
  year={2025}
}

@misc{wang2025embodiedgengenerative3dworld,
      title={EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence},
      author={Xinjie Wang and Liu Liu and Yu Cao and Ruiqi Wu and Wenkang Qin and Dehui Wang and Wei Sui and Zhizhong Su},
      year={2025},
      eprint={2506.10600},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.10600},
}

@article{yuan2025immersegen,
  title={ImmerseGen: Agent-Guided Immersive World Generation with Alpha-Textured Proxies},
  author={Yuan, Jinyan and Yang, Bangbang and Wang, Keke and Pan, Panwang and Ma, Lin and Zhang, Xuehai and Liu, Xiao and Cui, Zhaopeng and Ma, Yuewen},
  journal={arXiv preprint arXiv:2506.14315},
  year={2025}
}

@article{dominici2025dreamanywhere,
  title={DreamAnywhere: Object-Centric Panoramic 3D Scene Generation},
  author={Dominici, Edoardo Alberto and Hladky, Jozef and Verhoeven, Floor and Radl, Lukas and Deixelberger, Thomas and Ainetter, Stefan and Drescher, Philipp and Hauswiesner, Stefan and Coomans, Arno and Nazzaro, Giacomo and others},
  journal={arXiv preprint arXiv:2506.20367},
  year={2025}
}

@article{tang2025geometrictexturalconsistency3d,
  title={Towards Geometric and Textural Consistency 3D Scene Generation via Single Image-guided Model Generation and Layout Optimization},
  author={Tang, Xiang and Li, Ruotong and Fan, Xiaopeng},
  journal={arXiv preprint arXiv:2507.14841},
  year={2025}
}

@misc{hunyuanworld2025tencent,
    title={HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels},
    author={Tencent Hunyuan3D Team},
    year={2025},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@article{yang2025matrix3d,
  title     = {Matrix-3D: Omnidirectional Explorable 3D World Generation},
  author    = {Zhongqi Yang and Wenhang Ge and Yuqi Li and Jiaqi Chen and Haoyuan Li and Mengyin An and Fei Kang and Hua Xue and Baixin Xu and Yuyang Yin and Eric Li and Yang Liu and Yikai Wang and Hao-Xiang Guo and Yahui Zhou},
  year      = {2025}
}

@article{meng2025scenegen,
  author    = {Meng, Yanxu and Wu, Haoning and Zhang, Ya and Xie, Weidi},
  title     = {SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass},
  journal   = {arXiv preprint arXiv:2508.15769},
  year      = {2025},
}

@misc{li2025flashworldhighquality3dscene,
      title={FlashWorld: High-quality 3D Scene Generation within Seconds}, 
      author={Xinyang Li and Tengfei Wang and Zixiao Gu and Shengchuan Zhang and Chunchao Guo and Liujuan Cao},
      year={2025},
      eprint={2510.13678},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.13678}, 
}

@article{worldgrow2025,
  title   = {WorldGrow: Generating Infinite 3D World},
  author  = {Li, Sikuang and Yang, Chen and Fang, Jiemin and Yi, Taoran and Lu, Jia and Cen, Jiazhong and Xie, Lingxi and Shen, Wei and Tian, Qi},
  journal = {arXiv preprint arXiv:2510.21682},
  year    = {2025}
}

```
</details>

### Scene Other Additional Info

### Previous Papers

#### Year 2023-2024
For more details, please check the [2023-2024 3D Scene Papers](./docs/3d_scene/3d_scene_23-24.md), including 23 accepted papers and 8 arXiv papers.

<details close>
<summary>Awesome Repos</summary>

> ##### Survey
* [arXiv 16 Apr 2025]**Recent Advance in 3D Object and Scene Generation: A Survey** [[Paper](https://arxiv.org/abs/2504.11734)]
* [arXiv 8 May 2025]**3D Scene Generation: A Survey** [[Paper](https://arxiv.org/abs/2505.05474)][[GitHub](https://github.com/hzxie/Awesome-3D-Scene-Generation)]

> ##### Awesome Repos
- Resource1: [WorldGen: Generate Any 3D Scene in Seconds](https://github.com/ZiYang-xie/WorldGen)
- Resource2: RTFM: A Real-Time Frame Model [Blog](https://www.worldlabs.ai/blog/rtfm) [Demo Try-on](https://rtfm.worldlabs.ai/)

</details>

--------------


## Text to Human Motion

### 🎉 Motion Accepted Papers
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **MixerMDM: Learnable Composition of Human Motion Diffusion Models**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2504.01019)          | [Link](https://github.com/pabloruizponce/MixerMDM)  | [Link](https://www.pabloruizponce.com/papers/MixerMDM)  |
| 2025 | **Dynamic Motion Blending for Versatile Motion Editing**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2503.20724)          | [Link](https://github.com/emptybulebox1/motionRefit/)  | [Link](https://awfuact.github.io/motionrefit/)  |
| 2025 | **MoLA: Motion Generation and Editing with Latent Diffusion Enhanced by Adversarial Training**  | CVPR 2025 HuMoGen Workshop |          [Link](https://arxiv.org/abs/2406.01867)          | -- |  --  |
| 2025 | **MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space**  | ICCV 2025 |          [Link](https://arxiv.org/abs/2503.15451)          | [Link](https://github.com/zju3dv/MotionStreamer)  | [Link](https://zju3dv.github.io/MotionStreamer/)  |
| 2025 | **Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data**  | ICCV 2025 |          [Link](https://arxiv.org/abs/2507.07095)          | [Link](https://github.com/VankouF/MotionMillion-Codes)  | [Link](https://vankouf.github.io/MotionMillion/)  |
| 2025 | **KinMo: Kinematic-aware Human Motion Understanding and Generation**  | ICCV 2025 |          [Link](https://arxiv.org/abs/2411.15472)          | -- | [Link](https://andypinxinliu.github.io/KinMo/)  |
| 2025 | **MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm**  | ICCV 2025 |          [Link](https://arxiv.org/abs/2502.02358)          | [Link](https://github.com/Diouo/MotionLab)  | [Link](https://diouo.github.io/motionlab.github.io/)  |
| 2025 | **GENMO: A GENeralist Model for Human MOtion**  | ICCV 2025 (Highlight) |          [Link](https://arxiv.org/abs/2505.01425)          | [Link](https://github.com/NVlabs/GENMO) | [Link](https://research.nvidia.com/labs/dair/genmo/)  |
| 2025 | **ControlMM: Controllable Masked Motion Generation**  |  ICCV 2025 (Oral)  |          [Link](https://arxiv.org/abs/2410.10780)          | [Link](https://github.com/exitudio/ControlMM/) | [Link](https://exitudio.github.io/ControlMM-page/)  |
| 2025 | **SnapMoGen: Human Motion Generation from Expressive Texts**  | NeurIPS 2025 |          [Link](https://www.arxiv.org/abs/2507.09122)          | [Link](https://github.com/snap-research/SnapMoGen)  | [Link](https://snap-research.github.io/SnapMoGen/) |

<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

@article{ruiz2025mixermdm,
  title={MixerMDM: Learnable Composition of Human Motion Diffusion Models},
  author={Ruiz-Ponce, Pablo and Barquero, German and Palmero, Cristina and Escalera, Sergio and Garc{\'\i}a-Rodr{\'\i}guez, Jos{\'e}},
  journal={arXiv preprint arXiv:2504.01019},
  year={2025}
}

@article{jiang2025dynamic,
  title={Dynamic Motion Blending for Versatile Motion Editing},
  author={Jiang, Nan and Li, Hongjie and Yuan, Ziye and He, Zimo and Chen, Yixin and Liu, Tengyu and Zhu, Yixin and Huang, Siyuan},
  journal={arXiv preprint arXiv:2503.20724},
  year={2025}
}

@inproceedings{uchida2025mola,
  title={Mola: Motion generation and editing with latent diffusion enhanced by adversarial training},
  author={Uchida, Kengo and Shibuya, Takashi and Takida, Yuhta and Murata, Naoki and Tanke, Julian and Takahashi, Shusuke and Mitsufuji, Yuki},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={2910--2919},
  year={2025}
}

@article{xiao2025motionstreamer,
      title={MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space},
      author={Xiao, Lixing and Lu, Shunlin and Pi, Huaijin and Fan, Ke and Pan, Liang and Zhou, Yueer and Feng, Ziyong and Zhou, Xiaowei and Peng, Sida and Wang, Jingbo},
      journal={arXiv preprint arXiv:2503.15451},
      year={2025}
}

@misc{fan2025zerozeroshotmotiongeneration,
      title={Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data}, 
      author={Ke Fan and Shunlin Lu and Minyue Dai and Runyi Yu and Lixing Xiao and Zhiyang Dou and Junting Dong and Lizhuang Ma and Jingbo Wang},
      year={2025},
      eprint={2507.07095},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.07095}, 
}

@inproceedings{kinmo2025kinematicawarehumanmotion,
      title={{KinMo: Kinematic-aware Human Motion Understanding and Generation}},
      author={Pengfei Zhang and Pinxin Liu and Pablo Garrido and Hyeongwoo Kim and Bindita Chaudhuri},
      booktitle={IEEE/CVF International Conference on Computer Vision},
      year={2025},
}

@article{guo2025motionlab,
  title={Motionlab: Unified human motion generation and editing via the motion-condition-motion paradigm},
  author={Guo, Ziyan and Hu, Zeyu and Soh, De Wen and Zhao, Na},
  journal={arXiv preprint arXiv:2502.02358},
  year={2025}
}

@article{li2025genmo,
  title={GENMO: A GENeralist Model for Human MOtion},
  author={Li, Jiefeng and Cao, Jinkun and Zhang, Haotian and Rempe, Davis and Kautz, Jan and Iqbal, Umar and Yuan, Ye},
  journal={arXiv preprint arXiv:2505.01425},
  year={2025}
}

@misc{pinyoanuntapong2025maskcontrolspatiotemporalcontrolmasked,
      title={MaskControl: Spatio-Temporal Control for Masked Motion Synthesis}, 
      author={Ekkasit Pinyoanuntapong and Muhammad Usama Saleem and Korrawe Karunratanakul and Pu Wang and Hongfei Xue and Chen Chen and Chuan Guo and Junli Cao and Jian Ren and Sergey Tulyakov},
      year={2025},
      eprint={2410.10780},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.10780}, 
}

@article{guo2025snapmogen,
  title={SnapMoGen: Human Motion Generation from Expressive Texts},
  author={Guo, Chuan and Hwang, Inwoo and Wang, Jian and Zhou, Bing},
  journal={arXiv preprint arXiv:2507.09122},
  year={2025}
}

```
</details>

-------

### 💡 Motion ArXiv Papers

#### 1. Motion Anything: Any to Motion Generation
Zeyu Zhang, Yiran Wang, Wei Mao, Danning Li, Rui Zhao, Biao Wu, Zirui Song, Bohan Zhuang, Ian Reid, Richard Hartley

(The Australian National University, The University of Sydney, Tecent Canberra XR Vision Labs, McGill University, JD.com, University of Technology Sydney, Mohamed bin Zayed University of Artificial Intelligence, Zhejiang University, Google Research)
<details span>
<summary><b>Abstract</b></summary>
Conditional motion generation has been extensively studied in computer vision, yet two critical challenges remain. First, while masked autoregressive methods have recently outperformed diffusion-based approaches, existing masking models lack a mechanism to prioritize dynamic frames and body parts based on given conditions. Second, existing methods for different conditioning modalities often fail to integrate multiple modalities effectively, limiting control and coherence in generated motion. To address these challenges, we propose Motion Anything, a multimodal motion generation framework that introduces an Attention-based Mask Modeling approach, enabling fine-grained spatial and temporal control over key frames and actions. Our model adaptively encodes multimodal conditions, including text and music, improving controllability. Additionally, we introduce Text-Music-Dance (TMD), a new motion dataset consisting of 2,153 pairs of text, music, and dance, making it twice the size of AIST++, thereby filling a critical gap in the community. Extensive experiments demonstrate that Motion Anything surpasses state-of-the-art methods across multiple benchmarks, achieving a 15% improvement in FID on HumanML3D and showing consistent performance gains on AIST++ and TMD. 
</details>

#### 2. Animating the Uncaptured: Humanoid Mesh Animation with Video Diffusion Models
Marc Benedí San Millán, Angela Dai, Matthias Nießner

(Technical University of Munich)
<details span>
<summary><b>Abstract</b></summary>
Animation of humanoid characters is essential in various graphics applications, but requires significant time and cost to create realistic animations. We propose an approach to synthesize 4D animated sequences of input static 3D humanoid meshes, leveraging strong generalized motion priors from generative video models -- as such video models contain powerful motion information covering a wide variety of human motions. From an input static 3D humanoid mesh and a text prompt describing the desired animation, we synthesize a corresponding video conditioned on a rendered image of the 3D mesh. We then employ an underlying SMPL representation to animate the corresponding 3D mesh according to the video-generated motion, based on our motion optimization. This enables a cost-effective and accessible solution to enable the synthesis of diverse and realistic 4D animations.
</details>

#### 3. FlowMotion: Target-Predictive Conditional Flow Matching for Jitter-Reduced Text-Driven Human Motion Generation
Manolo Canales Cuba, Vinícius do Carmo Melício, João Paulo Gois

(Universidade Federal do ABC, Santo Andr ́e, Brazil)
<details span>
<summary><b>Abstract</b></summary>
Achieving high-fidelity and temporally smooth 3D human motion generation remains a challenge, particularly within resource-constrained environments. We introduce FlowMotion, a novel method leveraging Conditional Flow Matching (CFM). FlowMotion incorporates a training objective within CFM that focuses on more accurately predicting target motion in 3D human motion generation, resulting in enhanced generation fidelity and temporal smoothness while maintaining the fast synthesis times characteristic of flow-matching-based methods. FlowMotion achieves state-of-the-art jitter performance, achieving the best jitter in the KIT dataset and the second-best jitter in the HumanML3D dataset, and a competitive FID value in both datasets. This combination provides robust and natural motion sequences, offering a promising equilibrium between generation quality and temporal naturalness.
</details>

#### 4. ReAlign: Bilingual Text-to-Motion Generation via Step-Aware Reward-Guided Alignment
Wanjiang Weng, Xiaofeng Tan, Hongsong Wang, Pan Zhou

(Southeast University, Key Laboratory of New Generation Artificial Intelligence Technology and Its Interdisciplinary Applications, Singapore Management University)
<details span>
<summary><b>Abstract</b></summary>
Bilingual text-to-motion generation, which synthesizes 3D human motions from bilingual text inputs, holds immense potential for cross-linguistic applications in gaming, film, and robotics. However, this task faces critical challenges: the absence of bilingual motion-language datasets and the misalignment between text and motion distributions in diffusion models, leading to semantically inconsistent or low-quality motions. To address these challenges, we propose BiHumanML3D, a novel bilingual human motion dataset, which establishes a crucial benchmark for bilingual text-to-motion generation models. Furthermore, we propose a Bilingual Motion Diffusion model (BiMD), which leverages cross-lingual aligned representations to capture semantics, thereby achieving a unified bilingual model. Building upon this, we propose Reward-guided sampling Alignment (ReAlign) method, comprising a step-aware reward model to assess alignment quality during sampling and a reward-guided strategy that directs the diffusion process toward an optimally aligned distribution. This reward model integrates step-aware tokens and combines a text-aligned module for semantic consistency and a motion-aligned module for realism, refining noisy motions at each timestep to balance probability density and alignment. Experiments demonstrate that our approach significantly improves text-motion alignment and motion quality compared to existing state-of-the-art methods.
</details>

#### 5. UniHM: Universal Human Motion Generation with Object Interactions in Indoor Scenes
Zichen Geng, Zeeshan Hayder, Wei Liu, Ajmal Mian (University of Western Australia, Data61 CSIRO Australia)
<details span>
<summary><b>Abstract</b></summary>
Human motion synthesis in complex scenes presents a fundamental challenge, extending beyond conventional Text-to-Motion tasks by requiring the integration of diverse modalities such as static environments, movable objects, natural language prompts, and spatial waypoints. Existing language-conditioned motion models often struggle with scene-aware motion generation due to limitations in motion tokenization, which leads to information loss and fails to capture the continuous, context-dependent nature of 3D human movement. To address these issues, we propose UniHM, a unified motion language model that leverages diffusion-based generation for synthesizing scene-aware human motion. UniHM is the first framework to support both Text-to-Motion and Text-to-Human-Object Interaction (HOI) in complex 3D scenes. Our approach introduces three key contributions: (1) a mixed-motion representation that fuses continuous 6DoF motion with discrete local motion tokens to improve motion realism; (2) a novel Look-Up-Free Quantization VAE (LFQ-VAE) that surpasses traditional VQ-VAEs in both reconstruction accuracy and generative performance; and (3) an enriched version of the Lingo dataset augmented with HumanML3D annotations, providing stronger supervision for scene-specific motion learning. Experimental results demonstrate that UniHM achieves comparative performance on the OMOMO benchmark for text-to-HOI synthesis and yields competitive results on HumanML3D for general text-conditioned motion generation.
</details>

#### 6. ReMoMask: Retrieval-Augmented Masked Motion Generation
Zhengdao Li, Siheng Wang, Zeyu Zhang, Hao Tang (Peking University, Jiangsu University)
<details span>
<summary><b>Abstract</b></summary>
Text-to-Motion (T2M) generation aims to synthesize realistic and semantically aligned human motion sequences from natural language descriptions. However, current approaches face dual challenges: Generative models (e.g., diffusion models) suffer from limited diversity, error accumulation, and physical implausibility, while Retrieval-Augmented Generation (RAG) methods exhibit diffusion inertia, partial-mode collapse, and asynchronous artifacts. To address these limitations, we propose ReMoMask, a unified framework integrating three key innovations: 1) A Bidirectional Momentum Text-Motion Model decouples negative sample scale from batch size via momentum queues, substantially improving cross-modal retrieval precision; 2) A Semantic Spatio-temporal Attention mechanism enforces biomechanical constraints during part-level fusion to eliminate asynchronous artifacts; 3) RAG-Classier-Free Guidance incorporates minor unconditional generation to enhance generalization. Built upon MoMask's RVQ-VAE, ReMoMask efficiently generates temporally coherent motions in minimal steps. Extensive experiments on standard benchmarks demonstrate the state-of-the-art performance of ReMoMask, achieving a 3.88% and 10.97% improvement in FID scores on HumanML3D and KIT-ML, respectively, compared to the previous SOTA method RAG-T2M. 
</details>

#### 7. X-MoGen: Unified Motion Generation across Humans and Animals
Xuan Wang, Kai Ruan, Liyang Qian, Zhizhi Guo, Chang Su, Gaoang Wang

(Zhejiang University, Institute of Artificial Intelligence (TeleAI) China Telecom, Renmin University of China)

<details span>
<summary><b>Abstract</b></summary>
Text-driven motion generation has attracted increasing attention due to its broad applications in virtual reality, animation, and robotics. While existing methods typically model human and animal motion separately, a joint cross-species approach offers key advantages, such as a unified representation and improved generalization. However, morphological differences across species remain a key challenge, often compromising motion plausibility. To address this, we propose \textbf{X-MoGen}, the first unified framework for cross-species text-driven motion generation covering both humans and animals. X-MoGen adopts a two-stage architecture. First, a conditional graph variational autoencoder learns canonical T-pose priors, while an autoencoder encodes motion into a shared latent space regularized by morphological loss. In the second stage, we perform masked motion modeling to generate motion embeddings conditioned on textual descriptions. During training, a morphological consistency module is employed to promote skeletal plausibility across species. To support unified modeling, we construct \textbf{UniMo4D}, a large-scale dataset of 115 species and 119k motion sequences, which integrates human and animal motions under a shared skeletal topology for joint training. Extensive experiments on UniMo4D demonstrate that X-MoGen outperforms state-of-the-art methods on both seen and unseen species.
</details>

#### 8. EgoTwin: Dreaming Body and View in First Person
Jingqiao Xiu, Fangzhou Hong, Yicong Li, Mengze Li, Wentao Wang, Sirui Han, Liang Pan, Ziwei Liu

(National University of Singapore, Nanyang Technological University, Hong Kong University of Science and Technology, Shanghai AI Laboratory)

<details span>
<summary><b>Abstract</b></summary>
While exocentric video synthesis has achieved great progress, egocentric video generation remains largely underexplored, which requires modeling first-person view content along with camera motion patterns induced by the wearer's body movements. To bridge this gap, we introduce a novel task of joint egocentric video and human motion generation, characterized by two key challenges: 1) Viewpoint Alignment: the camera trajectory in the generated video must accurately align with the head trajectory derived from human motion; 2) Causal Interplay: the synthesized human motion must causally align with the observed visual dynamics across adjacent video frames. To address these challenges, we propose EgoTwin, a joint video-motion generation framework built on the diffusion transformer architecture. Specifically, EgoTwin introduces a head-centric motion representation that anchors the human motion to the head joint and incorporates a cybernetics-inspired interaction mechanism that explicitly captures the causal interplay between video and motion within attention operations. For comprehensive evaluation, we curate a large-scale real-world dataset of synchronized text-video-motion triplets and design novel metrics to assess video-motion consistency. Extensive experiments demonstrate the effectiveness of the EgoTwin framework.
</details>

#### 9. Pulp Motion: Framing-aware multimodal camera and human motion generation
Robin Courant, Xi Wang, David Loiseaux, Marc Christie, Vicky Kalogeiton

(LIX, Ecole Polytechnique, IP Paris; Inria Saclay; Inria, IRISA, CNRS, Univ. Rennes)

<details span>
<summary><b>Abstract</b></summary>
Treating human motion and camera trajectory generation separately overlooks a core principle of cinematography: the tight interplay between actor performance and camera work in the screen space. In this paper, we are the first to cast this task as a text-conditioned joint generation, aiming to maintain consistent on-screen framing while producing two heterogeneous, yet intrinsically linked, modalities: human motion and camera trajectories. We propose a simple, model-agnostic framework that enforces multimodal coherence via an auxiliary modality: the on-screen framing induced by projecting human joints onto the camera. This on-screen framing provides a natural and effective bridge between modalities, promoting consistency and leading to more precise joint distribution. We first design a joint autoencoder that learns a shared latent space, together with a lightweight linear transform from the human and camera latents to a framing latent. We then introduce auxiliary sampling, which exploits this linear transform to steer generation toward a coherent framing modality. To support this task, we also introduce the PulpMotion dataset, a human-motion and camera-trajectory dataset with rich captions, and high-quality human motions. Extensive experiments across DiT- and MAR-based architectures show the generality and effectiveness of our method in generating on-frame coherent human-camera motions, while also achieving gains on textual alignment for both modalities. Our qualitative results yield more cinematographically meaningful framings setting the new state of the art for this task. 
</details>

#### 10. Text2Interact: High-Fidelity and Diverse Text-to-Two-Person Interaction Generation
Qingxuan Wu, Zhiyang Dou, Chuan Guo, Yiming Huang, Qiao Feng, Bing Zhou, Jian Wang, Lingjie Liu

(University of Pennsylvania, The University of Hong Kong, Snap Inc.)

<details span>
<summary><b>Abstract</b></summary>
Modeling human-human interactions from text remains challenging because it requires not only realistic individual dynamics but also precise, text-consistent spatiotemporal coupling between agents. Currently, progress is hindered by 1) limited two-person training data, inadequate to capture the diverse intricacies of two-person interactions; and 2) insufficiently fine-grained text-to-interaction modeling, where language conditioning collapses rich, structured prompts into a single sentence embedding. To address these limitations, we propose our Text2Interact framework, designed to generate realistic, text-aligned human-human interactions through a scalable high-fidelity interaction data synthesizer and an effective spatiotemporal coordination pipeline. First, we present InterCompose, a scalable synthesis-by-composition pipeline that aligns LLM-generated interaction descriptions with strong single-person motion priors. Given a prompt and a motion for an agent, InterCompose retrieves candidate single-person motions, trains a conditional reaction generator for another agent, and uses a neural motion evaluator to filter weak or misaligned samples-expanding interaction coverage without extra capture. Second, we propose InterActor, a text-to-interaction model with word-level conditioning that preserves token-level cues (initiation, response, contact ordering) and an adaptive interaction loss that emphasizes contextually relevant inter-person joint pairs, improving coupling and physical plausibility for fine-grained interaction modeling. Extensive experiments show consistent gains in motion diversity, fidelity, and generalization, including out-of-distribution scenarios and user studies. We will release code and models to facilitate reproducibility.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **Motion Anything: Any to Motion Generation**  | 12 Mar 2025 |          [Link](https://arxiv.org/abs/2503.06955)          | [Link](https://github.com/steve-zeyu-zhang/MotionAnything)  | [Link](https://steve-zeyu-zhang.github.io/MotionAnything/)  |
| 2025 | **Animating the Uncaptured: Humanoid Mesh Animation with Video Diffusion Models**  | 20 Mar 2025 |          [Link](https://arxiv.org/abs/2503.15996)          | --  | [Link](https://marcb.pro/atu/)  |
| 2025 | **FlowMotion: Target-Predictive Conditional Flow Matching for Jitter-Reduced Text-Driven Human Motion Generation**  | 20 Apr 2025 |          [Link](https://arxiv.org/abs/2504.01338)          | --  | --  |
| 2025 | **ReAlign: Bilingual Text-to-Motion Generation via Step-Aware Reward-Guided Alignment**  | 8 May 2025 |          [Link](https://www.arxiv.org/abs/2505.04974)          | --  | [Link](https://wengwanjiang.github.io/ReAlign-page/)  |
| 2025 | **UniHM: Universal Human Motion Generation with Object Interactions in Indoor Scenes**  | 19 May 2025 |          [Link](https://arxiv.org/abs/2505.12774)          | --  | -- |
| 2025 | **ReMoMask: Retrieval-Augmented Masked Motion Generation**  | 4 Aug 2025 |          [Link](https://arxiv.org/abs/2508.02605)          | [Link](https://github.com/AIGeeksGroup/ReMoMask)  | [Link](https://aigeeksgroup.github.io/ReMoMask/) |
| 2025 | **X-MoGen: Unified Motion Generation across Humans and Animals**  | 7 Aug 2025 |          [Link](https://www.arxiv.org/abs/2508.05162)          | --  | -- |
| 2025 | **EgoTwin: Dreaming Body and View in First Person**  | 18 Aug 2025 |          [Link](https://arxiv.org/abs/2508.13013)          | --  | [Link](https://egotwin.pages.dev/) |
| 2025 | **Pulp Motion: Framing-aware multimodal camera and human motion generation**  | 6 Oct 2025 |          [Link](https://arxiv.org/abs/2510.05097)          | [Link](https://github.com/robincourant/pulp-motion)  | [Link](https://www.lix.polytechnique.fr/vista/projects/2025_pulpmotion_courant/) |
| 2025 | **Text2Interact: High-Fidelity and Diverse Text-to-Two-Person Interaction Generation**  | 7 Oct 2025 |          [Link](https://arxiv.org/abs/2510.06504)          | --  | -- |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@article{zhang2025motion,
  title={Motion Anything: Any to Motion Generation},
  author={Zhang, Zeyu and Wang, Yiran and Mao, Wei and Li, Danning and Zhao, Rui and Wu, Biao and Song, Zirui and Zhuang, Bohan and Reid, Ian and Hartley, Richard},
  journal={arXiv preprint arXiv:2503.06955},
  year={2025}
}

@misc{millán2025animatinguncapturedhumanoidmesh,
        title={Animating the Uncaptured: Humanoid Mesh Animation with Video Diffusion Models}, 
        author={Marc Benedí San Millán and Angela Dai and Matthias Nießner},
        year={2025},
        eprint={2503.15996},
        archivePrefix={arXiv},
        primaryClass={cs.GR},
        url={https://arxiv.org/abs/2503.15996}, 
}

@article{cuba2025flowmotion,
  title={FlowMotion: Target-Predictive Flow Matching for Realistic Text-Driven Human Motion Generation},
  author={Cuba, Manolo Canales and Gois, Jo{\~a}o Paulo},
  journal={arXiv preprint arXiv:2504.01338},
  year={2025}
}

@article{weng2025realign,
  title={ReAlign: Bilingual Text-to-Motion Generation via Step-Aware Reward-Guided Alignment},
  author={Weng, Wanjiang and Tan, Xiaofeng and Wang, Hongsong and Zhou, Pan},
  journal={arXiv preprint arXiv:2505.04974},
  year={2025}
}

@misc{geng2025unihmuniversalhumanmotion,
      title={UniHM: Universal Human Motion Generation with Object Interactions in Indoor Scenes}, 
      author={Zichen Geng and Zeeshan Hayder and Wei Liu and Ajmal Mian},
      year={2025},
      eprint={2505.12774},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2505.12774}, 
}

@article{li2025remomask,
  title={ReMoMask: Retrieval-Augmented Masked Motion Generation},
  author={Li, Zhengdao and Wang, Siheng and Zhang, Zeyu and Tang, Hao},
  journal={arXiv preprint arXiv:2508.02605},
  year={2025}
}

@article{wang2025x,
  title={X-MoGen: Unified Motion Generation across Humans and Animals},
  author={Wang, Xuan and Ruan, Kai and Qian, Liyang and Guo, Zhizhi and Su, Chang and Wang, Gaoang},
  journal={arXiv preprint arXiv:2508.05162},
  year={2025}
}

@misc{xiu2025egotwindreamingbodyview,
      title={EgoTwin: Dreaming Body and View in First Person}, 
      author={Jingqiao Xiu and Fangzhou Hong and Yicong Li and Mengze Li and Wentao Wang and Sirui Han and Liang Pan and Ziwei Liu},
      year={2025},
      eprint={2508.13013},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.13013}, 
}

@misc{courant2025pulpmotionframingawaremultimodal,
      title={Pulp Motion: Framing-aware multimodal camera and human motion generation}, 
      author={Robin Courant and Xi Wang and David Loiseaux and Marc Christie and Vicky Kalogeiton},
      year={2025},
      eprint={2510.05097},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2510.05097}, 
}

@misc{wu2025text2interacthighfidelitydiversetexttotwoperson,
      title={Text2Interact: High-Fidelity and Diverse Text-to-Two-Person Interaction Generation}, 
      author={Qingxuan Wu and Zhiyang Dou and Chuan Guo and Yiming Huang and Qiao Feng and Bing Zhou and Jian Wang and Lingjie Liu},
      year={2025},
      eprint={2510.06504},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.06504}, 
}
```
</details>


---

### Motion Other Additional Info

### Previous Papers

#### Year 2023-2024
For more details, please check the [2023-2024 Text to Human Motion Papers](./docs/human_motion/motion_23-24.md), including 36 accepted papers and 6 arXiv papers.

### 📚 Dataset Works

#### 1. HUMOTO: A 4D Dataset of Mocap Human Object Interactions
Jiaxin Lu, Chun-Hao Paul Huang, Uttaran Bhattacharya, Qixing Huang, Yi Zhou

(University of Texas at Austin, Adobe Research)
<details span>
<summary><b>Abstract</b></summary>
We present Human Motions with Objects (HUMOTO), a high-fidelity dataset of human-object interactions for motion generation, computer vision, and robotics applications. Featuring 736 sequences (7,875 seconds at 30 fps), HUMOTO captures interactions with 63 precisely modeled objects and 72 articulated parts. Our innovations include a scene-driven LLM scripting pipeline creating complete, purposeful tasks with natural progression, and a mocap-and-camera recording setup to effectively handle occlusions. Spanning diverse activities from cooking to outdoor picnics, HUMOTO preserves both physical accuracy and logical task flow. Professional artists rigorously clean and verify each sequence, minimizing foot sliding and object penetrations. We also provide benchmarks compared to other datasets. HUMOTO's comprehensive full-body motion and simultaneous multi-object interactions address key data-capturing challenges and provide opportunities to advance realistic human-object interaction modeling across research domains with practical applications in animation, robotics, and embodied AI systems.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Dataset Page                     | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **HUMOTO: A 4D Dataset of Mocap Human Object Interactions**  | 14 Apr 2024  |          [Link](https://arxiv.org/abs/2504.10414)          | [Link](https://adobe-research.github.io/humoto/) | [Link](https://jiaxin-lu.github.io/humoto/)  |

<details close>
<summary>References</summary>

```
%axiv papers

@article{lu2025humoto,
  title={HUMOTO: A 4D Dataset of Mocap Human Object Interactions},
  author={Lu, Jiaxin and Huang, Chun-Hao Paul and Bhattacharya, Uttaran and Huang, Qixing and Zhou, Yi},
  journal={arXiv preprint arXiv:2504.10414},
  year={2025}
}


```
</details>


#### Datasets
   | Motion | Info |                              URL                              |               Others                            | 
   | :-----: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
   |  AIST |  AIST Dance Motion Dataset  | [Link](https://aistdancedb.ongaaccel.jp/) |--|
   |  AIST++  |  AIST++ Dance Motion Dataset | [Link](https://google.github.io/aistplusplus_dataset/) | [dance video database with SMPL annotations](https://google.github.io/aistplusplus_dataset/download.html) |
   |  AMASS  |  optical marker-based motion capture datasets  | [Link](https://amass.is.tue.mpg.de/) |--|

#### Additional Info
<details>
<summary>AMASS</summary>

AMASS is a large database of human motion unifying different optical marker-based motion capture datasets by representing them within a common framework and parameterization. AMASS is readily useful for animation, visualization, and generating training data for deep learning.
  
</details>

<details close>
<summary>Awesome Repos</summary>

> ##### Survey
* [TPAMI 2025] **Human Motion Video Generation: A Survey** [[arXiv](https://arxiv.org/abs/2509.03883)] [[Paper](https://ieeexplore.ieee.org/document/11106267)] [[GitHub](https://github.com/Winn1y/Awesome-Human-Motion-Video-Generation)]
* [TPAMI 2023] **Human Motion Generation: A Survey** [[Paper](https://arxiv.org/abs/2307.10894)]
* [arXiv 7 Apr 2025] **A Survey on Human Interaction Motion Generation** [[Paper](https://arxiv.org/abs/2503.12763)] [[GitHub](https://github.com/soraproducer/Awesome-Human-Interaction-Motion-Generation)]
	

</details>


--------------


## Text to 3D Human

### 🎉 Human Accepted Papers
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2503.15851)          | [Link](https://github.com/ZhenglinZhou/Zero-1-to-A)  | [Link](https://zhenglinzhou.github.io/Zero-1-to-A/)  |
| 2025 | **GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2503.11143)          | [Link](https://github.com/silence-tang/GaussianIP)  | [Link](https://silence-tang.github.io/gaussian-ip/)  |
| 2025 | **ArtiScene: Language-Driven Artistic 3D Scene Generation Through Image Intermediary**  | CVPR 2025 |          [Link](https://arxiv.org/abs/2506.00742)          | [Link](https://github.com/NVlabs/ArtiScene)  | [Link](https://artiscene-cvpr.github.io/)  |
| 2025 | **CAP4D: Creating Animatable 4D Portrait Avatars with Morphable Multi-View Diffusion Models**  | CVPR 2025 Oral |          [Link](https://arxiv.org/abs/2412.12093)          | [Link](https://github.com/felixtaubner/cap4d/)  | [Link](https://felixtaubner.github.io/cap4d/)  |
| 2025 | **Text-based Animatable 3D Avatars with Morphable Model Alignment**  | SIGGRAPH 2025 |          [Link](https://arxiv.org/abs/2504.15835)          | [Link](https://github.com/oneThousand1000/AnimPortrait3D)  | [Link](https://onethousandwu.com/animportrait3d.github.io/)  |
| 2025 | **LAM: Large Avatar Model for One-shot Animatable Gaussian Head**  | SIGGRAPH 2025 |          [Link](https://arxiv.org/abs/2502.17796)          | [Link](https://github.com/aigc3d/LAM)  | [Link](https://aigc3d.github.io/projects/LAM/)  |
| 2025 | **Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars**  | ICCV 2025 |          [Link](https://arxiv.org/abs/2502.20220)          | --  | [Link](https://tobias-kirschstein.github.io/avat3r/)  |
| 2025 | **SIGMAN:Scaling 3D Human Gaussian Generation with Millions of Assets**  | ICCV 2025 |          [Link](https://arxiv.org/abs/2504.06982)          | [Link](https://github.com/yyvhang/SIGMAN_release)  | [Link](https://yyvhang.github.io/SIGMAN_3D/)  |
| 2025 | **AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion**  | ICCV 2025 |          [Link](https://arxiv.org/abs/2505.24877)          | [Link](https://github.com/NVlabs/AdaHuman)  | [Link](https://nvlabs.github.io/AdaHuman/)  |
| 2025 | **MVP4D: Multi-View Portrait Video Diffusion for Animatable 4D Avatars**  | SIGGRAPH Asia 2025 |          [Link](https://arxiv.org/abs/2510.12785)          | Code releases Nov 15th  | [Link](https://felixtaubner.github.io/mvp4d/)  |
| 2025 | **InfiniHuman: Infinite 3D Human Creation with Precise Control**  | SIGGRAPH Asia 2025 |          [Link](https://arxiv.org/abs/2510.11650)          | [Link](https://github.com/YuxuanSnow/InfiniHuman/)  | [Link](https://yuxuan-xue.com/infini-human/)  |

<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

@inproceedings{zhou2025zero1toa,
  title = {Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion},
  author = {Zhenglin Zhou and Fan Ma and Hehe Fan and Tat-Seng Chua},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
}

@article{tang2025gaussianip,
  title={GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior},
  author={Tang, Zichen and Yao, Yuan and Cui, Miaomiao and Bo, Liefeng and Yang, Hongyu},
  journal={arXiv preprint arXiv:2503.11143},
  year={2025}
}

@inproceedings{gu2025artiscene,
  title={ArtiScene: Language-Driven Artistic 3D Scene Generation Through Image Intermediary},
  author={Gu, Zeqi and Cui, Yin and Li, Zhaoshuo and Wei, Fangyin and Ge, Yunhao and Gu, Jinwei and Liu, Ming-Yu and Davis, Abe and Ding, Yifan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={2891--2901},
  year={2025}
}

@inproceedings{taubner2025cap4d,
  title={Cap4d: Creating animatable 4d portrait avatars with morphable multi-view diffusion models},
  author={Taubner, Felix and Zhang, Ruihang and Tuli, Mathieu and Lindell, David B},
  booktitle={2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={5318--5330},
  year={2025},
  organization={IEEE Computer Society}
}

@article{AnimPortrait3D_sig25,
      author = {Wu, Yiqian and Prinzler, Malte and Jin, Xiaogang and Tang, Siyu},
      title = {Text-based Animatable 3D Avatars with Morphable Model Alignment},
      year = {2025}, 
      isbn = {9798400715402}, 
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3721238.3730680},
      doi = {10.1145/3721238.3730680},
      articleno = {},
      numpages = {11},
      location = {Vancouver, BC, Canada},
      series = {SIGGRAPH '25}
}

@article{he2025lam,
  title={LAM: Large Avatar Model for One-shot Animatable Gaussian Head},
  author={He, Yisheng and Gu, Xiaodong and Ye, Xiaodan and Xu, Chao and Zhao, Zhengyi and Dong, Yuan and Yuan, Weihao and Dong, Zilong and Bo, Liefeng},
  journal={arXiv preprint arXiv:2502.17796},
  year={2025}
}

@article{kirschstein2025avat3r,
  title={Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars},
  author={Kirschstein, Tobias and Romero, Javier and Sevastopolsky, Artem and Nie{\ss}ner, Matthias and Saito, Shunsuke},
  journal={arXiv preprint arXiv:2502.20220},
  year={2025}
}

@article{yang2025sigman,
  title={SIGMAN: Scaling 3D Human Gaussian Generation with Millions of Assets},
  author={Yang, Yuhang and Liu, Fengqi and Lu, Yixing and Zhao, Qin and Wu, Pingyu and Zhai, Wei and Yi, Ran and Cao, Yang and Ma, Lizhuang and Zha, Zheng-Jun and others},
  journal={arXiv preprint arXiv:2504.06982},
  year={2025}
}

@article{huang2025adahuman,
  title={AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion},
  author={Huang, Yangyi and Yuan, Ye and Li, Xueting and Kautz, Jan and Iqbal, Umar},
  journal={arXiv preprint arXiv:2505.24877},
  year={2025}
}

@article{taubner2025mvp4d,
  title={MVP4D: Multi-View Portrait Video Diffusion for Animatable 4D Avatars},
  author={Taubner, Felix and Zhang, Ruihang and Tuli, Mathieu and Bahmani, Sherwin and Lindell, David B},
  journal={arXiv preprint arXiv:2510.12785},
  year={2025}
}

@article{xue2025infinihuman,
  author    = {Xue, Yuxuan and Xie, Xianghui and Kostyrko, Margaret and Pons-Moll, Gerard},
  title     = {InfiniHuman: Infinite 3D Human Creation with Precise Control},
  booktitle = {SIGGRAPH Asia 2025 Conference Papers},
  year      = {2025},
}

```
</details>


---------

### 💡 Human ArXiv Papers

#### 1. HumanDreamer-X: Photorealistic Single-image Human Avatars Reconstruction via Gaussian Restoration
Boyuan Wang, Runqi Ouyang, Xiaofeng Wang, Zheng Zhu, Guosheng Zhao, Chaojun Ni, Guan Huang, Lihong Liu, Xingang Wang

(GigaAI, Institute of Automation Chinese Academy of Sciences, Peking University)
<details span>
<summary><b>Abstract</b></summary>
Single-image human reconstruction is vital for digital human modeling applications but remains an extremely challenging task. Current approaches rely on generative models to synthesize multi-view images for subsequent 3D reconstruction and animation. However, directly generating multiple views from a single human image suffers from geometric inconsistencies, resulting in issues like fragmented or blurred limbs in the reconstructed models. To tackle these limitations, we introduce \textbf{HumanDreamer-X}, a novel framework that integrates multi-view human generation and reconstruction into a unified pipeline, which significantly enhances the geometric consistency and visual fidelity of the reconstructed 3D models. In this framework, 3D Gaussian Splatting serves as an explicit 3D representation to provide initial geometry and appearance priority. Building upon this foundation, \textbf{HumanFixer} is trained to restore 3DGS renderings, which guarantee photorealistic results. Furthermore, we delve into the inherent challenges associated with attention mechanisms in multi-view human generation, and propose an attention modulation strategy that effectively enhances geometric details identity consistency across multi-view. Experimental results demonstrate that our approach markedly improves generation and reconstruction PSNR quality metrics by 16.45% and 12.65%, respectively, achieving a PSNR of up to 25.62 dB, while also showing generalization capabilities on in-the-wild data and applicability to various human reconstruction backbone models.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **HumanDreamer-X: Photorealistic Single-image Human Avatars Reconstruction via Gaussian Restoration**  | 4 Apr 2025 |          [Link](https://arxiv.org/abs/2504.03536)          | [Link](https://github.com/GigaAI-research/HumanDreamer-X)  | [Link](https://humandreamer-x.github.io/)  |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@article{wang2025humandreamerx,
  title={HumanDreamer-X: Photorealistic Single-image Human Avatars Reconstruction via Gaussian Restoration}, 
  author={Boyuan Wang and Runqi Ouyang and Xiaofeng Wang and Zheng Zhu and Guosheng Zhao and Chaojun Ni and Guan Huang and Lihong Liu and Xingang Wang},
  journal={arXiv preprint arXiv:2504.03536},
  year={2025}
}

```
</details>

### Additional Info
### Previous Papers

#### Year 2023-2024
For more details, please check the [2023-2024 3D Human Papers](./docs/3d_human/human_23-24.md), including 19 accepted papers and 4 arXiv papers.

<details close>
<summary>Survey and Awesome Repos</summary>
 
#### Survey
- [PROGRESS AND PROSPECTS IN 3D GENERATIVE AI: A TECHNICAL OVERVIEW INCLUDING 3D HUMAN](https://arxiv.org/pdf/2401.02620.pdf), ArXiv 2024
  
#### Awesome Repos
- Resource1: [Awesome Digital Human](https://github.com/weihaox/awesome-digital-human)
</details>

<details close>
<summary>Pretrained Models</summary>

   | Pretrained Models (human body) | Info |                              URL                              |
   | :-----: | :-----: | :----------------------------------------------------------: |
   |  SMPL  |  smpl model (smpl weights) | [Link](https://smpl.is.tue.mpg.de/) |
   |  SMPL-X  |  smpl model (smpl weights)  | [Link](https://smpl-x.is.tue.mpg.de/) |
   |  human_body_prior  |  vposer model (smpl weights)  | [Link](https://github.com/nghorbani/human_body_prior) |
<details>
<summary>SMPL</summary>

SMPL is an easy-to-use, realistic, model of the of the human body that is useful for animation and computer vision.

- version 1.0.0 for Python 2.7 (female/male, 10 shape PCs)
- version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)
- UV map in OBJ format
  
</details>

<details>
<summary>SMPL-X</summary>

SMPL-X, that extends SMPL with fully articulated hands and facial expressions (55 joints, 10475 vertices)

</details>
</details>

--------------

[<u>🎯Back to Top - Text2X Resources</u>](#-awesome-text2x-resources)


## Related Resources

### Text to 'other tasks'
Here, other tasks refer to CAD, 3D modeling, music generation, and so on.

> ##### Text to CAD
* [arXiv 7 Nov 2024] **CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM** [[Paper](https://arxiv.org/abs/2411.04954)] [[GitHub](https://github.com/CAD-MLLM/CAD-MLLM)] [[Project Page](https://cad-mllm.github.io/)]
* [NeurIPS 2024 Spotlight] **Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts** [[Paper](https://arxiv.org/abs/2409.17106)] [[GitHub](https://github.com/SadilKhan/Text2CAD)] [[Project Page](https://sadilkhan.github.io/text2cad-project/)] [[Dataset](https://huggingface.co/datasets/SadilKhan/Text2CAD)]
* [CVPR 2025] **CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation** [[Paper](https://arxiv.org/abs/2505.04481)] 


> ##### Text to Music
* [arXiv 1 Sep 2024] **FLUX that Plays Music** [[Paper](https://arxiv.org/abs/2409.00587)] [[GitHub](https://github.com/feizc/FluxMusic)]
* [International Society for Music Information Retrieval(ISMIR) 2025] **Video-Guided Text-to-Music Generation Using Public Domain Movie Collections** [[Paper](https://arxiv.org/abs/2506.12573)] [[Code](https://github.com/havenpersona/ossl-v1)] [[Project Page](https://havenpersona.github.io/ossl-v1/)] 


> ##### Text to Model
* [ICLR Workshop on Neural Network Weights as a New Data Modality 2025] **Text-to-Model: Text-Conditioned Neural Network Diffusion for Train-Once-for-All Personalization** [[Paper](https://arxiv.org/abs/2405.14132)]



### Survey and Awesome Repos 

<details close>
<summary>🔥 Topic 1: 3D Gaussian Splatting</summary>

> ##### Survey
* [arXiv 6 May 2024] **Gaussian Splatting: 3D Reconstruction and Novel View Synthesis, a Review** [[Paper](https://arxiv.org/abs/2405.03417)]
* [arXiv 17 Mar 2024] **Recent Advances in 3D Gaussian Splatting** [[Paper](https://arxiv.org/abs/2403.11134)]
* [IEEE TVCG 2024] **3D Gaussian as a New Vision Era: A Survey** [[Paper](https://arxiv.org/abs/2402.07181)]
* [arXiv 8 Jan 2024] **A Survey on 3D Gaussian Splatting** [[Paper](https://arxiv.org/abs/2401.03890)] [[GitHub](https://github.com/guikunchen/Awesome3DGS)] [[Benchmark](https://github.com/guikunchen/3DGS-Benchmarks)]
  
> ##### Awesome Repos
- Resource1: [Awesome 3D Gaussian Splatting Resources](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)
- Resource2: [3D Gaussian Splatting Papers](https://github.com/Awesome3DGS/3D-Gaussian-Splatting-Papers)
- Resource3: [3DGS and Beyond Docs](https://github.com/yangjiheng/3DGS_and_Beyond_Docs)

</details>

<details close>
<summary>🔥 Topic 2: AIGC 3D </summary>
 
> ##### Survey
* [arXiv 15 May 2024] **A Survey On Text-to-3D Contents Generation In The Wild** [[Paper](https://arxiv.org/abs/2405.09431)]
* [arXiv 2 Feb 2024] **A Comprehensive Survey on 3D Content Generation** [[Paper](https://arxiv.org/abs/2402.01166)] [[GitHub](https://github.com/hitcslj/Awesome-AIGC-3D)]
* [arXiv 31 Jan 2024] **Advances in 3D Generation: A Survey** [[Paper](https://arxiv.org/abs/2401.17807)]

> ##### Awesome Repos
- Resource1: [Awesome 3D AIGC Resources](https://github.com/mdyao/Awesome-3D-AIGC)
- Resource2: [Awesome-Text/Image-to-3D](https://github.com/StellarCheng/Awesome-Text-to-3D)
- Resource3: [Awesome Text-to-3D](https://github.com/yyeboah/Awesome-Text-to-3D)


> ##### Benchmark
* [CVPR 2024] **GPT-4V(ision) is a Human-Aligned Evaluator for Text-to-3D Generation** [[Paper](https://arxiv.org/abs/2401.04092)] [[GitHub](https://github.com/3DTopia/GPTEval3D)] [[Project Page](https://gpteval3d.github.io/)]
  
> ##### Foundation Model
* [arXiv 19 Mar 2025] **Cube: A Roblox View of 3D Intelligence** [[Paper](https://arxiv.org/abs/2503.15475)] [[GitHub](https://github.com/Roblox/cube)]

</details>

<details close>
<summary>🔥 Topic 3: 3D Human & LLM 3D</summary>

> ##### Survey
* [arXiv 6 June 2024] **A Survey on 3D Human Avatar Modeling -- From Reconstruction to Generation** [[Paper](https://arxiv.org/abs/2406.04253)]
* [arXiv 5 Jan 2024] **Progress and Prospects in 3D Generative AI: A Technical Overview including 3D human** [[Paper](https://arxiv.org/abs/2401.02620)] 

> ##### Awesome Repos
- Resource1: [Awesome LLM 3D](https://github.com/ActiveVisionLab/Awesome-LLM-3D)
- Resource2: [Awesome Digital Human](https://github.com/weihaox/awesome-digital-human)
- Resource3: [Awesome-Avatars](https://github.com/pansanity666/Awesome-Avatars)

</details>

<details close>
<summary>🔥 Topic 4: AIGC 4D </summary>

> ##### Survey
* [arXiv 18 Mar 2025] **Advances in 4D Generation: A Survey** [[Paper](https://arxiv.org/abs/2503.14501)] [[GitHub](https://github.com/MiaoQiaowei/Awesome-4D)]
	
> ##### Awesome Repos
- Resource1: [Awesome 4D Generation](https://github.com/cwchenwang/awesome-4d-generation)

</details>

<details close>
<summary>🔥 Topic 5: Physics-based AIGC</summary>

> ##### Survey
* [arXiv 27 Mar 2025] **Exploring the Evolution of Physics Cognition in Video Generation: A Survey** [[Paper](https://arxiv.org/abs/2503.21765)] [[GitHub](https://github.com/minnie-lin/Awesome-Physics-Cognition-based-Video-Generation)]
* [arXiv 19 Jan 2025] **Generative Physical AI in Vision: A Survey** [[Paper](https://arxiv.org/abs/2501.10928)] [[GitHub](https://github.com/BestJunYu/Awesome-Physics-aware-Generation)]

</details>

<details close>
<summary>Dynamic Gaussian Splatting</summary>

> ##### Neural Deformable 3D Gaussians
* [CVPR 2024] **Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction** [[Paper](https://arxiv.org/abs/2309.13101)] [[GitHub](https://github.com/ingra14m/Deformable-3D-Gaussians)] [[Project Page](https://ingra14m.github.io/Deformable-Gaussians/)]
* [CVPR 2024] **4D Gaussian Splatting for Real-Time Dynamic Scene Rendering** [[Paper](https://arxiv.org/abs/2310.08528)] [[GitHub](https://github.com/hustvl/4DGaussians)] [[Project Page](https://guanjunwu.github.io/4dgs/index.html)]
* [CVPR 2024] **SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes** [[Paper](https://arxiv.org/abs/2312.14937)] [[GitHub](https://github.com/yihua7/SC-GS)] [[Project Page](https://yihua7.github.io/SC-GS-web/)]
* [CVPR 2024 Highlight] **3DGStream: On-the-Fly Training of 3D Gaussians for Efficient Streaming of Photo-Realistic Free-Viewpoint Videos** [[Paper](https://arxiv.org/abs/2403.01444)] [[GitHub](https://github.com/SJoJoK/3DGStream)] [[Project Page](https://sjojok.github.io/3dgstream/)]


> ##### 4D Gaussians
* [SIGGRAPH 2024] **4D-Rotor Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes** [[Paper](https://arxiv.org/abs/2402.03307)] 
* [ICLR 2024] **Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting** [[Paper](https://arxiv.org/abs/2310.10642)] [[GitHub](https://github.com/fudan-zvg/4d-gaussian-splatting)] [[Project Page](https://fudan-zvg.github.io/4d-gaussian-splatting/)]


> ##### Dynamic 3D Gaussians
* [CVPR 2024 Highlight] **Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle** [[Paper](https://arxiv.org/abs/2312.03431)] [[GitHub](https://github.com/NJU-3DV/Gaussian-Flow)] [[Project Page](https://nju-3dv.github.io/projects/Gaussian-Flow/)]
* [3DV 2024] **Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis** [[Paper](https://arxiv.org/abs/2308.09713)] [[GitHub](https://github.com/JonathonLuiten/Dynamic3DGaussians)] [[Project Page](https://dynamic3dgaussians.github.io/)]


</details>

--------------

[<u>🎯Back to Top - Table of Contents</u>](#table-of-contents)


## License 
This repo is released under the [MIT license](./LICENSE).

✉️ Any additions or suggestions, feel free to contact us. 
