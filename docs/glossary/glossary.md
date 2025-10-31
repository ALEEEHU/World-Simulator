# Glossary of Technical Terms

This glossary provides definitions and references for key technical terms and acronyms used throughout our survey **"Simulating the Real World: A Unified Survey of Multimodal Generative Models"** to assist readers in understanding domain-specific concepts.

---

## Neural Representations

### NeRF (Neural Radiance Field)
A continuous volumetric function encoded in a Multi-Layer Perceptron (MLP) that maps 3D spatial positions and viewing directions to density and color values, enabling photorealistic novel view synthesis through volumetric rendering.

**Reference**: [Mildenhall et al., ECCV 2020](https://www.matthewtancik.com/nerf)

### 3DGS (3D Gaussian Splatting)
An efficient 3D scene representation that models objects as collections of anisotropic Gaussian distributions with learnable parameters (position, covariance, opacity, and appearance), enabling fast training and real-time rendering.

**Reference**: [Kerbl et al., SIGGRAPH 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

### SDF (Signed Distance Field/Function)
An implicit 3D representation that encodes geometry by storing the signed distance from any point in space to the nearest surface, with negative values inside objects and positive values outside.

**Reference**: [Park et al., CVPR 2019](https://github.com/facebookresearch/DeepSDF)


### Triplane
A memory-efficient 3D representation that decomposes 3D volumes into three orthogonal 2D feature planes (XY, XZ, YZ), enabling faster rendering and reduced memory consumption compared to voxel-based approaches.

**Reference**: [Chan et al., CVPR 2022](https://nvlabs.github.io/eg3d/)

### DMTet (Deep Marching Tetrahedra)
A hybrid 3D representation that combines implicit SDF-based modeling with explicit tetrahedral mesh extraction, enabling differentiable mesh generation with topology flexibility.

**Reference**: [Shen et al., NeurIPS 2021](https://nv-tlabs.github.io/DMTet/)

---

## Generative Models

### Diffusion Models
A class of generative models that learn to generate data by iteratively denoising samples from a Gaussian noise distribution, guided by learned score functions.

**References**: 
- [Ho et al., NeurIPS 2020](https://arxiv.org/abs/2006.11239)
- [Rombach et al., CVPR 2022](https://arxiv.org/abs/2112.10752)

### VAE (Variational Autoencoder)
A generative model that learns a probabilistic mapping between input data and a latent representation, enabling generation by sampling from the learned latent space.

**Reference**: [Kingma & Welling, ICLR 2014](https://arxiv.org/abs/1312.6114)

### VQ-VAE (Vector Quantized VAE)
An extension of VAE that uses discrete latent representations through vector quantization, enabling more stable training and better reconstruction quality.

**Reference**: [Aaron van den Oord et al., NeurIPS 2017](https://arxiv.org/abs/1711.00937)

### GAN (Generative Adversarial Network)
A generative model framework consisting of a generator and discriminator network trained adversarially to generate realistic samples.

**Reference**: [Goodfellow et al., NeurIPS 2014](https://arxiv.org/abs/1406.2661)

### LDM (Latent Diffusion Model)
Also known as Stable Diffusion, a diffusion model that operates in a compressed latent space rather than pixel space, significantly improving computational efficiency while maintaining high generation quality.

**Reference**: [Rombach et al., CVPR 2022](https://github.com/CompVis/latent-diffusion)

---

## Training and Optimization

### SDS (Score Distillation Sampling)
A distillation technique introduced in DreamFusion that enables optimization of 3D representations using pre-trained 2D diffusion models as supervision, transferring their learned priors to 3D generation without requiring 3D training data.

**Reference**: [Poole et al., ICLR 2023](https://dreamfusion3d.github.io/)

### CLIP (Contrastive Language-Image Pre-training)
A vision-language model trained on image-text pairs that learns aligned embeddings for images and text, enabling zero-shot image classification and serving as a powerful prior for multimodal generation tasks.

**Reference**: [Radford et al., ICML 2021](https://openai.com/research/clip)

---

## 3D Generation Paradigms

### MVS (Multi-View Stereo)
A 3D reconstruction technique that synthesizes 3D geometry from multiple 2D images captured from different viewpoints. In the context of 3D generation, MVS-based approaches first generate multi-view images and then reconstruct 3D models from them.


### Optimization-based Approaches
Methods that iteratively optimize 3D representations (e.g., NeRF parameters) for each input prompt using techniques like SDS loss, achieving high quality at the cost of longer generation times.

---

## Key Systems and Frameworks

### DreamFusion
A pioneering text-to-3D generation system that introduced Score Distillation Sampling (SDS) to optimize NeRF representations using pre-trained text-to-image diffusion models.

**Reference**: [Poole et al., ICLR 2023](https://dreamfusion3d.github.io/)

### Stable Diffusion (SD)
A widely-used open-source latent diffusion model for high-quality text-to-image generation, serving as a foundation for many multimodal generation systems.

**Reference**: [Rombach et al., CVPR 2022](https://stability.ai/stable-diffusion)

### Instant-NGP
An efficient neural graphics primitive framework using multi-resolution hash encoding for fast NeRF training and rendering.

**Reference**: [Müller et al., SIGGRAPH 2022](https://nvlabs.github.io/instant-ngp/)

### D-NeRF
Dynamic Neural Radiance Field that extends NeRF to model temporal changes by mapping scenes from a canonical space to time-varying deformed states.

**Reference**: [Pumarola et al., CVPR 2021](https://www.albertpumarola.com/research/D-NeRF/)

### 4DGS (4D Gaussian Splatting)
An extension of 3D Gaussian Splatting that models dynamic scenes by representing Gaussian attributes (position, scale, rotation) as time-dependent functions.

**Reference**: [Wu et al., CVPR 2024](https://guanjunwu.github.io/4dgs/)

---

## Additional Terminology in This Paper

### Cross-attention
An attention mechanism that allows a model to attend to features from different modalities or sources, commonly used to inject conditioning information (e.g., text embeddings) into generation models.

### Volume Rendering
A technique for generating 2D images from 3D volumetric data by integrating color and density along viewing rays, central to NeRF-based rendering.

### Deformation Field
A function that maps points from a canonical 3D space to deformed positions, enabling modeling of non-rigid motion and dynamic scenes.

### Canonical Space
A reference coordinate system where objects are represented in a standard pose or configuration, facilitating learning of deformations and articulations.

### Spherical Harmonics
A mathematical basis for representing functions on a sphere, used in 3DGS to encode view-dependent appearance efficiently.

### Hash Encoding
A technique using learnable hash tables to encode spatial features at multiple resolutions, enabling fast training and inference in neural rendering.

**Reference**: [Müller et al., SIGGRAPH 2022](https://nvlabs.github.io/instant-ngp/)

---

## Common Abbreviations

Throughout this survey, we use standard abbreviations for common model architectures and techniques:

| Abbreviation | Full Form | Description |
|--------------|-----------|-------------|
| **2D/3D/4D** | Two/Three/Four Dimensional | Referring to images, spatial geometry, and spatial+temporal content |
| **T2I** | Text-to-Image | Text-to-image generation |
| **T2V** | Text-to-Video | Text-to-video generation |
| **T23D** | Text-to-3D | Text-to-3D generation |
| **T24D** | Text-to-4D | Text-to-4D generation |
| **I23D** | Image-to-3D | Image-to-3D generation |
| **V23D** | Video-to-3D | Video-to-3D generation |
| **DiT** | Diffusion Transformer | Transformer-based diffusion models |
| **U-Net** | U-shaped Network | A convolutional neural network architecture with encoder-decoder structure and skip connections |
| **MLP** | Multi-Layer Perceptron | Fully connected neural network |

For DiT, the seminal paper refers to [Peebles, W. and Xie, S., ICCV 2023](https://arxiv.org/abs/2212.09748).
For U-Net, the seminal paper refers to [Ronneberger et al., MICCAI 2015](https://arxiv.org/pdf/1505.04597).


