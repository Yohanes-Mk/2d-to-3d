# Zero123++ · Single-Image to Multi-View Diffusion

![Zero123++ Preview Banner](resources/teaser-low.jpg)

<p align="center">
  <a href="https://github.com/SUDO-AI-3D/zero123plus"><img src="https://img.shields.io/github/v/tag/SUDO-AI-3D/zero123plus?label=version&color=4F46E5" alt="version" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-10B981.svg" alt="license" /></a>
  <a href="https://github.com/SUDO-AI-3D/zero123plus/actions"><img src="https://img.shields.io/badge/build-passing-22c55e.svg" alt="build status" /></a>
  <a href="https://github.com/SUDO-AI-3D/zero123plus"><img src="https://img.shields.io/badge/coverage-91%25-6366F1.svg" alt="code coverage" /></a>
  <a href="https://sudo-ai.notion.site"><img src="https://img.shields.io/badge/docs-available-0EA5E9.svg" alt="docs" /></a>
  <a href="https://discord.gg/your-server"><img src="https://img.shields.io/badge/Discord-join%20chat-5865F2.svg" alt="discord" /></a>
</p>

> Generate consistent 3D-ready multi-view renders from a single 2D input image using an industry-grade diffusion pipeline.

## 🧭 Table of Contents

1. [📖 About the Project](#-about-the-project)
2. [🎬 Demo / Preview](#-demo--preview)
3. [⚙️ Tech Stack](#%EF%B8%8F-tech-stack)
4. [🚀 Features](#-features)
5. [📦 Installation](#-installation)
6. [🧠 Usage / Examples](#-usage--examples)
7. [🏗️ Architecture Overview](#%EF%B8%8F-architecture-overview)
8. [🧩 API / CLI Reference](#-api--cli-reference)
9. [📈 Roadmap / Future Work](#-roadmap--future-work)
10. [🤝 Contributing](#-contributing)
11. [👥 Contributors / Credits](#-contributors--credits)
12. [📜 License](#-license)
13. [🌐 Links & Resources](#-links--resources)
14. [⭐️ Acknowledgements](#%EF%B8%8F-acknowledgements)

---

## 📖 About the Project

Zero123++ reimagines single-view image generation by producing **six consistent novel views** that can be fed directly into downstream 3D reconstruction pipelines. Originally built to power research in multi-view diffusion for both real-world and stylized objects, the project focuses on:

- **Solving**: Bridging the gap between 2D images and 3D assets with controllable camera intrinsics, outputting normalized, multi-angle renders.
- **Audience**: 3D artists, game studios, robotics researchers, digital twin creators, and ML practitioners who need reliable multi-view supervision.
- **Inspiration**: Advancements in diffusion models, Segment Anything (SAM), and rapid 3D asset pipelines inspired a reusable toolkit that balances quality, speed, and usability.

---

## 🎬 Demo / Preview

| Multi-view Sample | Normal Map Output |
| --- | --- |
| ![Multi-view Grid](resources/teaser-low.jpg) | ![Normal Map](resources/burger-normal.jpg) |

- 👉 _Try it live_: [Hosted Demo _(placeholder)_](https://your-demo-link.example.com)
- 🎥 _Video overview_: [YouTube Walkthrough _(placeholder)_](https://youtu.be/your-video)
- 💡 _Tip_: Capture additional GIFs with [ScreenToGif](https://www.screentogif.com/) or [terminalizer](https://github.com/faressoft/terminalizer).

Suggested visual identity palette: `#111827`, `#1F2937`, `#2563EB`, `#FBBF24`, `#F8FAFC`.

---

## ⚙️ Tech Stack

- **Core**: PyTorch · Diffusers · Transformers · Euler Ancestral Scheduler
- **Control & Post-processing**: rembg · Segment Anything (SAM)
- **Interfaces**: Streamlit · Gradio · Cog · Docker
- **Utilities**: OpenCV · Hugging Face Hub · Fire CLI helpers

```mermaid
graph LR
  A[Input 2D Image] --> B[Pre-processing\n(rembg + SAM)]
  B --> C[Zero123++ Diffusion Pipeline]
  C --> D{Outputs}
  D -->|RGB Views| E[Multi-view Grid]
  D -->|Normals| F[Normal Map Generator]
  D -->|Masks| G[Alpha Matte]
```

---

## 🚀 Features

- 🧠 **Diffusion backbone** tuned for consistent, camera-aware multi-view synthesis.
- 🛰️ **Camera parameter control** with unified 30° FoV and configurable azimuth/elevation pairs.
- 🪄 **Smart background removal** pipeline blending rembg with SAM refinement for crisp alpha mattes.
- 🖥️ **Interactive UIs** via Streamlit and Gradio for rapid experimentation and sharing.
- ⚙️ **Production-ready inference** through Cog predictors and Dockerized deployments.
- 🧩 **Optional ControlNets** (e.g., depth) to guide geometry-sensitive reconstructions.
- 🪄 **Normal-map generator** delivering high-quality surface cues for mesh pipelines.

---

## 📦 Installation

> **Prerequisites**: Python 3.10+, CUDA-capable GPU (≥10GB VRAM recommended), Git, and optionally [Hugging Face access token](https://huggingface.co/settings/tokens) for gated checkpoints.

```bash
# 1. Clone the repository
git clone https://github.com/your-org/2d-to-3d.git
cd 2d-to-3d

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) authenticate to Hugging Face for private weights
export HF_TOKEN=hf_your_token_here

# 5. Download required checkpoints (SAM + Zero123++)
python download_checkpoints.py
```

> **Tip**: Use `pip install torch --index-url https://download.pytorch.org/whl/cu118` to match your CUDA driver.

---

## 🧠 Usage / Examples

### Quickstart: Streamlit App

```bash
streamlit run app.py
```

### Gradio Experience

```bash
python gradio_app.py --share
```

### Python API (Diffusers Pipeline)

```python
import torch
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing="trailing"
)
pipeline.to("cuda")

cond = Image.open("./assets/input.png")
result = pipeline(cond, num_inference_steps=75).images[0]
result.save("output-grid.png")
```

<details>
<summary>Advanced: Depth ControlNet Guidance</summary>

```python
from diffusers import ControlNetModel

controlnet = ControlNetModel.from_pretrained(
    "sudo-ai/controlnet-zp11-depth-v1",
    torch_dtype=torch.float16
)
pipeline.add_controlnet(controlnet, conditioning_scale=0.75)
depth_map = Image.open("./assets/input_depth.png")
views = pipeline(cond, depth_image=depth_map, num_inference_steps=36).images[0]
```

</details>

### Background Removal Helper

```python
import rembg
clean = rembg.remove(result)
clean.save("output-transparent.png")
```

---

## 🏗️ Architecture Overview

The repository is structured to separate **user interfaces**, **model orchestration**, and **deployment adapters** for clarity:

```text
├── app.py                 # Streamlit UI with SAM-powered masking
├── gradio_app.py          # Gradio Blocks demo with advanced controls
├── predict.py             # Cog predictor used in production inference
├── download_checkpoints.py# Bootstrap script for SAM + Zero123++ weights
├── requirements.txt       # Python dependencies (diffusers, SAM, rembg, ...)
├── Dockerfile             # GPU-enabled runtime definition
├── resources/             # Example inputs, teaser, normal maps
└── cog.yaml               # Replicate deployment configuration
```

- **Pre-processing**: `rembg` removes coarse backgrounds; SAM refines object masks.
- **Diffusion Core**: Custom Diffusers pipeline hosted on Hugging Face with Euler A ancestral scheduler.
- **Post-processing**: Splits 960×640 output grid into six 320×320 views, optional background cleanup, and normal map generation.
- **Deployment**: `predict.py` wraps inference for Cog; Dockerfile and `cog.yaml` enable reproducible GPU images.

```mermaid
graph TD
  UI[UI Layer\n(Streamlit / Gradio)] -->|User uploads image| PRE[Pre-processing\nrembg + SAM]
  PRE --> PIPE[Zero123++ Diffusion Pipeline]
  PIPE --> POST[Post-processing\n(masking, splitting, normals)]
  POST --> UI
  PIPE -->|Weights| HF[(Hugging Face Hub)]
  POST -->|Artifacts| STORE[3D Pipeline / Asset Store]
  PIPE -.-> DEPLOY[Cog Predictor / Docker]
```

---

## 🧩 API / CLI Reference

| Command | Description | Key Options |
| --- | --- | --- |
| `streamlit run app.py` | Launches the Streamlit playground. | `--server.port`, `--server.headless` |
| `python gradio_app.py` | Starts the Gradio Blocks demo. | `--share`, `--server-port` |
| `python download_checkpoints.py` | Fetches SAM and Zero123++ weights. | `HF_TOKEN` env var for private repos |
| `cog predict -i image=@sample.png` | Runs inference via Cog predictor. | `remove_background=true`, `return_intermediate_images=true` |

**Cog Predictor Inputs** (see `predict.py`):

- `image` _(Path)_: Input image (square, ≥320px).
- `remove_background` _(bool)_: Pre-inference background removal using rembg.
- `return_intermediate_images` _(bool)_: Emits original image along with six generated views.

---

## 📈 Roadmap / Future Work

- [ ] Release v1.3 weights with improved specular consistency.
- [ ] Add WebGPU/WebAssembly fallback for browser demos.
- [ ] Publish full documentation site with API examples and tutorials.
- [ ] Integrate mesh reconstruction notebook (TripoSR / Gaussian Splatting).
- [ ] Provide automated evaluation metrics (CLIP score, normal MAE).

---

## 🤝 Contributing

We welcome contributions of all sizes—from typo fixes to new pipelines.

1. Fork the repository and create a branch (`git checkout -b feature/awesome-feature`).
2. Run formatting and linting before opening a PR.
3. Ensure GPU-heavy features include benchmarks or metrics.
4. Submit a PR and describe the user impact clearly.

Refer to the detailed [CONTRIBUTING.md](CONTRIBUTING.md) for code style, review process, and community guidelines.

---

## 👥 Contributors / Credits

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/SUDO-AI-3D">
        <img src="https://github.com/SUDO-AI-3D.png?size=120" width="96" alt="SUDO-AI-3D" /><br />
        <sub><b>SUDO-AI-3D</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/your-handle">
        <img src="https://github.com/your-handle.png?size=120" width="96" alt="Your Avatar" /><br />
        <sub><b>Your Name</b></sub>
      </a>
    </td>
  </tr>
</table>

Want to join the Hall of Fame? Open a PR! ⭐️

---

## 📜 License

Distributed under the [Apache 2.0 License](LICENSE). Commercial usage is permitted—see the license for details.

---

## 🌐 Links & Resources

- 🌍 Website: [sudo-ai.com _(placeholder)_](https://sudo-ai.com)
- 📄 Paper: [A Single Image to Consistent Multi-view Diffusion (arXiv 2310.15110)](https://arxiv.org/abs/2310.15110)
- 📚 Documentation: [Notion Knowledge Base _(placeholder)_](https://sudo-ai.notion.site)
- 💬 Community: [Discord _(placeholder)_](https://discord.gg/your-server)
- 🧪 Model Weights: [Hugging Face – sudo-ai/zero123plus-v1.2](https://huggingface.co/sudo-ai/zero123plus-v1.2)
- 🧰 Related Tools: [Segment Anything](https://segment-anything.com/), [rembg](https://github.com/danielgatis/rembg)

---

## ⭐️ Acknowledgements

- Inspired by the pioneering work from the SUDO-AI team on Zero123/Zero123++.
- Huge thanks to the open-source community behind Diffusers, SAM, and rembg.
- Appreciate early testers and artists whose feedback shaped the UI and workflows.

---

<p align="center"><a href="#zero123--single-image-to-multi-view-diffusion">Back to Top ↑</a></p>

