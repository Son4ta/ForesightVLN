# ForesightVLN: Dynamic Adaptive Framework for Efficient Zero-Shot Navigation in Semantically Sparse Environments

### [Paper (Coming Soon)]() | [Project Page (Coming Soon)]() | [Video Demo (Coming Soon)]()

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

Official PyTorch implementation for the paper **"ForesightVLN: Dynamic Adaptive Framework for Efficient Zero-Shot Navigation in Semantically Sparse Environments"**.

ForesightVLN is a universal, **zero-shot** framework for Vision-Language Navigation (VLN) that excels in challenging, semantically sparse environments. It intelligently balances broad, efficient exploration with precise target localization to navigate efficiently without any task-specific training. Our work builds upon the excellent foundation laid by the UniGoal project.

![ForesightVLN Pipeline](assets/pipeline.png)

---

## 🌟 Key Features

-   **🧠 Adaptive Exploration Strategy**: Dynamically switches between efficient geometric exploration (in sparse areas) and focused semantic exploration (in promising regions) based on environmental cues.
-   **🗺️ Dual-Representation System**: Synergistically uses a context-aware semantic value map for high-level guidance and a scene graph for precise target matching.
-   **⚙️ TSP-Based Global Path Planner**: Optimizes exploration paths by solving the Traveling Salesperson Problem (TSP), eliminating path redundancy and oscillations common in greedy approaches.
-   **🏆 State-of-the-Art Performance**: Achieves new SOTA results on standard VLN benchmarks like **MP3D**, **HM3D**, and **RoboTHOR** across Object-goal, Instance-Image-goal, and Text-goal navigation tasks.

---

## 🛠️ Installation

### Step 1: Clone Repository & Create Environment

```bash
# Clone the ForesightVLN repository
git clone [https://github.com/your-username/ForesightVLN.git](https://github.com/your-username/ForesightVLN.git)
cd ForesightVLN

# Create and activate a Conda environment
conda create -n foresightvln python=3.8 -y
conda activate foresightvln
````

### Step 2: Install Dependencies

Install the required packages using the provided `requirements.txt` and `third_party/habitat-lab/requirements.txt` files.

```bash
# Install primary dependencies
pip install -r requirements.txt

# Install Habitat-Lab specific dependencies
pip install -r third_party/habitat-lab/requirements.txt

# Install Habitat-Lab in editable mode
pip install -e third_party/habitat-lab
```

### Step 3: Install Third-Party Libraries

Our framework relies on several powerful third-party libraries for perception and matching.

```bash
# Install Detectron2 for object detection
pip install git+[https://github.com/facebookresearch/detectron2.git](https://github.com/facebookresearch/detectron2.git)

# Install Pytorch3D for 3D operations
pip install git+[https://github.com/facebookresearch/pytorch3d.git](https://github.com/facebookresearch/pytorch3d.git)

# Install LightGlue for feature matching
pip install git+[https://github.com/cvg/LightGlue.git](https://github.com/cvg/LightGlue.git)

# Install Grounded-Segment-Anything for open-vocabulary segmentation
git clone [https://github.com/IDEA-Research/Grounded-Segment-Anything.git](https://github.com/IDEA-Research/Grounded-Segment-Anything.git) third_party/Grounded-Segment-Anything
cd third_party/Grounded-Segment-Anything
# Checkout to a specific commit for stability
git checkout 5cb813f
pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
cd ../../
```

### Step 4: Download Pre-trained Models

Download the necessary pre-trained model weights for segmentation and detection.

```bash
# Create directory for models
mkdir -p data/models/

# Download SAM model
wget -O data/models/sam_vit_h_4b8939.pth [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

# Download GroundingDINO model
wget -O data/models/groundingdino_swint_ogc.pth [https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
```

### Step 5: (Optional) LLM & VLM Setup with Ollama

Our framework uses a local LLM/VLM for semantic reasoning. We recommend using [Ollama](https://ollama.com/) for easy setup.

```bash
# Install Ollama
curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh

# Pull the required models (example uses gemma3)
ollama pull gemma3:12b-it-qat
```

> **Note**: You can also configure a different LLM/VLM provider by modifying the `api_key`, `base_url`, and model names in your configuration file (e.g., `configs/config_habitat.yaml`).

-----

## 💾 Dataset Preparation

Download the required datasets for the navigation tasks.

1.  **HM3D Scene Dataset**: Download from the [official source](https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar).
2.  **Instance-Image-Goal Episodes**: Download from the [Habitat challenge page](https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip).
3.  **Text-Goal Episodes**: Download from our provided [Google Drive link](https://drive.google.com/uc?export=download&id=1KNdv6isX1FDZi4KCVPiECYDxijg9cZ3L).

Please structure your `data` directory as follows:

```
ForesightVLN/
└── data/
    ├── datasets/
    │   ├── textnav/
    │   │   └── val/
    │   │       └── val_text.json.gz
    │   └── instance_imagenav/
    │       └── hm3d/
    │           └── v3/
    │               └── val/
    │                   ├── content/
    │                   └── val.json.gz
    └── scene_datasets/
        └── hm3d_v0.2/
            └── val/
                ├── 00800-TEEsavR23oF/
                └── ...
```

-----

## 🚀 Running Evaluation

You can easily run evaluation on the standard benchmarks with the following commands.

**Instance-Image-Goal Navigation on HM3D:**

```bash
python main.py --config-file configs/config_habitat.yaml --goal_type ins-image
```

**Text-Goal Navigation on HM3D:**

```bash
python main.py --config-file configs/config_habitat.yaml --goal_type text
```

**Object-Goal Navigation on RoboTHOR (AI2-THOR):**

```bash
python main.py --config-file configs/config_ai2thor.yaml --goal_type object
```

> Ensure your Ollama server is running if you are using it as the LLM/VLM backend.

-----

## 🔬 Code Structure

The repository is organized as follows:

```
ForesightVLN/
├── main.py                    # Main script for running evaluation
├── configs/                   # Configuration files for different environments and tasks
├── src/
│   ├── agent/                 # Agent logic, including UniGoal agent implementation
│   │   └── unigoal/agent.py
│   ├── envs/                  # Environment wrappers for Habitat, AI2-THOR, etc.
│   ├── graph/                 # Core logic for Scene Graph and Goal Graph
│   │   ├── graph.py
│   │   ├── graphbuilder.py
│   │   └── goalgraphdecomposer.py
│   └── map/
│       └── bev_mapping.py     # Bird's-Eye-View (BEV) occupancy map creation
└── third_party/               # External libraries like Habitat-Lab and Grounded-SAM
```

-----

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----

## ✏️ Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{lu2025foresightvln,
  title={ForesightVLN: Dynamic Adaptive Framework for Efficient Zero-Shot Navigation in Semantically Sparse Environments},
  author={Lü, Zhiwei and Fang, Chengjie and Lai, Xinhua and Chen, Dongjie and Xu, Jungang},
  booktitle={Conference Proceedings},
  year={2025}
}
```

-----

## 🙏 Acknowledgements

Our work builds upon the fantastic research from the Embodied AI community and utilizes several open-source projects. We especially thank the authors of [Unigoal](https://github.com/bagh2178/UniGoal) for their foundational work, which inspired our research. We also thank the creators of [Habitat](https://aihabitat.org/), [Detectron2](https://github.com/facebookresearch/detectron2), [LightGlue](https://github.com/cvg/LightGlue), and [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) for their invaluable contributions.

```
```
