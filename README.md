
# SAR: Scene-Action Representation for End-to-End Autonomous Driving

> **Peiwei Chen**<sup>1,2,3</sup>, **Kaiqiu Xu**<sup>2</sup>, **Shengyin Fan**<sup>2</sup>, **Yudong Zhang**<sup>2</sup>, **Zhigang Ling**<sup>†1,3</sup>, **Yaonan Wang**<sup>1,3</sup>  
> <sup>1</sup> School of Artificial Intelligence and Robotics, Hunan University, Changsha, China  
> <sup>2</sup> [Tianyijiaotong Technology Ltd.](https://www.tyjt-ai.com/), Suzhou, China  
> <sup>3</sup> National Engineering Research Center of RVC, Changsha, China  
> (<sup>†</sup>) Corresponding author: zgling_hunan@126.com

## Introduction

We present **SAR (Scene-Action Representation)**, a novel end-to-end framework designed to enhance sparse scene understanding through structured behavior modeling, achieving state-of-the-art performance with remarkable efficiency. Inspired by human cognitive processes in driving, SAR decomposes the driving scene into three core, synergistic components: **global sparse scene semantics**, **ego-state awareness**, and **multi-agent behavioral dynamics**. These elements are seamlessly integrated via a dedicated **Scene-Action Transformer**, generating a cohesive, interpretable, and interaction-aware representation that facilitates high-quality, long-horizon trajectory planning.

<div align="center">
  <img src="resources/Pipeline Comparisons.JPG" width="800"/>
  <p><em>Figure 1: Comparison of SAR with conventional pipelines.</em></p>
</div>

## Method Overview

SAR ingests multi-modal sensor data, including LiDAR Bird's-Eye-View (BEV) features and camera images, to construct rich, multi-modal BEV representations through dynamic cross-view interactions. These representations are then distilled into distinct **sparse scene queries** and an **ego-centric query**. Concurrently, LiDAR BEV features are utilized to generate **agent-centric queries**. A purpose-built **Scene-Action Transformer** module orchestrates cascaded cross-modal interactions, significantly enriching the scene representation, which is pivotal for accurate and robust trajectory prediction.

<div align="center">
  <img src="resources/SAR0730.jpg" width="800"/>
  <p><em>Figure 2: Detailed architecture of the SAR framework.</em></p>
</div>

## 📋 Available Benchmarks

We offer comprehensive training and evaluation pipelines for a range of leading autonomous driving benchmarks:

*   **nuScenes**: The standard benchmark for autonomous driving perception and prediction.
*   **Bench2Drive**: A benchmark emphasizing rich annotations for closed-loop evaluation.
*   **NAVSIM**: A high-fidelity navigation simulation environment for scenario testing.

Select and configure the desired benchmark from the `code` directory to suit your research or development needs.

## 🏆 Key Results

### nuScenes

#### UniAD-style Protocol

| Method | L2<sub>MAX</sub> (m) 1s | L2<sub>MAX</sub> (m) 2s | L2<sub>MAX</sub> (m) 3s | L2<sub>MAX</sub> (m) Avg. | CR<sub>MAX</sub> (%) 1s | CR<sub>MAX</sub> (%) 2s | CR<sub>MAX</sub> (%) 3s | CR<sub>MAX</sub> (%) Avg. |Download |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SAR** | 0.24 | 0.63 | 1.31 | 0.73 | 0.07 | 0.11 | 0.63 | 0.27 |[Checkpoint](https://drive.google.com/file/d/1VZACt3DArmt15JSn-CL6pQVnegms8kE2/view?usp=drive_link) |

#### VAD-style Protocol

| Method | L2<sub>AVG</sub> (m) 1s | L2<sub>AVG</sub> (m) 2s | L2<sub>AVG</sub> (m) 3s | L2<sub>AVG</sub> (m) Avg. | CR<sub>AVG</sub> (%) 1s | CR<sub>AVG</sub> (%) 2s | CR<sub>AVG</sub> (%) 3s | CR<sub>AVG</sub> (%) Avg. | Download |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SAR** | 0.18 | 0.35 | 0.60 | 0.38 | 0.08 | 0.09 | 0.23 | 0.13 | [Checkpoint](https://drive.google.com/file/d/1VZACt3DArmt15JSn-CL6pQVnegms8kE2/view?usp=drive_link) |

### NAVSIM

| Method | NC   | DAC  | EP   | TTC  | Comfort | PDMS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SAR** | **98.4** | **96.7** | **82.7** | **94.8** | **100** | **88.5** |

### Bench2Drive

| Method | L2 (m) 3s | Driving Score | Success Rate (%) | Config | Download | Eval Json |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SAR** | 0.78 | 67.5 | 46.8 | [Config](code/Bench2Drive/adzoo/sar/configs/SAR/SAR_base_e2e_b2d.py) | Checkpoint | [New Version](code/Bench2Drive/analysis/SAR.json) |

## 🎥 Visualization

We showcase qualitative results of SAR on the nuScenes dataset, demonstrating its capability in complex urban driving scenarios.

### nuScenes Examples

<div align="center">
  <img src="resources/1a9c83c482c84c9e874e21aab1820190_vis.png" width="800"/>
  <br/>
  <img src="resources/nuscenes_demo.gif" width="800"/>
</div>

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file for details.

## Acknowledgements

SAR builds upon the foundational work of several outstanding projects. We extend our sincere gratitude to the authors and contributors of:

*   [VAD](https://github.com/hustvl/VAD)
*   [SSR](https://github.com/PeidongLi/SSR)
*   [NAVSIM](https://github.com/autonomousvision/navsim/tree/main)
*   [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
*   [TokenLearner](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner)

Their contributions have been invaluable to the advancement of this research.
