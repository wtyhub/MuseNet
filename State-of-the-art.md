## Aerial-view Geo-localization

  * [University-1652 Dataset](#university-1652-dataset)
  * [SUES-200 Dataset](#sues-200-dataset)
  * [CVUSA Dataset](#cvusa-dataset)

### University-1652 Dataset

| Methods | R@1 | AP | R@1 | AP | Reference |
| ------- | --- | -- | --- | -- | --------- |
| | Drone -> Satellite | | Satellite -> Drone | |
| VGG16 | 49.96 | 55.03 | 69.52 | 48.50 | Karen Simonyan, Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR 2015. [[Paper]](https://arxiv.org/abs/1409.1556) [[Code]](https://github.com/Prabhu204/Very-Deep-Convolutional-Networks-for-Large-Scale-Image-Recognition)|
| Zheng et al. | 55.17 | 59.71 | 75.33 | 54.68 | Zhedong Zheng, Yunchao Wei, Yi Yang. University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization. MM 2020. [[Paper]](https://dl.acm.org/doi/abs/10.1145/3394171.3413896)[[Code]](https://github.com/layumi/University1652-Baseline) |
| ResNet-101 | 58.76 | 63.29 | 80.13 | 60.74 | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning For Image Recognition. CVPR 2016. [[Paper]](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)[[Code]](https://github.com/KaimingHe/deep-residual-networks) |
| DenseNet121 | 59.01 | 63.44 | 78.55 | 60.15 | Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger. Densely Connected Convolutional Networks. CVPR 2017. [[Paper]](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html)[[Code]](https://github.com/liuzhuang13/DenseNet) |
| Swin-T | 61.56 | 65.95 | 78.53 | 61.83 | Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo. Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows. ICCV 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper)[[Code]](https://github.com/microsoft/Swin-Transformer) |
| IBN-Net | 62.30 | 66.46 | 82.27 | 63.36 | Xingang Pan, Ping Luo, Jianping Shi, Xiaoou Tang. Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net. ECCV 2018. [[Paper]](https://openaccess.thecvf.com/content_ECCV_2018/html/Xingang_Pan_Two_at_Once_ECCV_2018_paper.html)[[Code]](https://github.com/Asthestarsfalll/IBNNet-MegEngine) |
| LPN | 64.16 | 68.14 | 83.64 | 65.08 | Tingyu Wang, Zhedong Zheng, Chenggang Yan, and Yi, Yang. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021.[[Paper]](https://arxiv.org/abs/2008.11646)[[Code]](https://github.com/wtyhub/LPN) |
| MuSe-Net | 65.15 | 69.16 | 84.68 | 65.75 | Tingyu Wang, Zhedong Zheng, Yaoqi Sun, Chenggang Yan, Yi Yang, Tat-Seng Chua. Multiple-environment Self-adaptive Network for Aerial-view Geo-localization. PR 2024. [[Paper]](https://arxiv.org/abs/2204.08381)[[Code]](https://github.com/wtyhub/MuseNet) |
| Safe-Net | 76.01 | 79.06 | - | - | Jinliang Lin, Zhiming Luo,  Dazhen Lin, Shaozi Li, Zhun Zhong. A Self-Adaptive Feature Extraction Method for Aerial-View Geo-Localization. TIP 2025. [[Paper]](https://ieeexplore.ieee.org/abstract/document/10797651) (Lack of rain, snow) |
| WeatherPrompt | 77.14 | 80.20 | 87.72 | 76.39 | Jiahao Wen, Hang Yu, Zhedong Zheng. WeatherPrompt: Multi-modality Representation Learning for All-Weather Drone Visual Geo-Localization. NeurIPS 2025. [[Paper]](https://arxiv.org/abs/2508.09560)[[Code]](https://githubcom/Jahawn-Wen/WeatherPrompt) |
| LRFR | 84.73 | 87.18 | - | - | Wenjian Gan, Yang Zhou, Xiaofei Hu, Luying Zhao, Gaoshuang Huang, Mingbo Hou. Learning robust feature representation for cross-view image geo-localization. GRSL 2025. [[Paper]](https://ieeexplore.ieee.org/abstract/document/10896706)[[Code]](https://github.com/WenjianGan/LRFR) (Lack of rain, snow) |
| CDM-Net | 84.85 | 85.74 | - | - | Xin Zhou, Xuerong Yang, Yanchun Zhang. CDM-Net: A Framework for Cross-View Geo-Localization With Multimodal Data. TGRS 2025. [[Paper]](https://ieeexplore.ieee.org/abstract/document/11105551)[[Code]](https://github.com/cver6/CDM-Net) (Lack of rain, snow) |
| Sample4Geo | 85.58 | 88.24 | 94.21 | 85.50 | Fabian Deuser, Konrad Habel, Norbert Oswald. Sample4Geo: Hard Negative Sampling For Cross-View Geo-Localisation. ICCV 2023. [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Deuser_Sample4Geo_Hard_Negative_Sampling_For_Cross-View_Geo-Localisation_ICCV_2023_paper.html)[[Code]](https://github.com/Skyy93/Sample4Geo) |
| DAC | 89.64 | 91.35 | - | - | Panwang Xia, Yi Wan, Zhi Zheng, Yongjun Zhang. Enhancing Cross-View Geo-Localization With Domain Alignment and Scene Consistency. TCSVT 2024. [[Paper]](https://ieeexplore.ieee.org/abstract/document/10054158)[[Code]](https://github.com/SummerpanKing/DAC) (Lack of rain, snow) |
| CGSI | 92.65 | 93.66 | 94.39 | 92.22 | Jian Sun, Junlang Huang, Xinyu Jiang, Yimin Zhou, Chi-Man VONG. CGSI: Context-Guided and UAV’s Status Informed Multimodal Framework for Generalizable Cross-View Geo-Localization. [[Paper]](https://ieeexplore.ieee.org/abstract/document/11145113) |


### SUES-200 Dataset

| Methods | R@1 | AP | R@1 | AP | Reference |
| ------- | --- | -- | --- | -- | --------- |
| | Drone -> Satellite | | Satellite -> Drone | |
| Zheng et al. | 33.40 | 39.63 | 45.75 | 30.58 | Zhedong Zheng, Yunchao Wei, Yi Yang. University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization. MM 2020. [[Paper]](https://dl.acm.org/doi/abs/10.1145/3394171.3413896)[[Code]](https://github.com/layumi/University1652-Baseline) |
| IBN-Net | 39.58 | 46.23 | 52.25 | 37.78 | Xingang Pan, Ping Luo, Jianping Shi, Xiaoou Tang. Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net. ECCV 2018. [[Paper]](https://openaccess.thecvf.com/content_ECCV_2018/html/Xingang_Pan_Two_at_Once_ECCV_2018_paper.html)[[Code]](https://github.com/Asthestarsfalll/IBNNet-MegEngine) |
| MuSe-Net | 41.59 | 48.53 | 53.38 | 39.20 | Tingyu Wang, Zhedong Zheng, Yaoqi Sun, Chenggang Yan, Yi Yang, Tat-Seng Chua. Multiple-environment Self-adaptive Network for Aerial-view Geo-localization. PR 2024. [[Paper]](https://arxiv.org/abs/2204.08381)[[Code]](https://github.com/wtyhub/MuseNet) |
| WeatherPrompt | 62.52 | 63.26 | 80.73 | 66.12 | Jiahao Wen, Hang Yu, Zhedong Zheng. WeatherPrompt: Multi-modality Representation Learning for All-Weather Drone Visual Geo-Localization. NeurIPS 2025. [[Paper]](https://arxiv.org/abs/2508.09560)[[Code]](https://github.com/Jahawn-Wen/WeatherPrompt) |


### CVUSA Dataset

| Methods | R@1 | AP | Reference |
| ------- | --- | -- | --------- |
| | Street -> Satellite | | |
| Zhengetal. | 59.30 | 63.52 | Zhedong Zheng, Yunchao Wei, Yi Yang. University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization. MM 2020. [[Paper]](https://dl.acm.org/doi/abs/10.1145/3394171.3413896)[[Code]](https://github.com/layumi/University1652-Baseline) |
| IBN-Net | 73.49 | 76.66 | Xingang Pan, Ping Luo, Jianping Shi, Xiaoou Tang. Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net. ECCV 2018. [[Paper]](https://openaccess.thecvf.com/content_ECCV_2018/html/Xingang_Pan_Two_at_Once_ECCV_2018_paper.html)[[Code]](https://github.com/Asthestarsfalll/IBNNet-MegEngine) |
| MuSe-Net | 75.00 | 78.04 | Tingyu Wang, Zhedong Zheng, Yaoqi Sun, Chenggang Yan, Yi Yang, Tat-Seng Chua. Multiple-environment Self-adaptive Network for Aerial-view Geo-localization. PR 2024. [[Paper]](https://arxiv.org/abs/2204.08381)[[Code]](https://github.com/wtyhub/MuseNet) |




