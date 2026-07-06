# [Pattern Recognition] Multiple-environment Self-adaptive Network for Aerial-view Geo-localization  
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) 

<div align="center">
	<img src="docs/dance.jpg" alt="Editor" width="700">
</div>
<div align="center">
	<img src="docs/visual.png" alt="Editor" width="800">
</div>


## MuseNet
[[Paper]](https://arxiv.org/abs/2204.08381) 

## Prerequisites
- Python 3.6
- GPU Memory >= 8G
- Numpy > 1.12.1
- Pytorch 0.3+
- scipy == 1.2.1
- imgaug == 0.4.0

## Getting started
### Dataset & Preparation
Download [University-1652](https://github.com/layumi/University1652-Baseline) upon request. You may use the request [template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).

<!--Download [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark).-->

Download [CVUSA](https://hdueducn-my.sharepoint.com/:u:/g/personal/wongtyu_hdu_edu_cn/EcaV9nPk2NxEgAp4MlV1FH4BplbDSqFMxEqqpwf9ooHshw?e=tIXFsB).

## Train & Evaluation
### Train & Evaluation University-1652
```  
sh run.sh
```
:sparkles:[Download The Trained Model](https://hdueducn-my.sharepoint.com/:u:/g/personal/wongtyu_hdu_edu_cn/EcUM81Bpc35DsgxbbscP9ewB_Qdrtk2TDLc1Z1dLRh0jAg?e=H32yw3)
### Train & Evaluation CVUSA
```  
python prepare_cvusa.py  
sh run_cvusa.sh
```
## Citation

```bibtex
@ARTICLE{wang2024Muse,
  title={Multiple-environment Self-adaptive Network for Aerial-view Geo-localization}, 
  author={Wang, Tingyu and Zheng, Zhedong and Sun, Yaoqi and Yan, Chenggang and Yang, Yi and Tat-Seng Chua},
  journal = {Pattern Recognition},
  volume = {152},
  pages = {110363},
  year = {2024},
  doi = {https://doi.org/10.1016/j.patcog.2024.110363}}
```

```bibtex
@ARTICLE{wang2021LPN,
  title={Each Part Matters: Local Patterns Facilitate Cross-View Geo-Localization}, 
  author={Wang, Tingyu and Zheng, Zhedong and Yan, Chenggang and Zhang, Jiyong and Sun, Yaoqi and Zheng, Bolun and Yang, Yi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  year={2022},
  volume={32},
  number={2},
  pages={867-879},
  doi={10.1109/TCSVT.2021.3061265}}
```
```bibtex
@article{zheng2020university,
  title={University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization},
  author={Zheng, Zhedong and Wei, Yunchao and Yang, Yi},
  journal={ACM Multimedia},
  year={2020}
}
```
## Related Work
## 🔗 Ecosystem

<p align="center"><i>Explore our ecosystem for UAV & Spatial Intelligence 🚁 </i></p >

### 🚁 UAV & Spatial Intelligence

<p align="center"><b>🎓 The University-1652 Family</b></p >

<div align="center">
  <table>
    <tr>
      <td align="center" width="33%">
        <a href=" ">
          <h3>🎓</h3>
          <b>University-1652</b>
        </a >
        <br><sub>Multi-view Multi-source Benchmark<br>Ground · Drone · Satellite · ACM MM'20</sub>
        <br><br>
        <a href="https://github.com/layumi/University1652-Baseline"><img src="https://img.shields.io/github/stars/layumi/University1652-Baseline.svg?style=social&label=Star" alt="GitHub stars"></a>
      </td>
      <td align="center" width="33%">
        <a href="https://github.com/wtyhub/MuseNet">
          <h3>🌦️</h3>
          <b>University-WX</b>
        </a >
        <br><sub>Multi-Weather Extension on the Fly<br>Pattern Recognition'24</sub>
        <br><br>
        <a href="https://github.com/wtyhub/MuseNet"><img src="https://img.shields.io/github/stars/wtyhub/MuseNet.svg?style=social&label=Star" alt="GitHub stars"></a>
      </td>
      <td align="center" width="33%">
        <a href="https://github.com/MultimodalGeo/GeoText-1652">
          <h3>💬</h3>
          <b>GeoText-1652</b>
        </a >
        <br><sub>Dense Text Extension<br>ECCV'24</sub>
        <br><br>
        <a href="https://github.com/MultimodalGeo/GeoText-1652"><img src="https://img.shields.io/github/stars/MultimodalGeo/GeoText-1652.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
    </tr>
  </table>
</div>

<p align="center"><b>🚀 New Open-Source Releases</b></p >

<div align="center">
  <table>
    <tr>
      <td align="center" width="25%">
        <a href="https://github.com/YsongF/GeoFuse">
          <h3>🛰️</h3>
          <b>GeoFuse</b>
        </a >
        <br><sub>Road Maps as Free Geometric Priors </sub>
        <br><br>
        <a href="https://github.com/YsongF/GeoFuse"><img src="https://img.shields.io/github/stars/YsongF/GeoFuse.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
      <td align="center" width="25%">
        <a href="https://github.com/JT-Sun/UAVReason">
          <h3>🧠</h3>
          <b>UAVReason</b>
        </a >
        <br><sub>Aerial Scene Reasoning & Generation Benchmark</sub>
        <br><br>
        <a href="https://github.com/JT-Sun/UAVReason"><img src="https://img.shields.io/github/stars/JT-Sun/UAVReason.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
      <td align="center" width="25%">
        <a href="https://github.com/HaoDot/Video2BEV-Open">
          <h3>🗺️</h3>
          <b>Video2BEV</b>
        </a >
        <br><sub>Drone Video → Bird's-Eye-View</sub>
        <br><br>
        <a href="https://github.com/HaoDot/Video2BEV-Open"><img src="https://img.shields.io/github/stars/HaoDot/Video2BEV-Open.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
      <td align="center" width="25%">
        <a href="https://github.com/YaxuanLi-cn/PairUAV">
          <h3>🚁</h3>
          <b>PairUAV</b>
        </a >
        <br><sub>Paired UAV Data for Matching</sub>
        <br><br>
        <a href="https://github.com/YaxuanLi-cn/PairUAV"><img src="https://img.shields.io/github/stars/YaxuanLi-cn/PairUAV.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
    </tr>
  </table>
</div>

---

<p align="center">
  ⭐ If you find our projects helpful, a <b>star</b> is the best support! ⭐
</p >
