# Dissimilarity-Attention-Module-for-Weakly-supervised-Video-Anomaly Detection
Video anomaly detection under weak supervision is complicated due to the difficulties in identifying the anomaly and normal instances during training, hence, resulting in non-optimal margin of separation. In this paper, we propose a framework consisting of Dissimilarity Attention Module (DAM) to discriminate the anomaly instances from normal ones both at feature level and score level. In order to decide instances to be normal or anomaly, DAM takes local spatio-temporal (i.e. clips within a video) dissimilarities into account rather than the global temporal context of a video. This allows the framework to detect anomalies in  real-time (i.e. online) scenarios without the need of extra window buffer time. Further more, we adopt two-variants of DAM for learning the dissimilarity between the successive video clips. The proposed framework along with DAM is validated on two large scale anomaly detection datasets i.e. UCF-Crime and ShanghaiTech, outperforming the online state-of-the-art approaches by 1.5% and 3.4% respectively.

## Proposed Framework
![Prposed Framework](https://github.com/snehashismajhi/Dissimilarity-Attention-Module-for-Weakly-supervised-Video-AnomalyDetection/blob/main/AVSS21%20Framework.jpg)

## DAM: Dissimilarity Attention Module
![DAM](https://github.com/snehashismajhi/Dissimilarity-Attention-Module-for-Weakly-supervised-Video-AnomalyDetection/blob/main/Dissimilarity%20Attention%20Module.jpg)

## State-of-the-art Performance Comparision on UCF-Crime Dataset
![SOTA](https://github.com/snehashismajhi/DAM-Anomaly-Detection/blob/main/state_of_the_art_avss_modified.png)

## Citing DAM
```
@inproceedings{9663810,
      author={Majhi, Snehashis and Das, Srijan and Brémond, François},  
      booktitle={2021 17th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)},
      title={DAM: Dissimilarity Attention Module for Weakly-supervised Video Anomaly Detection},
      year={2021},  volume={},  number={},
      pages={1-8},
      doi={10.1109/AVSS52988.2021.9663810}}
```
