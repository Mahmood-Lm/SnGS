# Beyond the Baseline: Enhanced SoccerNet Game State Reconstruction

[![Paper](https://img.shields.io/badge/paper-PDF-red)](./BTB.pdf)
[![Challenge](https://img.shields.io/badge/SoccerNet-GSR%20Challenge-blue)](https://github.com/SoccerNet/sn-gamestate/tree/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and methodology for our high-performance pipeline for the **SoccerNet Game State Reconstruction (GSR) Challenge**. Our approach systematically enhances the official baseline model, addressing key bottlenecks to achieve a significant boost in accuracy. Our work secured the **second-best performance** among all participants, demonstrating a **120% improvement** over the baseline on the challenge set.

## Abstract

The SoccerNet Game State Reconstruction (GSR) challenge provides a crucial benchmark for analyzing player tracking and identification from broadcast soccer videos. While the official baseline model offers a solid foundation, its performance is hampered by key bottlenecks in complex scenarios. This paper introduces a series of targeted enhancements to create a new, high-performance pipeline. We systematically address the primary weaknesses of the baseline by integrating four key improvements: (1) a fine-tuned YOLOv8 model for robust athlete detection, (2) a fine-tuned CLIP model for accurate jersey number recognition, (3) a novel team clustering method using SIGLIP embeddings, and (4) the "No Bells, Just Whistles" method for precise pitch localization. Evaluated on the SoccerNet-GSR dataset, our proposed pipeline demonstrates a remarkable performance increase, achieving a GS-HOTA score of 51.52 on the challenge set—a 120% improvement over the baseline.

---

## 🚀 Key Features & Enhancements

Our pipeline replaces critical components of the SoccerNet-GSR baseline to achieve state-of-the-art results.

1.  **⚽ Robust Athlete Detection:**
    * Replaced the standard YOLOv8x with a model **fine-tuned on the SoccerNet v3 H250 dataset**.
    * This significantly reduces false positives (e.g., detecting fans) and false negatives (e.g., missing occluded players).

2.  **🔢 Accurate Jersey Number Recognition:**
    * Replaced the baseline's MMOCR with a **fine-tuned CLIP (ViT-L-14) model**.
    * Overcomes failures of traditional OCR on low-resolution text and varied fonts by leveraging powerful vision-language representations.
    * Includes a data cleaning step to correct label noise in the original dataset.

3.  **👕 Superior Team Clustering:**
    * Replaced K-means clustering on ReID embeddings with a more robust method using pre-trained **SIGLIP embeddings**.
    * Accurately separates teams even when uniforms are visually similar.

4.  **🗺️ Precise Pitch Localization:**
    * Replaced the baseline's TVCalib with the state-of-the-art **"No Bells, Just Whistles" (NBJW)** method.
    * Achieves highly accurate camera calibration and player projection, especially in challenging center-pitch camera views.

## 📊 Performance

Our enhanced pipeline demonstrates a substantial improvement over the baseline across all official SoccerNet-GSR datasets. The core evaluation metric is **GS-HOTA**, which measures both localization and identification accuracy.

### Challenge Set Results

Our method achieved the **2nd rank** on the official challenge leaderboard.

| Rank | Participant team   | GS-HOTA(↑) | GS-DetA(↑) | GS-AssA(↑) |
|:----:|:-------------------|:-----------|:-----------|:-----------|
| 1    | Constructor tech   | 63.81      | 49.52      | 82.23      |
| **2**| **Ours** | **51.52** | **37.35** | **71.07** |
| 3    | UPCxMobius         | 43.15      | 30.46      | 61.16      |
| 12   | *Baseline* | *23.36* | *9.80* | *55.69* |

### Performance Summary

| Dataset   | Baseline GS-HOTA | Our GS-HOTA | Improvement |
|:----------|:-----------------|:------------|:------------|
| Validation| 18.05            | **37.10** | **+105%** |
| Test      | 29.53            | **50.11** | **+69%** |
| Challenge | 23.36            | **51.52** | **+120%** |

---
