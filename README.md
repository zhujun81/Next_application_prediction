# Introduction

This repository contains code for our paper Improving Next-Application Prediction with Deep Personalized-Attention Neural Network accepted by 20th IEEE International Conference on Machine Learning and Applications (ICMLA). Our implementation is based on https://github.com/gabrielspmoreira/chameleon_recsys


# Requirements

Python packages required:
- Python 3.7
- Tensorflow 1.15.0

# Dataset
## CareerBuilder12 (CB12) 
- Raw data downloaded from (https://www.kaggle.com/competitions/job-recommendation/data) 

## Data preparation
- You can follow the steps described in ```1.Prepare_data```, ```2.Prepare_content``` and ```3.Prepare_session``` to generate training/test sessions.
- Or you can used directly our preprocessed data (```job_14d_30_metadata_and_embeddings_d2v``` can be downloaded from https://drive.google.com/file/d/1jgnT042Rqd0V1RpQA6Y0dJwc3HoLFQk3/view?usp=sharing)


# Reference
Please cite our paper if you use our data or code:

```
@inproceedings{zhu2021improving,
  title={Improving next-application prediction with deep personalized-attention neural network},
  author={Zhu, Jun and Viaud, Gautier and Hudelot, C{\'e}line},
  booktitle={2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA)},
  pages={1615--1622},
  year={2021},
  organization={IEEE}
}

