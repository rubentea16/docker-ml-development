# Name Entity Recognition for Warung-Order

## Background
Next process in the pipeline of ASR to POS, The Sound Recognized should have entity embedded on it. But the ASR model didn't know what type of entity should it resulted. 
Thus, this model will get the entity of the words from ASR.

## Problem
Sequence labelling problem

## Goal
Achieve 90% on Accuracy per Word

## Paper Reference
1. End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF (Xuezhe Ma , Eduard Hovy)
2. Character-level Convolutional Networks for Text Classification (Xiang Zhang , Junbo Zhao , Yann LeCun)
3. Multi-Head-Attention
4. Some trial and error approach

## Timeline Project
Development Cycle

No|Task|Deadline
---|---|---
1|data preparation (create data train & labelling)| 24-01-2020
2|re-train model (use FastText word embedding)| 30-01-2020
3|evaluate performance on "sent-comb" data| 03-02-2020
4|create new word embedding & evaluate performance of the embedding| 05-02-2020
5|hyperparameter exploration & tuning| 07-02-2020
6|add more train data (balancing data)| 17-02-2020
7|re-train model| 17-02-2020
8|fine tuning| 18-02-2020
9|create api endpoint for web-app| 02-04-2020    
10|split/update description label| 14-04-2020
11|improve ner systems (change some layers, try some params, etc)| 17-04-2020    
## Branch

    |-- master
    |-- old-version

## Methods
Your methods -- (**test-data** is  "sent-comb.txt" as generated text from ASR output from Bentar)

No|Embedding|Methods|Accuracy per **sentence** on *val-data*|Accuracy per **word** on *val-data*|Accuracy per **sentence** on *test-data*|Accuracy per **word** on *test-data* 
---|---|---|---|---|---|---
1|FastText Pre-trained Indonesia|Bi-LSTM-CNN-LSTM-MultiHeadAttention-CRF| 73% | 83% | - | - |
2|GloVe|Bi-LSTM-CNN-LSTM-MultiHeadAttention-CRF| 88 % | 97% | 58.8% | 90.5% |
3|Embedding Deep Learning by trainable|Bi-LSTM-CNN-LSTM-MultiHeadAttention-CRF| 86.5%| 95.2% | 59.4% | 89% |
4|GloVe|Bi-LSTM-CNN(modified)-MultiHeadAttention(head_num modified)-CRF| 89.6%| 97.6% | 60.8% | 91.4% |
4|GloVe|Bi-LSTM-CNN(modified)-MultiHeadAttention(head_num modified)-CRF **char-embed-dim**=30 **head-num**=16| 96.5%| 99% | 62.5% | 90.46% |

## Project Trees
    
    |-- config
        |-- Dockerfile
        |-- common_libs.txt
        |-- specific_libs.txt
        |-- jupyter_notebook_config.py
        |-- run_jupyter.sh
    |-- data
        |-- pickle_file
        |-- resource
        |-- test
        |-- train
    |-- model
    |-- notebooks
    |-- web-app
    |-- build_image.sh
    |-- container_name.txt
    |-- image_name.txt
    |-- port.txt
    |-- run.sh
    |-- start.sh
    |-- stop.sh
    |-- README.md


## Installation
    
    This software is running inside Docker container
    You can set (image name, container name, port for jupyterlab) :
        - image_name.txt
        - container_name.txt
        - port.txt
    
    In /config directory you will find :
        - Dockerfile
        - common_libs.txt (library you need to install)
        - specific_libs.txt (library you need to install)
        - jupyter notebook config file
    
    
## Getting Started

How to run your project

    - development
        notebooks are in the order of each steps development
        
    - production
        to build:
            sh build_image.sh
        to run:
            sh run.sh
        to start:
            sh start.sh
        to stop:
            sh stop.sh
