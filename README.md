# Machine Learning Project Name

## Background
This is background why we did this project

## Problem
Problem Type (Example: Sequence labelling problem)

## Goal
Example: Achieve 90% on Accuracy, F1, and Precision

## Paper Reference (example)
1. End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF (Xuezhe Ma , Eduard Hovy)
2. Character-level Convolutional Networks for Text Classification (Xiang Zhang , Junbo Zhao , Yann LeCun)
3. Multi-Head-Attention
4. Some trial and error approach

## Timeline Project
Development Cycle

No|Task|Deadline
---|---|---
1|A| 05-01-2020
2|B| 10-02-2020
3|C| 23-02-2020
4|D| 03-03-2020

## Branch

    |-- master
    |-- dev

## Methods
Your methods

No|Embedding|Methods|Accuracy on *val-data*|Accuracy on *test-data* 
---|---|---|---|---
1|TF-IDF|XGBoost| 73% | 83% |
2|CountVectorizer|LSTM| 88 % | 97% |

## Project Trees
    
    |-- config
        |-- Dockerfile
        |-- common_libs.txt
        |-- specific_libs.txt
        |-- jupyter_notebook_config.py
        |-- run_jupyter.sh
    |-- data
    |-- model
    |-- notebook
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
        to build docker image :
            sh build_image.sh
        to run docker container :
            sh run.sh
        to start docker container :
            sh start.sh
        to stop docker container :
            sh stop.sh
