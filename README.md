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

## Timeline Project
Development Cycle

No|Task|Deadline
---|---|---
1|data preparation| 01-01-2019
2|cleansing| 10-01-2019
3|preprocessing| 20-01-2019
4|modeling| 30-01-2019
5|reports & documentation| 10-02-2019
    
## Branch

    |-- master

## Methods
Your methods

No|Methods|Metrics
---|---|---
1|Method-1| 90%
2|Method-1| 90%
3|Method-1| 90%

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
        - common_libs.txt
        - specific_libs.txt
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
