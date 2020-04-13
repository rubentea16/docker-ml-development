# Name Entity for Warung-related Entity

## Background
Next process in the pipeline of ASR to POS, The Sound Recognized should have entity embedded on it. But the ASR model didn't know what type of entity should it resulted. 
Thus, this model will get the entity of the words from ASR so it could resulted as a json.

## Problem
Different with normal NER Task, this work will recognize 3 types of entities.[Nominal, Types, Product]

## Goal
1. From the text input, return its entities
    

## Methods
Your methods

No|Methods|Metrics
---|---|---
1|Word+Chars| 98%

## Installation

    All dependencies that are neeeded to run this software
    
    Example:
    glob
    keras
    keras_contrib
    keras_multi_head
    numpy
    pandas
    random
    sklearn
    spacy
    tqdm
    
## Getting Started

For Creating a new model, Please go to Development notebook. 
Don't forget to save the word2idx, char2idx, and label2idx from the development process, because the result is really depends on them
Model in the model folder is based on Python 3.5