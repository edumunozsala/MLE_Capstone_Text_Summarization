
# Capstone Project about Text Summarization using Machine Learning techniques


This folder of the repository contains the notebook where we implement the abstractive text summarization model based on a Encoder-Decoder with the pointer generator mechanism:

- pointer-generator-in-pytorch

## Requirements

To run the notebook you need the most common libraries in python like pandas, numpy, matplotlib, os, io, random, pickle, Counter, typing, copy, nltk.

But you also need PYTORCH version 1.4.0. It is easily installed using pip install: pip install torch

Other packages you need to download and install using the pip command is the rouge package and the pkbar. At the begining of the notebook you can install that packages just running a cell with the commands:

!pip install pkbar

!pip install rouge

(You should uncomment this lines)

## How to run it

**Important**: *this notebook must be executed in the kaggle notebook* because it requires some specific CUDA version and configuration to run properly. We have tried to run it in Google notebooks or Azure virtual machines and the configuration of CUDA components where incompatible and many "weird" errors appear continuously. Kaggle provides GPU resources for free and is easy tu use. **In Kaggle you can access to the code in this [link](https://www.kaggle.com/edumunozsala/pointer-generator-in-pytorch)**

Another point to remark is that **this model takes about 6-8 hours** to train and predict on the defaults parameters and get acceptable results. You must take it into account if you want to test it. 


In the forth cell we define a class, called Parameters where you can set many hyperparameters and parameters of the model to try different options for training. The most relevant are:
- embed_file: Where to find the embeddings file, default *'/kaggle/input/glove6b100dtxt/glove.6B.100d.txt'*
- data_path: Where the input data is located, default *'/kaggle/input/cleaned-news-summary/cl_train_news_summary_more.csv'*
- valid_path: Where the validation data is located, default *'/kaggle/input/cleaned-news-summary/cl_train_news_summary_more.csv'*
- test_path: Where the test data is located, default *'/kaggle/input/cleaned-news-summary/cl_valid_news_summary_more.csv'*
- hidden_size: encoder hidden size, default 100
- dec_hidden_size: decoder hidden size, default 200
- vocab_min_frequency: minimun ocurrencies of a word to be included in the vocabulary.

**Another settings to review** are included in the section *Main Code*, where we create the datasets, there is a parameter, *max_rows* that we can modify to the maximum number of rows to load from the datafiles. To reduce training time it can be modified but **be careful because this value must be a multiple of the batch size**, by default is 16. You can chose values like 16,000, 3,200, etc.  

Finally in the cell where we call the train function **we can set the number of epochs and the learning rate**.

The datasets need it to run the model are uploaded to Kaggle and added to the notebook. The datasets are:
- Cleaned News summary
- GloVe: Global Vectors for Word Representation

## Data

The files containing the data are included in the data directory of this repository.

We said previously that you can find the data on Kaggle, Cleaned News summary and Glove vector. 

We have cleaned and processed the data previously using the techniques included in the notebook in the folder data_analysis.

You can download our cleaned dataset in a Kaggle public dataset called [Cleaned News Summary](https://www.kaggle.com/edumunozsala/cleaned-news-summary).
You can also download the Glove embeddings from Kaggle in the folowing dataset [GloVe: Global Vectors for Word Representation](https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation), glove.6B.100d.txt.

## Pointer Generator model

Based on the paper *“Get To The Point: Summarization with Pointer-Generator Networks”*, Pointer generator networks solve the repetition problem, common in encoder-decoder architectures, by calculating “generating probability” which represents the probability of generating a word from the vocabulary versus copying the word from the source. It is actualy a hybrid network, a combination approach combining both extraction (pointing) and abstraction (generating). 
We calculate an attention distribution and a vocabulary distribution. However, we also calculate the generation probability, which is a scalar value between 0 and 1. This represents the probability of generating a word from the vocabulary, versus copying a word from the source.

The encoder-decoder model with attention comprised main componentes:
-	Encoder: The encoder is responsible for stepping through the input time steps, read the input words one by one and encoding the entire sequence into a fixed length vector called a context vector.
-	Decoder: The decoder is responsible for stepping through the output time steps while reading from the context vector, extracting the words one by one.

## License
This repository is under the GNU General Public License v3.0.
