
# Capstone Project about Text Summarization using Machine Learning techniques

This folder of the repository contains the notebook where we implement our benchmark model using the summarizer module on the gensim library:

- text_summarization_gensim.ipynb

## Requirements

To run the notebook you need the most common libraries in python: pandas, numpy, random, os, matplotlib, seaborn, and sklearn.

You also need the package gensim which is very commom in NLP tasks. You can download and install, runing: pip install gensim. There is a cell in the code where you can run this command.

!pip install gensim
(You should uncomment this line)

You need the package rouge that you can download and install using the pip command. At the begining of the notebook you can install that packag just running a cell with the commands:

!pip install rouge
(You should uncomment thi line)

## How to run it
The notebook is very simple and easy to execute. There are some commented cells to install the libraries gensim and rouge, that you might need to uncomment the first time you run the notebook.
There is also a cell to download the data from the cloud storage, this code is commented and you should not need to execute it.

In the cell where some global variables are defined, **you must review and modify depending on where the datafiles are stored** and what is the current directory. Those variables are:

data_path='MLE_Capstone_Text_Summarization/data'

traindata_file= 'cl_train_news_summary.csv'

validdata_file= 'cl_valid_news_summary.csv'

The output files will be stored in the output path variable, if you want to store the files in a different folder, please modify this lineas:

output_path = 'MLE_Capstone_Text_Summarization/data'
train_results='gensim_output_'+traindata_file
valid_results='gensim_output_'+validdata_file

Those are all the changes to apply to run the notebook successfully.

If you want to try with a different length of the summary you can change the variable summary length, but you must be careful with the values ​​that you assign to this variable.

## Data

The files containing the data are included in the data directory of this repository.

This notebook was developed in a notebook on Google AI Platform and the data files where stored on Google Cloud Storage. That is the reason for a cell where gsutil command is used to download the data from GCS to local disk. 

We have cleaned and processed the data previously using the techniques included in the notebook in the folder data_analysis.

You can download our cleaned dataset in a Kaggle public dataset called [Cleaned News Summary](https://www.kaggle.com/edumunozsala/cleaned-news-summary).
You can also download the Glove embeddings from Kaggle in the folowing dataset [GloVe: Global Vectors for Word Representation](https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation), glove.6B.100d.txt.


## Gensin Summarizer

This is a simple method to tackle the problem of text summarization, it is an extractive approach which select sentences from the corpus that best represent it and arrange them to form a summary. These techniques are very popular in the industry as they are very easy to implement. They use existing natural language phrases and are reasonably accurate. And they are very fast since they are an unsupervised algorithm, so they do not have to calculate loss function in every step

This notebook load the dataset and simply calls the summarize function in gensim library to get the predicted summary from the source text. And finally evaluate the results using the rouge metric.

## License
This repository is under the GNU General Public License v3.0.
