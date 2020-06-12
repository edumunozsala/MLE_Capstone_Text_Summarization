
# Capstone Project about Text Summarization using Machine Learning techniques


This folder of the repository contains the notebook where we implement the extractive text summarization model based on sentence embeddings:
- sentences_clustering_summarizer

## Requirements

To run the notebook you need the most common libraries in python: pandas, numpy, random, os, re, matplotlib, seaborn, nltk and sklearn.

But you also need the package rouge that you can download and install using the pip command. At the begining of the notebook you can install that packag just running a cell with the commands:

!pip install rouge
(You should uncomment this lines)

## How to run it

The notebook is easy to execute and follow. There are some commented cells to install the library rouge, that you might need to uncomment the first time you run the notebook.
There is also a cell to download the data and the embeddings vectors from the cloud storage, this code is commented and you should not need to execute it.

In the cell where some global variables are defined, **you must review and modify depending on where the datafiles are stored** and what is the current directory. Those variables are:

data_path='MLE_Capstone_Text_Summarization/data'

traindata_file= 'cl_train_news_summary.csv'

validdata_file= 'cl_valid_news_summary.csv'

glove_file = 'glove.6B.100d.txt'

The output files will be stored in the output path variable, if you want to store the files in a different folder, please modify this lineas:

output_path = 'MLE_Capstone_Text_Summarization/data'
output_train_file='gensim_output_'+traindata_file
output_valid_file='gensim_output_'+validdata_file

Those are all the changes to apply to run the notebook successfully.

If you want to try with a different length of the summary you can change the variable k in the cell where the main algorithm is executed, but you must be careful with the values ​​that you assign to this variable.

**The model takes about 3-5 minutes** to predict the whole dataset, including training and validation. 

## Data

The files containing the data are included in the data directory of this repository.

This notebook was developed in a notebook on Google AI Platform and the data files where stored on Google Cloud Storage. That is the reason for a cell where gsutil command is used to download the data from GCS to local disk. 

We have cleaned and processed the data previously using the techniques included in the notebook in the folder data_analysis.

You can download our cleaned dataset in a Kaggle public dataset called [Cleaned News Summary](https://www.kaggle.com/edumunozsala/cleaned-news-summary).
You can also download the Glove embeddings from Kaggle in the folowing dataset [GloVe: Global Vectors for Word Representation](https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation), glove.6B.100d.txt.

## Sentence Embeddings

This notebook implements and describes a solution to the text summarization problem, using an extractive approach. We develop a model based on clustering of sentences previously embedded. We create k clusters of sentences, meaning that all the sentences in a cluster are semantically similar, and then for every cluster we select the sentence closest to its centroid.

The process can be described as:

- Data preprocess and cleaning.
- Sentence embedding using pretrained vectors

Word embeddings are a type of word representation that allows words with similar meaning to have a similar representation. Word embeddings are in fact a class of techniques where individual words are represented as real-valued vectors in a predefined vector space. Each word is mapped to one vector and the vector values are learned based on the usage of words. This allows words that are used in similar ways to result in having similar representations, naturally capturing their meaning. A well-trained set of word vectors will place similar words close to each other in that space. The words oak, elm and birch might cluster in one corner, while war, conflict and strife huddle together in another.

GloVe, algorithm is an extension to the word2vec method for efficiently learning word vectors, is an unsupervised learning algorithm that constructs an explicit word-context or word co-occurrence matrix using statistics across the whole text corpus.

- Sentence clustering Once the sentences are embedded we apply a K-Means clustering method to group the sentences in k clusters. Each cluster of sentence embeddings can be interpreted as a set of semantically similar sentences whose meaning can be expressed by just one candidate sentence in the summary.

- Select the most representative sentence of each cluster The candidate sentence chosen in every cluster is the one whose vector representation is closest to the cluster center.

- Ordering the sentences to form the summary Candidate sentences corresponding to each cluster are then ordered to form a summary. The order of the candidate sentences in the summary is determined by the positions of the sentences in their corresponding clusters in the original document


## License
This repository is under the GNU General Public License v3.0.
