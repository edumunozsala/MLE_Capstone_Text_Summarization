
# Capstone Project about Text Summarization using Machine Learning techniques


This folder of the repository contains the notebooks we use to analyze the data and to preprocess and clean the data:
- Text_summarization_EDA
- data_preprocess

## Requirements

To run the notebooks you only need the most common libraries in python like pandas, numpy, matplotlib, nltk, sklearn,...

But there are some packages that you may need to download and install using the pip command. At the begining of the notebook you can install those packages just running a cell with the commands:

!pip install wordcloud
!pip install gensim
!pip install pyLDAvis
(You should uncomment this lines)

## Exploratory Data Analysis

In the notebook *Text_summarization_EDA.ipynb* we try to discover some data distributions and insights from data to better understand how to deal with the text data.

Statistical Count Features from headlines and text are calculated:

- Sentence Count 
- Word Count 
- Character Count 
- Sentence density 
- Word Density 
- Punctuation Count 
- Stopwords Count 
- Unknown words

We also apply part-of-the-speech tagging or POS tagging and visualize their distribution. And we show the most frequent words using a Word Cloud.

Finally we use the Latent Dirichlet Allocation (LDA) for topic modeling.

## Data preprocess

We apply some of the well-known techniques to handle and clean text data in the notebook *data_preprocess.ipynb*. Some of them are listed below:

- Remove URLs
- Remove html tags
- Remove some emojis
- Expand common contractions
- Expand some Slang abbrevation
- Remove punctuation
- Remove non-character (Unicode \xFF)
- Remove break line \n
- Remove &amp
- Remove mention @
- Remove hastag #
- Lowercase
- Remove very short tokens
- Remove Stopwors
- Steeming and Lemmatization

In the last section of the notebook we create two dataset, the training and the validation dataset.

## Dataset
The project is intended to use a **Kaggle dataset called News Summary**, [click this link to access it](https://www.kaggle.com/sunnysai12345/news-summary). The datafiles are also included in the **data** directory in this repository.

The dataset consists in 4515 examples of news and their summaries and some extra data like Author_name, Headlines, Url of Article, Short text, Complete Article. This data was extracted from Inshorts, scraping the news article from Hindu, Indian times and Guardian.
An example:
• Text: "Isha Ghosh, an 81-year-old member of Bharat Scouts and Guides (BSG), has been imparting physical and mental training to schoolchildren ..."
• Summary: "81-yr-old woman conducts physical training in J'khand schools" 

This dataset also include a version with shorter news and summaries, about 98,000 news. They will provide us training and validation data for our abstractive model.


## License
This repository is under the GNU General Public License v3.0.
