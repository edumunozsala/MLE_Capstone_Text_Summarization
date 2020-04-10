# Capstone Project Proposal about Text Summarization using Machine Learning

## Domain background

In these times with the rise of information technologies, globalization and Internet, an enormous amount of information is created daily, including a large volume of written texts. The International Data Corporation (IDC) projects that the total amount of digital data circulating annually around the world would sprout from 4.4 zettabytes in 2013 to hit 180 zettabytes in 2025. Dealing with such a huge amount of data is a challenging problem where automatization techiniques can help many industries and businesses. Without summaries it would be practically impossible for human beings to get access to the ever growing mass of information available online. Hundreds of news are. Hundreds of news are published around the world in a few hours and condensing them to make them available to the public is a manual and very expensive task, in time and money. So the development of automatic techniques to get short, concise and understandable summaries would be of great help to many global companies and organizations. Furthermore, applying text summarization reduces reading time, accelerates the process of researching for information, and increases the amount of information that can fit in an area.

But this is not a recent subject, for decades studies and projects have been carried out in this field in order to achieve automatic text summarization systems. For example, Luhn in 1958 [2] or the DimSum in 1997 [3] are two examples of papers where Natural Language Processing techniques are applied to text summarization. In the 80s and 90s the classical NLP techniques where the foundations of the automatic text summarization systems but recently the advances in deep learning algorithms has become a new source of inspiration to face and solve this problem.

Text Summarization is a challenging problem these days and it can be defined as a technique of shortening a long piece of text to create a coherent and fluent summary having only the main points in the document. But, what is a summary? It is a *text that is produced from one or more texts, that contains a significant portion of the information in the original text(s), and
that is no longer than half of the original text(s)* [1].

## Problem description and statement
Text Summarization is a challenging problem these days and it can be defined as a technique of shortening a long piece of text to create a coherent and fluent summary having only the main points in the document. But, what is a summary? It is a *text
that is produced from one or more texts, that contains a significant portion of the information in the original text(s), and
that is no longer than half of the original text(s)* [1]. So given a long, multisentence document we need to extract the main concepts, ideas or topics in the document and generate a new short text containing the same concepts, ideas or topics.
It is not a well-defined with just one solution problem, there could be many diferent summaries for a given text, so it is not an easy and simple problem. In fact, today is a very active field of investigation and the state-of-the-art solutions are still not so impressive and accurate than we could expect.

Our goal will be to build a machine learning model to produce a summary (about 50-word length) from a source text much more longer, 300-400 words. It will be a single document summarization (there are some techniques to approach a multidocument summarization) and it output could be an *extract* (containing pieces of the source text) or *abstract* (a new text is created). We are looking for a efficient method, not very time comsuming, that can be applied easily in a software solution.
We will use a dataset of pairs of source-summary texts, labeled solution, to train and evaluate the performance using some common metrics and we would try to apply the solution to some well-known dataset to compare this algorithm to other solutions

## Dataset and inputs
When searching for information and data about text summarization I found hard to obtain a "good" dataset. Some of the most popular  data sets are intended for research use, containing hundred of thounsands examples and gigabytes of data that require high computational capacity and days or weeks to train. But we were interested in a dataset that could be trained faster, in a few hours, where we can experiment and develop a !!!!!!.

We will use a dataset from Kaggle, [here is the link](https://www.kaggle.com/sunnysai12345/news-summary), that consists in 4515 examples and *and contains Author_name, Headlines, Url of Article, Short text, Complete Article*. This data was extracted from [Inshorts](https://inshorts.com) and scraped the news article from Hindu, Indian times and Guardian.
There are two sets of data: one containing some extra features like Author, Date and the other one with only two columns, text and headline (or summary). This last one
An example:

**Text:** *"Isha Ghosh, an 81-year-old member of Bharat Scouts and Guides (BSG), has been imparting physical and mental training to schoolchildren in Jharkhand for several decades. Chaibasa-based Ghosh reportedly..."* 

**Summary:** *"81-yr-old woman conducts physical training in J'khand schools"*


### Content

- [Predicting_Mortgage_Approvals_EDA](https://github.com/edumunozsala/Predicting_Mortgage_Approvals/blob/master/Predicting_Mortgage_Approvals_EDA.ipynb)
    Code and visualization in Python to develop a Exporatory Data Analysis of the problem and data. 
## Machine learning model built in Azure Machine Learning Studio
From Microsoft Doc:

*Microsoft Azure Machine Learning Studio (classic) is a collaborative, drag-and-drop tool you can use to build, test, and deploy predictive analytics solutions on your data. Azure Machine Learning Studio (classic) publishes models as web services that can easily be consumed by custom apps or BI tools such as Excel.

Machine Learning Studio (classic) is where data science, predictive analytics, cloud resources, and your data meet.*

In the second part of the capstone I built a predictive model on Azure ML Studio to predict when an mortgage approval would be accepted or not, [here is the link for a description](https://medium.com/analytics-vidhya/predicting-mortgage-approvals-data-analysis-and-prediction-with-azure-ml-studio-part-2-2c190e83c9f4)

#### Links related:
- https://towardsdatascience.com/a-quick-introduction-to-text-summarization-in-machine-learning-3d27ccf18a9f

####
[1] - Hovy, E. H. Automated Text Summarization. In R. Mitkov (ed), The Oxford Handbook of Computational Linguistics, chapter 32, pages 583–598. Oxford University Press, 2005
[2] -  Luhn, H., P. The Automatic Creation of Literature Abstracts. In Inderjeet Mani and Mark Marbury, editors, Advances in Automatic Text Summarization. MIT Press, 1999
[3] - Aonet, C., Okurowskit, M. E., Gorlinskyt, J., et al. A Scalable Summarization System Using Robust NLP. In Proceedings of the ACL’07/EACL’97 Workshop on Intelligent Sclalable Text Summarization, pages 66-73, 1997

## License
This repository is under the GNU General Public License v3.0