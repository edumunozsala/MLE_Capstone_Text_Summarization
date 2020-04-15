
# Capstone Project Proposal about Text Summarization using Machine Learning techniques

## Proposal
The capstone proposal is written in the file name *proposal.pdf* following the rubric defined in the nanodegree program. It contains the sections:
- Domain background
- Problem description and statement
- Dataset and inputs
- Solution Statement
- Benchmark Model
- Evaluation Metric
- Project Design
-  Links

The same content is included in the proposal.md file but with not formatted text.

## Dataset
The project is intended to use a **Kaggle dataset called News Summary**, [click this link to access it]([https://www.kaggle.com/sunnysai12345/news-summary](https://www.kaggle.com/sunnysai12345/news-summary))
The datafiles are also included in the data directory in this repository.

The dataset consists in 4515 examples of news and their summaries and some extra data like Author_name, Headlines, Url of Article, Short text, Complete Article. This data was extracted from Inshorts, scraping the news article from Hindu, Indian times and Guardian.
An example:
• Text: "Isha Ghosh, an 81-year-old member of Bharat Scouts and Guides (BSG), has been imparting physical and mental training to schoolchildren ..."
• Summary: "81-yr-old woman conducts physical training in J'khand schools" 

For a better performance, we should consider using a greater dataset like the CNN/Daily Mail dataset. The DeepMind Q&A Dataset is a large collection of news articles from CNN and the Daily Mail with associated questions. The dataset was developed as a question and answering task for deep learning and was presented in the 2015 paper “[Teaching Machines to Read and Comprehend](https://arxiv.org/abs/1506.03340).”
If it is necessary, we will work with the CNN dataset, specifically the download of the ASCII text of the news stories available here:
-   [cnn_stories.tgz](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ)  (151 Megabytes)

This dataset contains more than 93,000 news articles where each article is stored in a single “_.story_” file.
There is a great description on how to download, clean and transform this dataset to be consumed by a model in this [link]([https://machinelearningmastery.com/prepare-news-articles-text-summarization/](https://machinelearningmastery.com/prepare-news-articles-text-summarization/))

## License
This repository is under the GNU General Public License v3.0.
