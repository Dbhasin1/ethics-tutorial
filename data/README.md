### Data used for building the [NLP from scratch tutorial](https://github.com/Dbhasin1/numpy-tutorials/blob/ethics-tutorial/content/tutorial-nlp-from-scratch.md)
---
#### [IMDb Reviews Dataset](https://github.com/Dbhasin1/ethics-tutorial/blob/lstm-update/data/IMDB%20Dataset.csv)

**Purpose**: Training the Deep Learning model

> Information courtesy of
IMDb
(http://www.imdb.com).
Used with permission.

IMDB Reviews Dataset is a large movie review dataset collected and prepared by Andrew L. Maas from the popular movie rating service, IMDB. The IMDB Reviews dataset is used for binary sentiment classification, whether a review is positive or negative. It contains 25,000 movie reviews for training and 25,000 for testing. All these 50,000 reviews are labeled data that may be used for supervised deep learning. To make things a bit more comprehensible we're using the `pandas` dataframe version downloaded from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

#### [Glove Embeddings](https://nlp.stanford.edu/projects/glove/)
**Purpose**: To represent text data in machine-readable i.e numeric format
> Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)

GloVe is an unsupervised algorithm developed for generating word embeddings by generating global word-word co-occurence matrix from a corpus. You can download the zipped files containing the embeddings from https://nlp.stanford.edu/projects/glove/. 
Here you can choose any of the four options for different sizes or training datasets, we opted for the least resource-heavy file with 50 dimensional representations for each word. 

---

#### [Speech Dataset](https://github.com/Dbhasin1/ethics-tutorial/blob/lstm-update/data/speeches.csv)
**Purpose**: The trained Deep Learning Model will perform sentiment analysis on this data 
> Curated by the authors of the tutorial 

We have chosen speeches by activists around the globe talking about issues like climate change, feminism, lgbtqa+ rights and racism. These were sourced from newspapers, the official website of the United Nations and the archives of established universities as cited in the table below. A CSV file was created containing the transcribed speeches, their speaker and the source the speeches were obtained from. 
We made sure to include different demographics in our data and included a range of different topics, most of which focus on social and/or ethical issues. The dataset is subjected to the CC0 Creative Common License, which means that is free for the public to use and there are no copyrights reserved.

| Speech                                           | Speaker                 | Source                                                     |
|--------------------------------------------------|-------------------------|------------------------------------------------------------|
| Barnard College Commencement                     | Leymah Gbowee           | Barnard College - official website                         |
| UN Speech on youth Education                     | Malala Yousafzai        | The Guardian                                               |
| Remarks in the UNGA on racial discrimination     | Linda Thomas Greenfield | United States mission to the United Nation                 |
| How Dare You                                     | Greta Thunberg          | NBC - official website                                     |
| The speech that silenced the world for 5 minutes | Severn Suzuki           | Earth Charter                                               |
| The Hope Speech                                  | Harvey Milk             | University of Maryland - official website                  |
| Speech at the time to Thrive Conference          | Ellen Page              | Huffpost                                                   |
| I have a dream                                   | Martin Luther King      | Marshall University - official website                     |
