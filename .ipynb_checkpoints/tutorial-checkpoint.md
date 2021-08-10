---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Sentiment Analysis on notable speeches of the last decade

---

This tutorial demonstrates how to build a simple <a href = 'https://en.wikipedia.org/wiki/Recurrent_neural_network'> Vanilla Recurrent Neural Network </a> from scratch in NumPy to perform sentiment analysis on a socially relevant and ethically acquired dataset

You will first learn how to process textual data and convert it to a numeric format that the machine understands. As a part of the data preprocessing, you will also employ techniques such as text summarisation which will produce a brief representation of the text and finally sentiment analysis will deduce the emotion expressed in the text. 

Your deep learning model - The VanillaRNN is the simplest form of a Recurrent Neural Network will learn to classify a piece of text as positive or negative from the IMDB reviews dataset. The dataset contains 40,000 training and 10,000 test reviews and corresponding labels. Based on the numeric representations of these reviews and their corresponding labels <a href = 'https://en.wikipedia.org/wiki/Supervised_learning'> (supervised learning) </a> the neural network will be trained to learn the sentiment using forward propagation and backpropagaton through time since we are dealing with sequential data here. The output will be a vector of size 2 containing the probabilities for both positive and negative sentiments

+++

![rnn-2.jpg](attachment:rnn-2.jpg)

+++

Today, Deep Learning is getting adopted in everyday life and now it is more important to ensure that decisions that have been taken using AI are not reflecting discriminatory behavior towards a set of populations. It is important to take fairness into consideration while consuming the output from AI. Throughout the tutorial we'll try to question all the steps in our pipeline from an ethics point of view.

+++

## Prerequisites 

---

You are expected to be familiar with the language python, array manipulation with NumPy, linear algebra and Calculus. You should also be familiar with how Neural Networks work.

To refresh your memory you can take the [Python](https://docs.python.org/dev/tutorial/index.html), [Linear algebra on n-dimensional arrays](https://numpy.org/doc/stable/user/tutorial-svd.html) and [Calculus](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/multivariable-calculus.html) tutorials. implemented.

You are advised to read the [Deep learning](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) paper published in 2015 by Yann LeCun, Yoshua Bengio, and Geoffrey Hinton who are regarded as some of the pioneers of the field. You should also consider reading [the d2l.ai book](https://d2l.ai/chapter_recurrent-neural-networks/index.html), which is an interactive deep learning book with multi-framework code, math, and discussions. You can also go through the [Deep learning on MNIST from scratch tutorial](https://numpy.org/numpy-tutorials/content/tutorial-deep-learning-on-mnist.html) to understand how a basic neural network is implemented from scratch.

In addition to NumPy, you will be utilizing the following Python standard modules for data loading and processing:
- [`pandas`](https://pandas.pydata.org/docs/) for handling dataframes 
- [`re`](https://docs.python.org/3/library/re.html) for tokenising textual data 
- [`string`](https://docs.python.org/3/library/string.html) for string operations  
- [`cupy`](https://docs.cupy.dev/en/stable/) for implementing Numpy arrays on Nvidia GPUs by leveraging the CUDA GPU library.

    as well as:
- [Matplotlib](https://matplotlib.org/) for data visualization

This tutorial can be run locally in an isolated environment, such as [Virtualenv](https://virtualenv.pypa.io/en/stable/) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). You can use [Jupyter Notebook or JupyterLab](https://jupyter.org/install) to run each notebook cell. Don't forget to [set up NumPy](https://numpy.org/doc/stable/user/absolute_beginners.html#installing-numpy) and [Matplotlib](https://matplotlib.org/users/installing.html#installing-an-official-release).

+++

## Table of contents

---

1. Collect the data 

2. Preprocess the datasets

3. Build and train a small recurrent neural network from scratch

4. Perform sentiment analysis on collected speeches 

5. Next steps

+++

---

## 1. Collect the datasets

Before we begin there are a few pointers you should always keep in mind before choosing the data you wish to train your model on:
- Unless you have their consent, your data should not link back to named people. It's difficult to obtain a hundred percent anonymous datasets hence it should be made clear to the users that you're minimizing the risk and possible harm of data leaks 
- [Trevisan and Reilly](https://eprints.whiterose.ac.uk/91157/1/Ethical%20dilemmas.pdf) identified a list of sensitive topics that need to be handled with extra care such as :
    - personal daily routines;
    - individual details about impairment and/or medical records;
    - emotional accounts of pain and chronic illness;
    - financial information about income and/or welfare payments;
    - discrimination and abuse episodes;
    - criticism/praise of individual providers of healthcare and support services
    - suicidal thoughts.

- While it can be difficult taking consent from so many people especially on online platforms, the necessity of it depends upon the sensitivity of the topics your data includes and other indicators like whether the platform the data was obtained from allows users to operate under pseudonyms. If the website has a policy that forces the use of a real name, then the users probably need to be asked for consent 

In this section, you will be collecting two different datasets, the IMDB movie reviews dataset and the second contains 10 speeches of activists from different countries around the world, historical and present and with a focus on different topics. The former would be used to train the deep learning model while the latter will be used to perform sentiment analysis on.
   > The IMDb platform allows the usage of their public datasets for personal and non-commercial use. We did our best to ensure that these reviews do not contain any of the aforementioned sensitive topics pertaining to the reviewer.

+++

### Loading the IMDB reviews dataset

Load the data into a pandas dataframe. First check if the data is stored locally; if not, then download it. The dataset can be found on the website by [Stanford AI Lab](http://ai.stanford.edu/~amaas/data/sentiment/).

```{code-cell} ipython3
import pandas as pd

imdb_filepath = 'IMDB Dataset.csv'
imdb_data = pd.read_csv(imdb_filepath)
```

### Collecting and loading the speech transcripts

We have chosen speeches by activists around the globe talking about issues like climate change, feminism, lgbtqa+ rights and racism. These were sourced them from newspapers, the official website of the United Nations and the archives of established universities as cited in the table below. A csv file was created containing the transcribed speeches, their speaker and the source the speeches were obtained from. 

| Speech                                           | Speaker                 | Source                                                     |
|--------------------------------------------------|-------------------------|------------------------------------------------------------|
| Barnard College Commencement                     | Leymah Gbowee           | Barnard College - Columbia University official website     |
| UN Speech on youth Education                     | Malala Yousafzai        | Iowa state university archives                             |
| Remarks in the UNGA on racial discrimination     | Linda Thomas Greenfield | United States mission to the United Nation                 |
| How Dare You                                     | Greta Thunberg          | NBC’s official website                                     |
| The speech that silenced the world for 5 minutes | Severn Suzuki           | NTU blogs                                                  |
| The Hope Speech                                  | Harvey Milk             | University of Maryland archives                            |
| Violence against LGBTQA+                         | Michelle Bachelet       | United Nations office of high commisioner official website |
| I have a dream                                   | Martin Luther King      | Brittanica official website                                |
|                                                  |                         |                                                            |
|                                                  |                         |

```{code-cell} ipython3
speech_filepath = 'speeches.csv'

speech_data = pd.read_csv(speech_filepath)
speech_list = speech_data['speech'].tolist()
```

---

## 2. Preprocess the data

1. Before converting your text into vectors, it is important to clean it and remove all unhelpful parts a.k.a the noise from your data by converting all characters to lowercase, removing html tags, brackets and stop words. Without this cleaning step the dataset is often a cluster of words that the computer doesn't understand.

```{code-cell} ipython3
import re
import string

# converting all characters to lowercase 
imdb_data['review'] = imdb_data['review'].str.lower()

# List of common stopwords taken from https://gist.github.com/sebleier/554280
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", 
             "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
             "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", 
             "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
             "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
             "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", 
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
             "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
             "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
             "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
             "your", "yours", "yourself", "yourselves" ]

def remove_stopwords(data, column):
    data['review without stopwords'] = data[column].apply(lambda x : ' '.join([word for word in x.split() if word not in (stopwords)]))
    return data
    
def remove_tags(string):
    result = re.sub('<.*?>','',string)
    return result

# remove html tags and brackets from text 
data_without_stopwords = remove_stopwords(imdb_data, 'review')
data_without_stopwords['clean_review']= data_without_stopwords['review without stopwords'].apply(lambda cw : remove_tags(cw))
data_without_stopwords['clean_review'] = data_without_stopwords['clean_review'].str.replace('[{}]'.format(string.punctuation), ' ')
```

2. Split the data into training and test sets using the standard notation of x for data and y for labels, calling the training and test set reviews X_train and X_test, and the labels y_train and y_test:

```{code-cell} ipython3
import numpy as np

reviews_list = data_without_stopwords['clean_review'].to_numpy()
sentiment = data_without_stopwords['sentiment'].to_numpy()
    
def shuffle_split_data(X, y, split_percentile):
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, split_percentile)
    X_train = X[split]
    y_train = y[split]
    X_test =  X[~split]
    y_test = y[~split]

    return X_train, y_train, X_test, y_test

# map sentiments to integer values 
y = np.array(list(map(lambda x: 1 if x=="positive" else 0, sentiment)))

# obtaing training and testing data 
X_train, Y_train,X_test, Y_test = shuffle_split_data(reviews_list, y, split_percentile=30)
```

3. So far the text we have is in its raw form, it needs to be broken apart into chunks called tokens because the most common way of processing language happens at the token level. This process of separating a piece of text into smaller units is called Tokenisation. The tokens obtained are then used to build a vocabulary. Vocabulary refers to a set of all tokens in the corpus along with a unique index allotted to each of them.

```{code-cell} ipython3
class Vocabulary:
  def __init__(self):
    PAD_token = 0   # Used for padding short sentences
    SOS_token = 1   # Start-of-sentence token
    EOS_token = 2   # End-of-sentence token
    self.word2index = {}
    self.word2count = {}
    self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
    self.num_words = 0

  # First entry of word into vocabulary
  # Word exists; increase word count
  def add_word(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.num_words
      self.word2count[word] = 1
      self.index2word[self.num_words] = word
      self.num_words += 1
    else:
      self.word2count[word] += 1

  # Find word corresponding to given index
  def to_word(self, index):
    return self.index2word[index]

  # Find word corresponding to given word
  def to_index(self, word): 
    return self.word2index[word]

# split a piece of text into unique words/tokens 
def word_tokeniser (text):    
    tokens = re.split(r"([-\s.,;!?])+", text)
    words = [x for x in tokens if (x not in '- \t\n.,;!?\\' and '\\' not in x)]
    return words

# build vocabulary for the imdb dataset
imdb_voc = Vocabulary()
for review in X_train:
    review_tokens = word_tokeniser(review)
    for token in review_tokens:
        imdb_voc.add_word(token)
 
# build vocabulary for the speech dataset 
speech_voc = Vocabulary()
for speech in speech_list:
    speech_tokens = word_tokeniser(speech)
    for token in speech_tokens:
        speech_voc.add_word(token)
        
```

4. A word embedding is a learned representation for text where words that have the same meaning have a similar representation. Individual words are represented as real-valued vectors in a predefined vector space. GloVe is n unsupervised algorithm developed by Stanford for generating word embeddings by generating global word-word co-occurence matrix from a corpus. You can download the zipped files containing the embeddings from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/). Here you can choose any of the four options for different sizes or training datasets
 >The GloVe word embeddings include sets that were trained on billions of tokens, some up to 840 billion tokens. These algorithms exhibit stereotypical biases, such as gender bias which can be traced back to the original training data. For example certain occupations seem to be more biased towards a particular gender, reinforcing problematic stereotypes. The nearest solution to this problem are some de-biasing algorithms as the one presented in https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6835575.pdf which one can use on embeddings of their choice to mitigate bias, if present.

```{code-cell} ipython3
# Creates a dictionary mapping each
# word to its corresponding embedding 
def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")
    return gloveModel

word_to_vec_map = loadGloveModel('glove.6B.300d.txt')
```

5. Now you'll build separate matrices for the imdb dataset and the speech dataset where each row corresponds to a unique word and maps its word embedding to the unique index it was allotted in the vocabulary.

```{code-cell} ipython3
# Create a matrix where each row contains 
# the embedding of the word having index number 
# equal to the row number 
def emb_matrix (word_to_vec_map, vocabulary):
    words_to_index = vocabulary.word2index
    vocab_len = len(words_to_index)
    embed_vector_len = word_to_vec_map['moon'].shape[0]
    emb_matrix = np.zeros((vocab_len, embed_vector_len))
    for word, index in words_to_index.items():
        embedding_vector = word_to_vec_map.get(word)
        if embedding_vector is not None:
            emb_matrix[index, :] = embedding_vector
        else:
            emb_matrix[index, :] = np.random.rand(embed_vector_len,)
    return emb_matrix 

imdb_emb_matrix = emb_matrix(word_to_vec_map, imdb_voc)
speech_emb_matrix = emb_matrix(word_to_vec_map, speech_voc)
```

6. Since our mapping defined in the function above is between word indexes and embeddings, we will have to transform the input to our neural network and replace each word in it by its corresponding word index

```{code-cell} ipython3
# Given word indices, we create their 
# corresponding word embedding matrix 
def fetch_embeddings (X_indices, emb_matrix):

    text_emb = np.zeros((len(X_indices), emb_matrix[0].shape[0]))
    for index, i in enumerate(X_indices):
       
        x = emb_matrix[int(i)]
        text_emb[index, :] = x

    return text_emb

# Replace word in training and testing data 
# with their indices which are mapped to the word embeddings 
def transform_input (text_input, maxlen, vocabulary):
    text_input_indices = np.zeros((len(text_input),maxlen))
    indexes = []
    for index, text in enumerate(text_input):
        text_indices = []
        for word in text.split(' '):
            if word in vocabulary.word2index:
                text_indices.append(vocabulary.to_index(word))
        text_len = len(text_indices)
        if text_len<maxlen:
            for i in range(maxlen-text_len):
                text_indices.append(0)
        text_indices = np.array(text_indices)
        text_input_indices[index, :] = text_indices

    return text_input_indices

imdb_maxlen = max([len(i) for i in reviews_list])
X_train_indices = transform_input(X_train, imdb_maxlen, imdb_voc)
X_test_indices = transform_input(X_test, imdb_maxlen, imdb_voc)
    
speech_maxlen = max([len(i) for i in speech_list])
speech_indices = transform_input(speech_list, speech_maxlen, speech_voc)
```

## 3. Build the Deep Learning Model

+++

It’s time to start implementing our RNN! You will have to first familiarize yourself with some high-level concepts of the basic building blocks of a deep learning model. You can refer to the [Deep learning on MNIST from scratch tutorial](https://numpy.org/numpy-tutorials/content/tutorial-deep-learning-on-mnist.html) for the same. 

You will then learn how a Recurrent Neural Network differs from a plain Neural Network and what makes it so suitable for processing sequential data. Afterwards, you will construct the building blocks of a simple deep learning model in Python and NumPy and train it to learn to classify the sentiment of a piece of text as positive or negative with a certain level of accuracy

### Introduction to a Recurrent Neural Network

In a plain neural network, the information only moves in one direction — from the input layer, through the hidden layers, to the output layer. The information moves straight through the network and never takes the previous nodes into account at a later stage. Because it only considers the current input, it has no notion of order in time. It simply can’t remember anything about what happened in the past except its training.

In a RNN the information cycles through a loop. When it makes a decision, it considers the current input and also what it has learned from the inputs it received previously. The same can be illustrated in the diagram below: 

![Screenshot%202021-08-02%20at%202.14.18%20PM.png](attachment:Screenshot%202021-08-02%20at%202.14.18%20PM.png)

The hidden layer represented by the color green is a representation of the previous inputs and keeps on getting updated as we move ahead in the sequence. Let's go over the model architecture and the training summary in more detail in the section below.

+++

### Model Architecture and Training Summary

- _The input layer_:

    It is the input for the network — the embeddings of the previously preprocessed text data that are loaded from `X_train_indices` into `layer_0`.

- _The hidden (middle) layer_:

    `layer_1` takes the output from the layer_0 and the previous layer_1 and performs summation on the matrix-multiplication of the input by weights (`Wxh`), the pervious layer_1 by weights (`Whh`) and a bias (`bh`)).

    Then, this output is passed through the tanh activation function for non-linearity.
    This process is repeated until the end of the sequence is reached. 
    
- _The output (last) layer_:

    `layer_2` ingests the output from `layer_1` and repeats the same "dot multiply" process with weights (`Why`).

    `layer_2` is passed through a softmax function to obtain 2 scores for each of the positive-negative sentiment labels. The network model ends with a size 2 layer — a 2-dimensional vector.

- _Forward propagation, backpropagation, training loop_:

    In the beginning of model training, your network randomly initializes the weights and feeds the input data forward through the hidden and output layers. This process is the forward pass or forward propagation.

    Then, the network propagates the "signal" from the loss function back through the hidden layer and adjusts the weights values with the help of the learning rate parameter (more on that later).

> **Note:** In more technical terms, you:
>
>    1. Measure the error by comparing the actual sentiment of the review (the truth) with the prediction of the model.
>    2. Differentiate the loss function.
>    3. Ingest the [gradients](https://en.wikipedia.org/wiki/Gradient) with the respect to the output, and backpropagate them with the respect to the inputs through the layer(s).
>
>    Since the network contains tensor operations and weight matrices, backpropagation uses the [chain rule](https://en.wikipedia.org/wiki/Chain_rule).
>
>    With each iteration (epoch) of the neural network training, this forward and backward propagation cycle adjusts the weights, which is reflected in the accuracy and error metrics. As you train the model, your goal is to minimize the error and maximize the accuracy on the training data, where the model learns from, as well as the test data, where you evaluate the model.
> One major difference between

```{code-cell} ipython3
import numpy as cp
from numpy.random import randn
from numpy.random import random

def softmax(xs):
  # Applies the Softmax Function to the output array.
  return cp.exp(xs) / sum(cp.exp(xs))

def ProcessData (input_data, emb_matrix, weights, learn_rate = 0.1,  target_values = None, backprop = True, predict = False):
    
    Whh = parameters["Whh"] 
    Wxh = parameters["Wxh"] 
    Why = parameters["Why"] 
    bh = parameters["bh"] 
    by = parameters["by"] 
    loss = 0
    num_correct = 0
    prob = []
    
    for index,x in enumerate(input_data):

        inputs = fetch_embeddings(x, emb_matrix)

        # Forward
        '''
        Perform a forward pass of the RNN using the given inputs.
        Returns the final output and hidden state.
        - inputs is an array of one hot vectors with shape (input_size, 1).
        '''
        #print(inputs)
        layer_1 = cp.zeros((Whh.shape[0], 1))
        layer_0 = cp.array(inputs)
        print('layer_0:',layer_0)
        last_inputs = layer_0
        last_hs = { 0: layer_1 }


        # Perform each step of the RNN
        for i, x in enumerate(layer_0):
          print('x in layer_0:', x.shape)
          print('layer_1:', layer_1)
          x = x.reshape((x.shape[0], 1))
          print('before tanh:', Wxh @ x + Whh @ layer_1 + bh)
          layer_1 = cp.tanh(Wxh @ x + Whh @ layer_1 + bh)
          last_hs[i + 1] = layer_1

        # Compute the output
        layer_2 = Why @ layer_1 + by
        print('layer_2:', layer_2)
        probs = softmax(layer_2)
        
        if predict:
            #print(prob)
            prob.append(probs)
            continue
        else:
            # Calculate loss / accuracy
            target = target_values[index]
            loss -= cp.log(probs[target])
            num_correct += int(cp.argmax(probs) == target)

        # Build dL/dy
        d_L_d_y = probs
        d_L_d_y[target] -= 1

        if backprop:

            # Backward
            '''
            Perform a backward pass of the RNN.
            - d_y (dL/dy) has shape (output_size, 1).
            - learn_rate is a float.
            '''
            n = len(last_inputs)
            d_y = d_L_d_y

            # Calculate dL/dWhy and dL/dby.
            d_Why = d_y @ last_hs[n].T
            d_by = d_y

            # Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
            d_Whh = cp.zeros(Whh.shape)
            d_Wxh = cp.zeros(Wxh.shape)
            d_bh = cp.zeros(bh.shape)

            # Calculate dL/dh for the last h.
            # dL/dh = dL/dy * dy/dh
            d_h = Why.T @ d_y

            # Backpropagate through time.
            for t in reversed(range(n)):
              #print(t)
              # An intermediate value: dL/dh * (1 - h^2)
              temp = ((1 - last_hs[t + 1] ** 2) * d_h)

              # dL/db = dL/dh * (1 - h^2)
              d_bh += temp

              # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
              #print(temp.shape, last_hs[t].shape)
              d_Whh += temp @ last_hs[t].T

              # dL/dWxh = dL/dh * (1 - h^2) * x
              last = last_inputs[t].reshape(300,1)
              d_Wxh += temp @ last.T

              # Next dL/dh = dL/dh * (1 - h^2) * Whh
              d_h = Whh @ temp

            # Clip to prevent exploding gradients.
            for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
              cp.clip(d, -1, 1, out=d)
            
            # Update parameters 
            Whh -= learn_rate * d_Whh
            Wxh -= learn_rate * d_Wxh
            Why -= learn_rate * d_Why
            bh -= learn_rate * d_bh
            by -= learn_rate * d_by
    if predict:
        return probs
    else:
        return loss / input_data.shape[0], num_correct /  input_data.shape[0], [Whh, Wxh, Why, bh, by]

emb_matrix = emb_matrix
X_train_indices = X_train_indices
Y_train = Y_train
hidden_size = 64
output_size = 2
input_size = 300
parameters = {}
epochs = 2 

# Weights 
parameters["Whh"] = cp.load('weights/Whh_adv.npy')
parameters["Wxh"] = cp.load('weights/Wxh_adv.npy')
parameters["Why"] = cp.load('weights/Why_adv.npy')

# Biases
parameters["bh"] = cp.load('weights/bh_adv.npy')
parameters["by"] = cp.load('weights/by_adv.npy')
```

```{code-cell} ipython3
for epoch in range(epochs):
    
    # Obtain loss and accuracy for each epoch
    train_loss, train_accuracy, parameters =  ProcessData(X_train_indices, imdb_emb_matrix, parameters, learn_rate = 0.1, target_values=Y_train, backprop=True, predict=False)
    
    # Update parameters
    parameters["Whh"] = paramters[0]
    parameters["Wxh"] = paramters[1]
    parameters["Why"] = paramters[2]
    parameters["bh"] = paramters[3]
    parameters["by"] = paramters[4]
    
    cp.cuda.Stream.null.synchronize()
    print(f'epoch : {epoch}, training_loss : {train_loss}, training_accuracy : {train_acc}')
```
