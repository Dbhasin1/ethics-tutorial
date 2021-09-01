---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Sentiment Analysis on notable speeches of the last decade

---

This tutorial demonstrates how to build a simple <a href = 'https://en.wikipedia.org/wiki/Long_short-term_memory'> Long Short Term memory network (LSTM) </a> from scratch in NumPy to perform sentiment analysis on a socially relevant and ethically acquired dataset.

Your deep learning model - The LSTM is a form of a Recurrent Neural Network and will learn to classify a piece of text as positive or negative from the IMDB reviews dataset. The dataset contains 40,000 training and 10,000 test reviews and corresponding labels. Based on the numeric representations of these reviews and their corresponding labels <a href = 'https://en.wikipedia.org/wiki/Supervised_learning'> (supervised learning) </a> the neural network will be trained to learn the sentiment using forward propagation and backpropagaton through time since we are dealing with sequential data here. The output will be a vector containing the probabilities that the text samples are positive.

+++

Today, Deep Learning is getting adopted in everyday life and now it is more important to ensure that decisions that have been taken using AI are not reflecting discriminatory behavior towards a set of populations. It is important to take fairness into consideration while consuming the output from AI. Throughout the tutorial we'll try to question all the steps in our pipeline from an ethics point of view.

+++

## Prerequisites 

---

You are expected to be familiar with the Python programming language and array manipulation with NumPy. In addition, some understanding of Linear Algebra and Calculus is recommended. You should also be familiar with how Neural Networks work. For reference, you can visit the [Python](https://docs.python.org/dev/tutorial/index.html), [Linear algebra on n-dimensional arrays](https://numpy.org/doc/stable/user/tutorial-svd.html) and [Calculus](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/multivariable-calculus.html) tutorials.

To get a refresher on Deep Learning basics, You should consider reading [the d2l.ai book](https://d2l.ai/chapter_recurrent-neural-networks/index.html), which is an interactive deep learning book with multi-framework code, math, and discussions. You can also go through the [Deep learning on MNIST from scratch tutorial](https://numpy.org/numpy-tutorials/content/tutorial-deep-learning-on-mnist.html) to understand how a basic neural network is implemented from scratch.

In addition to NumPy, you will be utilizing the following Python standard modules for data loading and processing:
- [`pandas`](https://pandas.pydata.org/docs/) for handling dataframes 
- [`re`](https://docs.python.org/3/library/re.html) for tokenising textual data 
- [`string`](https://docs.python.org/3/library/string.html) for string operations  

    as well as:
- [Matplotlib](https://matplotlib.org/) for data visualization

This tutorial can be run locally in an isolated environment, such as [Virtualenv](https://virtualenv.pypa.io/en/stable/) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). You can use [Jupyter Notebook or JupyterLab](https://jupyter.org/install) to run each notebook cell.

+++

## Table of contents

---

1. Data Collection 

2. Preprocess the datasets

3. Build and train a LSTM network from scratch

4. Perform sentiment analysis on collected speeches 

5. Next steps

```{code-cell} ipython3
# Importing the necessary packages 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from text_preprocessing import TextPreprocess 
import string
```

## 1. Data Collection
----

Before we begin there are a few pointers you should always keep in mind before choosing the data you wish to train your model on:
- **Identifying Data Bias** - Bias is a component of the human thought process, and data collected from humans therefore inherently reflects that bias. Some ways in which this bias tends to occur in Machine Learning datasets are:
    - *Bias in historical data*: Historical data are often skewed towards, or against, particular groups.
        Data can also be severely imbalanced with limited information on protected groups.
    - *Bias in data collection mechanisms*: Lack of representativeness introduces inherent biases in the data collection process.  
    - *Bias towards observable outcomes*: In some scenarios, we have the information about True Outcomes only for a certain section of the population. In the absence of information on all outcomes, one cannot even measure fairness
- **Preserving human anonymity for sensitive data**: [Trevisan and Reilly](https://eprints.whiterose.ac.uk/91157/1/Ethical%20dilemmas.pdf) identified a list of sensitive topics that need to be handled with extra care. We present the same below along with a few additions:
    - personal daily routines (including location data);
    - individual details about impairment and/or medical records;
    - emotional accounts of pain and chronic illness;
    - financial information about income and/or welfare payments;
    - discrimination and abuse episodes;
    - criticism/praise of individual providers of healthcare and support services
    - suicidal thoughts;
    - criticism/praise of a power structure especially if it compromises their safety
    - personally-identifying information (even if anonymized in some way) including things like fingerprints or voice

In this section, you will be collecting two different datasets, the IMDB movie reviews dataset and a collection of 10 speeches curated for this tutorial including activists from different countries around the world, different times, and different topics. The former would be used to train the deep learning model while the latter will be used to perform sentiment analysis on.

+++

### Collecting the IMDB reviews dataset
The IMDB dataset can be found on the website by [Stanford AI Lab](http://ai.stanford.edu/~amaas/data/sentiment/). To make things a bit simpler we're using the dataframe version downloaded from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   > The IMDb platform allows the usage of their public datasets for personal and non-commercial use. We did our best to ensure that these reviews do not contain any of the aforementioned sensitive topics pertaining to the reviewer.

+++

### Collecting and loading the speech transcripts
We have chosen speeches by activists around the globe talking about issues like climate change, feminism, lgbtqa+ rights and racism. These were sourced them from newspapers, the official website of the United Nations and the archives of established universities as cited in the table below. A csv file was created containing the transcribed speeches, their speaker and the source the speeches were obtained from. 
We made sure to include different demographics in our data and included a range of different topics, most of which focus on social and/or ethical issues. The dataset is subjected to the CCO Creative Common License, which means that is free for the public to use and there are no copyrights reserved.

| Speech                                           | Speaker                 | Source                                                     |
|--------------------------------------------------|-------------------------|------------------------------------------------------------|
| Barnard College Commencement                     | Leymah Gbowee           | Barnard College - Columbia University official website     |
| UN Speech on youth Education                     | Malala Yousafzai        | Iowa state university archives                             |
| Remarks in the UNGA on racial discrimination     | Linda Thomas Greenfield | United States mission to the United Nation                 |
| How Dare You                                     | Greta Thunberg          | NBC’s official website                                     |
| The speech that silenced the world for 5 minutes | Severn Suzuki           | NTU blogs                                                  |
| The Hope Speech                                  | Harvey Milk             | University of Maryland archives                            |
| Violence against LGBTQA+                         | Michelle Bachelet       | United Nations office of high commisioner official website |
| I have a dream                                   | Martin Luther King      | Brittanica official website

+++

## 2. Preprocess the datasets
>Preprocessing data is an extremely crucial step before building any Deep learning model, however in an attempt to keep the tutorial focused on building the model, we will not dive deep into the code for preprocessing. Given below is a brief overview of all the steps we undertake to clean our data and convert it to its numeric representation. The [code](https://github.com/Dbhasin1/ethics-tutorial/blob/lstm-update/tutorials/text_preprocessing.py) is public and we encourage you to look it up for a better understanding of this section

1. **Text Denoising** : Before converting your text into vectors, it is important to clean it and remove all unhelpful parts a.k.a the noise from your data by converting all characters to lowercase, removing html tags, brackets and stop words (words that don't add much meaning to a sentence). Without this step the dataset is often a cluster of words that the computer doesn't understand.


2. **Tokenization** : So far the text we have is in its raw form, it needs to be broken apart into chunks called tokens because the most common way of processing language happens at the token level. This process of separating a piece of text into smaller units is called Tokenisation. The tokens obtained are then used to build a vocabulary. Vocabulary refers to a set of all tokens in the corpus along with a unique index allotted to each of them.


3. **Converting words to vectors** : A word embedding is a learned representation for text where words that have the same meaning have a similar representation. Individual words are represented as real-valued vectors in a predefined vector space. GloVe is n unsupervised algorithm developed by Stanford for generating word embeddings by generating global word-word co-occurence matrix from a corpus. You can download the zipped files containing the embeddings from https://nlp.stanford.edu/projects/glove/. Here you can choose any of the four options for different sizes or training datasets
 >The GloVe word embeddings include sets that were trained on billions of tokens, some up to 840 billion tokens. These algorithms exhibit stereotypical biases, such as gender bias which can be traced back to the original training data. For example certain occupations seem to be more biased towards a particular gender, reinforcing problematic stereotypes. The nearest solution to this problem are some de-biasing algorithms as the one presented in https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6835575.pdf which one can use on embeddings of their choice to mitigate bias, if present.

+++

You will need two files for the initial preprocessing:

```{code-cell} ipython3
imdb_data_path = '../../lstm/server/IMDB Dataset.csv'
emb_path = '../../lstm/server/glove.6B.300d.txt'
```

Next, you will load the IMDB dataset into a dataframe using Pandas

```{code-cell} ipython3
imdb_df = pd.read_csv(imdb_data_path)
```

We will use the text preprocessing class imported from the aforementioned [code](https://github.com/Dbhasin1/ethics-tutorial/blob/lstm-update/tutorials/text_preprocessing.py) to carry out the data preprocessing:

```{code-cell} ipython3
imdb_textproc = TextPreprocess(emb_path)
X = imdb_textproc.cleantext(imdb_df, 'review', remove_stopwords = True, remove_punc = True)
```

Now, we need to create a split between training and testing datasets. You can vary the split_percentile to try different ratios:

```{code-cell} ipython3
y = imdb_df['sentiment'].to_numpy()
X_train, Y_train, X_test, Y_test = imdb_textproc.split_data(X, y, split_percentile=10)
```

In order to replace each word with its word embedding, we will first need to replace it with its unique index.

```{code-cell} ipython3
X_train_indices = imdb_textproc.transform_input(X_train)
```

We will use the training corpus to build a matrice mapping each word index and word embedding. This will act as a cache for when we have to replace each word indice with its respective word embedding.

```{code-cell} ipython3
imdb_emb_matrix = imdb_textproc.emb_matrix(X_train)
```

Now, we will apply the same process to the speeches in our dataset:

```{code-cell} ipython3
speech_data_path = '../data/speeches.csv'
speech_df = pd.read_csv(speech_data_path)
speech_textproc = TextPreprocess(emb_path)
X_pred = speech_textproc.cleantext(speech_df, 'speech', remove_stopwords = True, remove_punc = False)
speakers = speech_df['speaker'].to_numpy()
speech_emb_matrix = speech_textproc.emb_matrix(X_pred)
```

### 3. Build the Deep Learning Model¶
 ---
 It’s time to start implementing our LSTM! You will have to first familiarize yourself with some high-level concepts of the basic building blocks of a deep learning model. You can refer to the [Deep learning on MNIST from scratch tutorial](https://numpy.org/numpy-tutorials/content/tutorial-deep-learning-on-mnist.html) for the same. 

You will then learn how a Recurrent Neural Network differs from a plain Neural Network and what makes it so suitable for processing sequential data. Afterwards, you will construct the building blocks of a simple deep learning model in Python and NumPy and train it to learn to classify the sentiment of a piece of text as positive or negative with a certain level of accuracy

### Introduction to a Long Short Term Memory Network

In a plain neural network, the information only moves in one direction — from the input layer, through the hidden layers, to the output layer. The information moves straight through the network and never takes the previous nodes into account at a later stage. Because it only considers the current input, it has no notion of order in time. It simply can’t remember anything about what happened in the past except its training.

In a RNN the information cycles through a loop. When it makes a decision, it considers the current input and also what it has learned from the inputs it received previously. 

The problem with an RNN however, is that the influence of a given input on the hidden layer, and therefore on the network output, either decays or blows up exponentially as it cycles around the network’s recurrent connections. This shortcoming is referred to as the vanishing gradient problem. Long Short-Term Memory (LSTM) is an RNN architecture specifically designed to address the vanishing gradient problem.

+++

### Overview of the Model Architecture 

![lstm.jpg](attachment:lstm.jpg)

+++

In the above image, The rectangles labelled 'A' are called `Cells` and they are the **Memory Blocks** of our LSTM network. They are responsible for choosing what to remember in a sequence and pass on that information to the next cell via two states called the `hidden state` $H_{t}$ and the `cell state` $C_{t}$ where $t$ indicates the time-step. To know how these states are calculated you'll need to understand the mechanisms happening inside a cell, we will recommend you to go through [ Long Short-Term Memory (LSTM)](http://d2l.ai/chapter_recurrent-modern/lstm.html).

+++

### But how do we obtain sentiment from the LSTM's output?
The hidden state we obtain from the last word in our sequence is considered to be a representation of all the information contained in a sequence. To classify this information into various classes (2 in our case, positive and negative) we can use a Fully Connected layer which firstly maps this information to a desired output size and an activation layer like sigmoid on top of it finally converts the output to a value between 0 and 1. We'll consider values greater than 0.5 to be indicative of a positive sentiment.

+++

Define a function to randomly initialise the parameters which will be learnt while our model trains

```{code-cell} ipython3
def initialise_params (hidden_dim, input_dim):
        Wf = np.random.randn(hidden_dim, hidden_dim + input_dim) # forget gate 
        bf = np.random.randn(hidden_dim, 1)
        Wi = np.random.randn(hidden_dim, hidden_dim + input_dim) # input gate 
        bi = np.random.randn(hidden_dim, 1)
        Wcm = np.random.randn(hidden_dim, hidden_dim + input_dim) # candidate memory gate 
        bcm = np.random.randn(hidden_dim, 1)
        Wo = np.random.randn(hidden_dim, hidden_dim + input_dim) # output gate 
        bo = np.random.randn(hidden_dim, 1)
        
        W2 = np.random.randn(1, hidden_dim) # fully connected classification layer 
        b2 = np.zeros((1, 1))

        parameters = {"Wf": Wf, "bf": bf, "Wi": Wi, "bi": bi, "Wcm": Wcm, "bcm": bcm, "Wo": Wo, "bo": bo, "W2": W2, "b2": b2}
        return parameters    
```

### Forward Propagation

Now that we have our initialised parameters we pass the input data in a forward direction through the network. Each layer accepts the input data, processes it and passes it to the successive layer. This process is called `Forward Propagation`. You will undertake the following mechanism to implement the same:
- Loading the word embeddings of the input data
- Passing the embeddings to an LSTM to obtain the output of the final cell
- Passing the final output from the LSTM through a fully connected layer to obtain the probability with which the sequence is positive 
- Storing all the intermediate outputs in a cache to utilise during backpropagation

+++

Define a function to calculate the sigmoid of an array

```{code-cell} ipython3
def sigmoid(x):
    # To prevent overflow 
    x = np.clip(x, -709.78, 709.78)
    res = 1 / (1 + np.exp(-x))
    return res
```

Define a function to carry out forward propagation

```{code-cell} ipython3
def forward_prop (X_vec, parameters):
    
    # Loading input data 
    X_vec = X_vec
    
    # Retrieve values of all parameters 
    Wf, bf, Wi, bi, Wcm, bcm, Wo, bo, W2, b2 = parameters.values()
    
    hidden_dim = Wf.shape[0]
    batch_size = X_vec.shape[1]
    time_steps = X_vec.shape[2]
    
    # Initialise hidden and cell state before passing to first time step
    prev_hidden_state = np.zeros((hidden_dim, batch_size))
    prev_cell_state = np.zeros(prev_hidden_state.shape)
    
    # Store all the intermediate and final variables here 
    caches = {'lstm_values':[], 'fc_values':[]}
    
    # Hidden state from the last cell in the LSTM layer is calculated.
    for t in range(time_steps):
        # Retrieve embedding for one word for each time step
        X_t = X_vec[:, :, t]
        
        # Concatenate prev_hidden_state and xt
        concat = np.vstack((prev_hidden_state, X_t))
        
        # Calculate output of the forget gate 
        ft = sigmoid(np.dot(Wf, concat) + bf)
        
        # Calculate output of the input gate 
        it = sigmoid(np.dot(Wi, concat) + bi) 
        cmt = np.tanh(np.dot(Wcm, concat) + bcm)
        io = it * cmt 
        
        # Update the cell state 
        next_cell_state = (ft * prev_cell_state) + io
        
        # Calculate output of the output gate 
        ot = sigmoid(np.dot(Wo, concat) + bo)
        
        # Update the hidden input 
        next_hidden_state =  ot * np.tanh(next_cell_state)
        
        # store values needed for backward propagation in cache
        cache = (next_hidden_state, next_cell_state, prev_hidden_state, prev_cell_state, ft, it, cmt, ot, X_t)
        caches['lstm_values'].append(cache)
        
        # Update hidden state and cell state for next time step
        prev_hidden_state = next_hidden_state
        prev_cell_state = next_cell_state

    # Pass through a fully connected layer to perform binary classification 
    z2 = np.dot(W2, next_hidden_state) + b2
    a2 = sigmoid(z2)
    cache = (a2, W2)
    caches['fc_values'].append(cache)
    
    return caches 
```

### Backpropagation

After each forward pass through the network, you will implement the `backpropagation through time` algorithm to accumulate gradients of each parameter over the time steps. Backpropagation through a LSTM is not as straightforward as through other common Deep Learning architectures, due to the special way its underlying layers interact. Nonetheless, the approach is largely the same; identifying dependencies and applying the chain rule.

+++

Lets start with defining a function to initialise gradients of each parameter as arrays made up of zeros with same dimensions as the corresponding parameter

```{code-cell} ipython3
# Initialise the gradients 
def initialise_grads (parameters):
    grads = {}
    for param in parameters.keys():
        grads[f'd{param}'] = np.zeros((parameters[param].shape))
    return grads    
```

Now we'll define a function to calculate the gradients of each intermediate value in the neural network with respect to the loss and accumulate those gradients over the entire sequence. To understand how the gradients are calculated at each step in greater depth, you are suggested to follow this helpful [blog](https://christinakouridi.blog/2019/06/19/backpropagation-lstm/) by Christina Kouridi

```{code-cell} ipython3
 def backprop (y, caches, hidden_dim, input_dim, time_steps,  parameters):
    # Retrieve output and corresponding weights of fully connected layer
    A2, W2 = caches['fc_values'][0]
    # Retrieve hidden state calculated in the last time step
    h_last = caches['lstm_values'][-1][0]
    # Retrieve batch size 
    batch_size = y.shape[1]
    
    pred_value = np.array(A2)
    target_value = np.array(y)
    
    # Initialise gradients 
    dWf, dbf, dWi, dbi, dWcm, dbcm, dWo, dbo, dW2, db2 = initialise_grads(parameters).values()
    # Store gradients in a dictionary
    grads = {"dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi, "dWcm": dWcm, "dbcm": dbcm, 
                 "dWo": dWo, "dbo": dbo, "dW2": dW2, "db2": db2}
    
    # Calculate gradients of the fully connected layer 
    # dZ2 = dL/da2 * da2/dZ2
    dZ2 = pred_value - target_value
    # dW2 = dL/da2 * da2/dZ2 * dZ2/dW2
    dW2 = (1 / batch_size) * np.dot(dZ2, h_last.T)
    # db2 = dL/da2 * da2/dZ2 * dZ2/db2
    db2 = (1 / batch_size) * np.sum(dZ2)
    
    # Gradient of Loss w.r.t the last hidden output of the LSTM 
    # dh_last = dZ2 * W2 
    dh_last = np.dot(W2.T, dZ2)  
    
    # Initialise gradients w.r.t previous hidden state and cell state 
    dh_prev = dh_last
    dc_prev = np.zeros((dh_prev.shape))
    
    # loop back over the whole sequence
    for t in reversed(range(time_steps)):
        cache = caches['lstm_values'][t]
        
        # Retrieve parameters from "parameters"
        Wf = parameters["Wf"]
        Wi = parameters["Wi"]
        Wcm = parameters["Wcm"]
        Wo = parameters["Wo"]

        # Retrieve information from "cache"
        (next_hidden_state, next_cell_state, prev_hidden_state, prev_cell_state, ft, it, cmt, ot, X_t) = cache
        # Input to gates of LSTM is [prev_hidden_state, X_t]
        concat = np.concatenate((prev_hidden_state, X_t), axis=0)
        
        # Compute gates related derivatives
        # Calculate derivative w.r.t the parameters of forget gate 
        # dft = dL/da2 * da2/dZ2 * dZ2/dh_prev * dh_prev/dc_prev * dc_prev/dft
        dft = (dc_prev * prev_cell_state + ot * (1 - np.square(np.tanh(next_cell_state))) * prev_cell_state * dh_prev) * ft * (1 - ft)
        # dWf = dft * dft/dWf
        dWf = np.dot(dft, concat.T)
         # dbf = dft * dft/dbf
        dbf = np.sum(dft, axis=1, keepdims=True)
        # dh_f = dft * dft/dh_prev
        dh_f =  np.dot(Wf[:, :hidden_dim].T, dft)
        
        # Calculate derivative w.r.t the parameters of input gate 
        # dit = dL/da2 * da2/dZ2 * dZ2/dh_prev * dh_prev/dc_prev * dc_prev/dit
        dit = (dc_prev * cmt + ot * (1 - np.square(np.tanh(next_cell_state))) * cmt * dh_prev) * it * (1 - it)
        # dcmt = dL/da2 * da2/dZ2 * dZ2/dh_prev * dh_prev/dc_prev * dc_prev/dcmt
        dcmt = (dc_prev * it + ot * (1 - np.square(np.tanh(next_cell_state))) * it * dh_prev) * (1 - np.square(cmt))
        # dWi = dit * dit/dWi
        dWi = np.dot(dit, concat.T)
        # dWcm = dcmt * dcmt/dWcm
        dWcm = np.dot(dcmt, concat.T)
        # dbi = dit * dit/dbi
        dbi = np.sum(dit, axis=1, keepdims=True)
        # dWcm = dcmt * dcmt/dbcm
        dbcm = np.sum(dcmt, axis=1, keepdims=True)
        # dhi = dit * dit/dh_prev
        dh_i =  np.dot(Wi[:, :hidden_dim].T, dit)
        # dhcm = dcmt * dcmt/dh_prev
        dh_cm = np.dot(Wcm[:, :hidden_dim].T, dcmt)
        
        # Calculate derivative w.r.t the parameters of output gate 
        # dot = dL/da2 * da2/dZ2 * dZ2/dh_prev * dh_prev/dot 
        dot = dh_prev * np.tanh(next_cell_state) * ot * (1 - ot)
        # dWo = dot * dot/dWo
        dWo = np.dot(dot, concat.T)
        # dbo = dot * dot/dbo
        dbo = np.sum(dot, axis=1, keepdims=True)
        # dho = dot * dot/dho
        dh_o = np.dot(Wo[:, :hidden_dim].T, dot)
       
        # Compute derivatives w.r.t previous hidden state and the previous cell state 
        dh_prev = dh_f + dh_i + dh_cm + dh_o 
        dc_prev = dc_prev * ft + ot * (1 - np.square(np.tanh(next_cell_state))) * ft * dh_prev
        
        # sum up the gradients over the sequence 
        grads["dWf"] += dWf
        grads["dWi"] += dWi
        grads["dWcm"] += dWcm
        grads["dWo"] += dWo
        grads["dbf"] += dbf
        grads["dbi"] += dbi
        grads["dbcm"] += dbcm
        grads["dbo"] += dbo
    
    # Rescale the gradients to improve training stability 
    for key in grads:
        factor= np.linalg.norm(grads[key])
        if factor != 0:
            grads[key] = grads[key]/factor
        
    return grads
```

### Updating the Parameters 

We update the parameters through an optimization algorithm called [Adam](https://optimization.cbe.cornell.edu/index.php?title=Adam) which is an extension to stochastic gradient descent that has recently seen broader adoption for deep learning applications in computer vision and natural language processing. Specifically, the algorithm calculates an exponential moving average of the gradient and the squared gradient, and the parameters beta1 and beta2 control the decay rates of these moving averages. Adam has shown increased convergence and robustness over other gradient descent algorithms and is often recommended as the default optimizer for training.

+++

Define a function to initialise the moving averages for each parameter

```{code-cell} ipython3
# initialise the moving averages
def initialise_mav (hidden_dim, input_dim):
    v = {}
    s = {}

    grad_keys = ["dWf", "dbf", "dWi", "dbi", "dWcm", "dbcm", "dWo", "dbo", "dW2", "db2"]
    grad_shapes = [(hidden_dim, hidden_dim + input_dim), (hidden_dim, 1),  # shape of dWf, dbf
               (hidden_dim, hidden_dim + input_dim), (hidden_dim, 1),  # shape of dWi, dbi
               (hidden_dim, hidden_dim + input_dim), (hidden_dim, 1),  # shape of dWcm, dbcm
               (hidden_dim, hidden_dim + input_dim), (hidden_dim, 1),  # shape of dWo, dbo
               (1, hidden_dim), (1, 1)]  # shape of dW2, db2

    # Initialize dictionaries v, s
    for key, shape in zip(grad_keys, grad_shapes):
        v[key] = np.zeros(shape)
        s[key] = np.zeros(shape)
        
    # Return initialised moving averages 
    return v,s 
    
```

Define a function to update the parameters

```{code-cell} ipython3
# Update the parameters using Adam optimization 
def update_parameters (parameters, gradients, v, s, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    param_keys = list(parameters.keys())
    grad_keys = list(gradients.keys())
    
    for param_key, grad_key in zip(param_keys, grad_keys):
        # Moving average of the gradients
        v[grad_key] = beta1 * v[grad_key] + (1 - beta1) * gradients[grad_key]

        # Moving average of the squared gradients
        s[grad_key] = beta2 * s[grad_key] + (1 - beta2) * (gradients[grad_key] ** 2)

        # Update parameters
        parameters[param_key] = parameters[param_key] - learning_rate * v[grad_key] / np.sqrt(
            s[grad_key] + epsilon)
    # Return updated parameters and moving averages 
    return parameters, v, s  
```

### Training the Network
---

+++

You will start by initialising all the parameters and hyperparameters being used in your network

```{code-cell} ipython3
batch_size = 8
hidden_dim = 64
input_dim = 300
time_steps = X_train_indices.shape[1]
learning_rate = 0.01
epochs = 20
parameters = initialise_params(hidden_dim, input_dim)
v,s = initialise_mav(hidden_dim, input_dim)
```

You will then split your input data into batches as it requires way less memory as compared to loading the complete data set and is way faster than loading the data set one input at a time

```{code-cell} ipython3
# Retrieve total number of input samples (tweets in our case)
num_samples = X_train_indices.shape[0]
# Obtain the total number of batches 
no_batches = int(num_samples / batch_size)
# Initialise batch vectors of the training data 
X_batches, y_batches = np.zeros((no_batches, batch_size, time_steps)), np.zeros((no_batches, 1, batch_size))
y_train = Y_train.reshape((1, Y_train.shape[0]))
for i in range(no_batches):
    start_example, end_example = i * batch_size, (i + 1) * batch_size
    X_batches[i, :, :] = X_train_indices[start_example:end_example,:] 
    y_batches[i, :, :] = y_train[:, start_example:end_example] 
```

Define a function that replaces each word indice contained in an array with its corresponding word embedding

```{code-cell} ipython3
def embedding_vectors(X_b, embedding_matrix):
    # Retrieve batch size and number of time steps 
    batch_size, time_steps = X_b.shape[0], X_b.shape[1]
    # Retrieve dimension of word embeddings 
    emb_dim = embedding_matrix[0].shape[0]
    
    # Obtain array of dimensions (emb_dim, m, T_x) 
    # containing word embeddings for corresponding word indice
    X_emb = np.zeros((emb_dim, batch_size, time_steps))
    for i in range(batch_size):
        for t in range(time_steps):
            X_emb[:, i, t] = embedding_matrix[int(X_b[i, t])]
            
    # Return embedding array
    return X_emb
```

To optimise your deep learning network, you need to calculate a loss based on how well the model is doing on the training data. Loss value implies how poorly or well a model behaves after each iteration of optimization. <br>
Define a function to calculate the loss using [negative log likelihood](http://d2l.ai/chapter_linear-networks/softmax-regression.html?highlight=negative%20log%20likelihood#log-likelihood)

```{code-cell} ipython3
def loss_f(A, Y):
    # Retrieve batch size 
    batch_size = Y.shape[1]
    # Implement formula for negative log likelihood 
    loss = - Y * np.log(A) - (1 - Y) * np.log(1 - A)
    # Calculate cost function as the average of losses for one batch
    cost = (1 / batch_size) * np.sum(loss)
    # Return cost function 
    return np.squeeze(cost)
```

Set up the neural network's learning experiment with a training loop and start the training process.
>Skip running this cell if you already have the trained parameters stored in a `npy` file

```{code-cell} ipython3
# To store training losses 
# training_losses = []

# This is a training loop.
# Run the learning experiment for a defined number of epochs (iterations).
for epoch in range(epochs):
    #################
    # Training step #
    #################
    training_losses = []
    for b in range(no_batches):
        # retrieve a single batch and its corresponding target variables 
        x_b = X_batches[b, :, :]
        y_b = y_batches[b, :, :]
        # replace word indices with word embeddings before passing to neural network
        x_emb = embedding_vectors(x_b, imdb_emb_matrix)
        
        # Forward propagation/forward pass:
        caches = forward_prop(x_emb, parameters)
        
        # Backward propagation/backward pass:
        gradients = backprop(y_b, caches, hidden_dim, input_dim, time_steps,  parameters)
        
        # Update the weights and biases for the LSTM and fully connected layer 
        parameters, v, s = update_parameters (parameters, gradients, v, s, learning_rate=learning_rate, 
                                              beta1=0.999, beta2=0.9, epsilon=1e-8)
        
        # Measure the training error (loss function) between the actual
        # sentiment (the truth) and the prediction by the model.
        y_pred = caches['fc_values'][0][0]
        loss = loss_f(y_pred, y_b)
        # Store training set losses
        training_losses.append(loss)
        
    # Calculate average of training losses for one epoch
    mean_cost = np.mean(training_losses)
    print(f'Epoch {epoch + 1} finished. \t  Loss : {mean_cost}')

# save the trained parameters to a npy file 
np.save('parameters.npy',parameters)
```

### Sentiment Analysis on the Speech Data
---

+++

Once our model is trained, we can use the updated parameters to start making our predicitons. We break each speech into paragraphs of uniform size before passing them to the Deep Learning model and predicting the sentiment of each paragraph

```{code-cell} ipython3
# To store predicted sentiments 
preds = []
para_len = 100

# Retrieve trained values of the parameters
parameters = np.load('parameters.npy', allow_pickle='TRUE').item()

# This is the prediction loop.
for text in X_pred:
    # retrieve the speaker and corresponding speech
    speech = text
    # split the speech into a list of words 
    words = speech.split()
    # obtain the total number of paragraphs
    no_paras = int(np.ceil(len(words)/para_len))
    # split the speech into a list of sentences 
    sentences = speech_textproc.sent_tokeniser(speech)
    # aggregate the sentences into paragraphs
    k, m = divmod(len(sentences), no_paras)
    agg_sentences = [sentences[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(no_paras)]
    paras = [' '.join(sents) for sents in agg_sentences]
    # replace each word with its corresponding word indice 
    X_pred_indices = speech_textproc.transform_input(paras)
    # replace word indices with word embeddings before passing to neural network
    x_emb = embedding_vectors(X_pred_indices, speech_emb_matrix)
    # Forward Propagation
    caches = forward_prop(x_emb, parameters)
    
    # Retrieve the output of the fully connected layer 
    A2 = caches['fc_values'][0][0]
    threshold = np.mean(A2)

    # Mark all predictions >0.5 as positive and <0.5 as negative 
    pred = np.zeros(A2.shape)
    indices = np.where(A2 > threshold)  # indices where output > 0.5
    pred[indices] = 1  # are set to 1
    
    # Store predictions 
    preds.append(pred[0])
```

Visualising our predictions using `Matplotlib`:

```{code-cell} ipython3
x_axis = []
data = {'positive':[], 'negative':[]}
for i in range(len(preds)):
    # Extract the speaker and the corresponding sentiment predictions 
    speaker = speakers[i]
    sentiment = preds[i]
    # The speakers will be used to label the x-axis in our plot 
    x_axis.append(speaker)
    # Obtain percentage of paragraphs with positive predicted sentiment 
    pos_perc = sum(sentiment)/len(sentiment)
    # Store positive and negative percentages 
    data['positive'].append(pos_perc*100)
    data['negative'].append(100*(1-pos_perc))    

index = pd.Index(x_axis, name='speaker')
df = pd.DataFrame(data, index=index)
ax = df.plot(kind='bar', stacked=True, figsize=(10, 6))
ax.set_ylabel('percentage')
plt.legend(title='labels', bbox_to_anchor=(1.0, 1), loc='upper left')
plt.show()
```

### Looking at our Neural Network from an ethical perspective
---

+++

It's crucial to understand that accurately identifying a text's sentiment is not easy primarily because of the complex ways in which humans express sentiment, using irony, sarcasm, humor, or, in social media, abbreviation. Moreover neatly placing text into two categories: 'positive' and 'negative' can be problematic because it is being done without any context. Words or abbreviations can convey very different sentiments depending on age and location, none of which we took into account while building our model.

Along with data, there are also growing concerns that data processing algorithms are influencing policy and daily lives in ways that are not transparent and introduce biases. Certain biases such as the [Inductive Bias](https://en.wikipedia.org/wiki/Inductive_bias#:~:text=The%20inductive%20bias%20(also%20known,that%20it%20has%20not%20encountered.&text=The%20kind%20of%20necessary%20assumptions,in%20the%20phrase%20inductive%20bias) are absolutely essential to help a Machine Learning model generalise better, for example the LSTM we built earlier is biased towards preserving contextual information over long sequences which makes it so suitable for processing sequential data. The problem arises when [societal biases](https://hbr.org/2019/10/what-do-we-do-about-the-biases-in-ai) creep into algorithmic predictions. Optimizing Machine algorithms via methods like [hyperparameter tuning](https://en.wikipedia.org/wiki/Hyperparameter_optimization) can then further amplify these biases by learning every bit of information in the data. 


There are also cases where bias is only in the output and not the inputs (data, algorithm). For example, in sentiment analysis [accuracy tends to be higher on female-authored texts than on male-authored ones]( https://doi.org/10.3390/electronics9020374). End users of sentiment analysis should be aware that its small gender biases can affect the conclusions drawn from it and apply correction factors when necessary. Hence, it is important that demands for algorithmic accountability should include the ability to test the outputs of a system, including the ability to drill down into different user groups by gender, ethnicity and other characteristics, to identify, and hopefully suggest corrections for, system output biases.

+++

### Next Steps
---

+++

You have learned how to build and train a simple Long Short Term Memory network from scratch using just NumPy to perform sentiment analysis.

To further enhance and optimize your neural network model, you can consider one of a mixture of the following:

- Increase the training sample size by increasing the `split_percentile`.
- Alter the architecture by introducing multiple LSTM layers to make the network deeper.
- Use a higher epoch size to train longer and add more regularization techniques, such as early stopping, to prevent overfitting.
- Introduce a validation set for an unbiased valuation of the model fit.
- Apply batch normalization for faster and more stable training.
- Tune other parameters, such as the learning rate and hidden layer size.
- Replace LSTM with a [Bidirectional LSTM](https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks) to use both left and right context for predicting sentiment.

Nowadays, LSTMs have been replaced by the [Transformer](https://jalammar.github.io/illustrated-transformer/) which uses [Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) to tackle all the problems that plague an LSTM such as as lack of [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning), lack of [parallel training](https://web.stanford.edu/~rezab/classes/cme323/S16/projects_reports/hedge_usmani.pdf) and a long gradient chain for lengthy sequences

Building a neural network from scratch with NumPy is a great way to learn more about NumPy and about deep learning. However, for real-world applications you should use specialized frameworks — such as PyTorch, JAX, TensorFlow or MXNet — that provide NumPy-like APIs, have built-in automatic differentiation and GPU support, and are designed for high-performance numerical computing and machine learning.
