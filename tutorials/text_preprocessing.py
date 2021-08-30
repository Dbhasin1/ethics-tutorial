import pandas as pd
import argparse
import numpy as np
import re
import string 

class TextPreprocess:
    """Text Preprocessing for a Natural Language Processing model.

    Attributes
    ----------
    emb_path : str
        Path to the word embedding file.
    word2index : :obj:`dict` of :obj:`str`
        Mapping between word and corresponding index.
    word2count : :obj:`dict` of :obj:`str`
        Mapping between word and number of occurences.
    index2word : :obj:`dict` of :obj:`int`
        Mapping between index and corresponding word. 
    num_words : int
        Counter for total words in text.
    maxlen : int
        Maximum length of each subsample of text data 

    """
    def __init__(self, emb_path):
        self.emb_path = emb_path
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_words = 0
        self.maxlen = 150

    def cleantext(self, df, text_column, remove_stopwords = True, remove_punc = True):
        """Function to clean text data by removing stopwords, tags and punctuation.

        Parameters
        ----------
        df : pandas dataframe 
            The dataframe housing the input data.
        text_column : str
            Column in dataframe whose text is to be cleaned.
        remove_stopwords : bool
            if True, remove stopwords from text
        remove_punc : bool
            if True, remove punctuation suymbols from text

        Returns
        -------
        Numpy array 
            Cleaned text.

        """
        data = df
        # converting all characters to lowercase 
        data[text_column] = data[text_column].str.lower()

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
            data[f'{column} without stopwords'] = data[column].apply(lambda x : ' '.join([word for word in x.split() if word not in (stopwords)]))
            return data

        def remove_tags(string):
            result = re.sub('<*>','',string)
            return result

        # remove html tags and brackets from text 
        if remove_stopwords:
            data_without_stopwords = remove_stopwords(data, text_column)
            data_without_stopwords[f'clean_{text_column}']= data_without_stopwords[f'{text_column} without stopwords'].apply(lambda cw : remove_tags(cw))
        if remove_punc:
            data_without_stopwords[f'clean_{text_column}'] = data_without_stopwords[f'clean_{text_column}'].str.replace('[{}]'.format(string.punctuation), ' ')

        X = data_without_stopwords[f'clean_{text_column}'].to_numpy()

        return X 

    def split_data (self, X, y, split_percentile):
        """Function to split data into training and testing data.

        Parameters
        ----------
        X : Numpy Array
            Contains textual data.
        y : Numpy Array
            Contains target data.
        split_percentile : int
            Proportion of training to testing data.
         

        Returns
        -------
        Tuple 
            Contains numpy arrays of test and training data.

        """
        y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
        arr_rand = np.random.rand(X.shape[0])
        split = arr_rand < np.percentile(arr_rand, split_percentile)
        X_train = X[split]
        y_train = y[split]
        X_test =  X[~split]
        y_test = y[~split]

        return (X_train, y_train, X_test, y_test)


    def add_word(self, word):
        """Function to allot unique index to word.

        Parameters
        ----------
        word : str
            To be added in the vocabulary.

        """
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1


    def to_word(self, index):
        """Function to retrieve corresponding word.

        Parameters
        ----------
        index : int 
            integer representation of a word.

        Returns
        -------
        str
            word corresponding to given index.

        """
        return self.index2word[index]


    def to_index(self, word): 
        """Function to retrieve corresponding index.

        Parameters
        ----------
        word : str 
            given word.

        Returns
        -------
        int
            index corresponding to given word.

        """
        return self.word2index[word]

    def create_voc(self, x):
        """Function to create corpus's vocabulary.

        Parameters
        ----------
        x : :obj:`list` of :obj:`str`
            contains input textual data.

        """
        for sample in x:
            #print(sample)
            tokens = re.split(r"([-\s.,;!?])+", sample)
            words = [x for x in tokens if (x not in '- \t\n.,;!?\\' and '\\' not in x)]
            for token in words:
                self.add_word(token)
    
    def sent_tokeniser (self, x):
        """Function to split text into sentences.

        Parameters
        ----------
        x : str
            piece of text

        Returns
        -------
        list 
            sentences with punctuation removed.

        """
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', x)
        sentences.pop()
        sentences_cleaned = [re.sub(r'[^\w\s]', '', x) for x in sentences]
        return sentences_cleaned 

    def loadGloveModel(self):
        """Function to read from the word embedding file.

        Returns
        -------
        Dict 
            mapping from word to corresponding word embedding.

        """
        print("Loading Glove Model")
        File = self.emb_path
        f = open(File,'r')
        gloveModel = {}
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
        print(len(gloveModel)," words loaded!")
        return gloveModel

    def emb_matrix (self, x):
        """Function to map word index and word embedding.

        Returns
        -------
        Numpy array 
            Each row contains word embedding of a word with index = row number.

        """
        self.create_voc(x)
        word_to_vec_map = self.loadGloveModel()
        words_to_index = self.word2index
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


    def transform_input(self, text_data):
        """Function to replace words with corresponding word indices.
           Converts all sub-samples to same length.

        Parameters
        ----------
        text_data ::obj:`array` of :obj:`str`
            Contains textual data
            
        Returns
        -------
        Numpy array 
            indice representation of each text sub-sample.

        """
        text_input_indices = np.zeros((len(text_data), self.maxlen))
        indexes = []
        for index, text in enumerate(text_data):
            text_indices = []
            for word in text.split(' '):
                if word in self.word2index:
                    text_indices.append(self.to_index(word))
            text_len = len(text_indices)
            if text_len<=self.maxlen:
                for i in range(self.maxlen-text_len):
                    text_indices.append(0)
            else:
                continue 
            text_indices = np.array(text_indices)
            text_input_indices[index, :] = text_indices

        return text_input_indices
