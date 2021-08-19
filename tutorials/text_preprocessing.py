import pandas as pd
import argparse
import numpy as np
import re
import string 

class TextPreprocess:
    def __init__(self, filepath, emb_path):
        self.filepath = filepath
        self.emb_path = emb_path
        self.file_df = pd.read_csv(self.filepath)
        PAD_token = 0   # Used for padding short sentences
        SOS_token = 1   # Start-of-sentence token
        EOS_token = 2   # End-of-sentence token
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 0
        self.maxlen = 150

    def cleantext(self, text_column, remove_stopwords = True, remove_punc = True, target_column = None):
        data = self.file_df
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
        #self.maxlen = max([len(i) for i in X])
        if target_column:
            y = data_without_stopwords[f'{target_column}'].to_numpy()
            return X, y
        else:
            return X 

    def split_data (self, X, y, split_percentile):
        y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
        arr_rand = np.random.rand(X.shape[0])
        split = arr_rand < np.percentile(arr_rand, split_percentile)
        X_train = X[split]
        y_train = y[split]
        X_test =  X[~split]
        y_test = y[~split]

        return X_train, y_train, X_test, y_test

    # First entry of word into vocabulary
    # Word exists; increase word count
    def _add_word(self, word):
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

    def create_voc(self, x):
        for sample in x:
            #print(sample)
            tokens = re.split(r"([-\s.,;!?])+", sample)
            words = [x for x in tokens if (x not in '- \t\n.,;!?\\' and '\\' not in x)]
            for token in words:
                self._add_word(token)
        return self.index2word
    
    def sent_tokeniser (self, x):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', x)
        sentences.pop()
        sentences_cleaned = [re.sub(r'[^\w\s]', '', x) for x in sentences]
        return sentences_cleaned 

    def _loadGloveModel(self):
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

    def emb_matrix (self):
        word_to_vec_map = self._loadGloveModel()
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
        text_input_indices = np.zeros((len(text_data), self.maxlen))
        indexes = []
        for index, text in enumerate(text_data):
            text_indices = []
            for word in text.split(' '):
                if word in self.word2index:
                    text_indices.append(self.to_index(word))
            text_len = len(text_indices)
            #print('maxlen', self.maxlen)
            if text_len<=self.maxlen:
                for i in range(self.maxlen-text_len):
                    text_indices.append(0)
            else:
                continue 
            text_indices = np.array(text_indices)
            text_input_indices[index, :] = text_indices

        return text_input_indices




    

       



    
