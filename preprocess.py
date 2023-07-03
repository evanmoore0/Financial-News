# Numpy, pandas
import pandas as pd
import numpy as np

# TQDM
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")


# Sklearn
from sklearn import utils
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences

# Gensim
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


# Keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional, SpatialDropout1D, Conv1D, MaxPooling1D, InputLayer


# Regular expression
import re

# Matplot
import seaborn as sns
import matplotlib.pyplot as plt

# Beautiful Soup
from bs4 import BeautifulSoup

# NLTK
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords

# Get Data
df = pd.read_csv('all-data.csv',delimiter=',',encoding='latin-1')

# Label columns
df = df.rename(columns={'neutral':'sentiment','According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .':'Message'})

# Set index
df.index = range(4845)
df['Message'].apply(lambda x: len(x.split(' '))).sum()

# Get values to plot the distribution of the data
cnt_pro = df['sentiment'].value_counts()

# plot cnt_pro
plt.figure(figsize=(12,6))
sns.barplot(x = cnt_pro.index, y= cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('sentiment', fontsize=12)
plt.show();

#Convert sting to numeric
sentiment  = {'positive': 0,'neutral': 1,'negative':2} 

df.sentiment = [sentiment[item] for item in df.sentiment] 


# Remove symbols
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text


df['Message'] = df['Message'].apply(cleanText)


train, test = train_test_split(df, test_size=0.000001 , random_state=42)

# To
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            #if len(word) < 0:
            if len(word) <= 0:
                continue
            tokens.append(word.lower())
    return tokens
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.sentiment]), axis=1)
test_tagged = test.apply(lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.sentiment]), axis=1)

# The maximum number of words to be used. (most frequent)
max_fatures = 500000

# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 50

#tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer = Tokenizer(num_words=max_fatures, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Message'].values)
X = tokenizer.texts_to_sequences(df['Message'].values)
X = pad_sequences(X)

print('Found %s unique tokens.' % len(X))

X = tokenizer.texts_to_sequences(df['Message'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

d2v_model = Doc2Vec(dm=1, dm_mean=1, vector_size=20, window=8, min_count=1, workers=1, alpha=0.065, min_alpha=0.065)
d2v_model.build_vocab([x for x in tqdm(train_tagged.values)])


for epoch in range(30):
    d2v_model.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    d2v_model.alpha -= 0.002
    d2v_model.min_alpha = d2v_model.alpha


# save the vectors in a new matrix
embedding_matrix = np.zeros((len(d2v_model.wv)+ 1, 20))

for i, vec in enumerate(d2v_model.dv.vectors):
    while i in vec <= 1000:
          embedding_matrix[i]=vec


Y = pd.get_dummies(df['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 42)
