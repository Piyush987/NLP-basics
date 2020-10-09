import numpy as np
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec, Phrases
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


df = pd.read_csv(r'IMDB Dataset.csv')
print(df.info()) #Check non NULL values in columns

#Cleaning, preprocess
def clean_data(text):
  text = re.sub(r'<br />', ' ', text) #Removes Html tag
  text = re.sub(r'[^\ a-zA-Z0-9]+', '', text)  #Removes non alphanumeric
  text = re.sub(r'^\s*|\s\s*', ' ', text).strip() #Removes extra whitespace, tabs
  stop_words = set(stopwords.words('english')) 
  lemmatizer = WordNetLemmatizer()
  text = text.lower().split() #Converts text to lowercase
  cleaned_text = list()
  for word in text:        
    if word in stop_words:    #Removes Stopwords, i.e words that don't convey any meaningful context/sentiments
      continue    
    word = lemmatizer.lemmatize(word, pos = 'v')    #Lemmatize words, pos = verbs, i.e playing, played becomes play
    cleaned_text.append(word)
  text = ' '.join(cleaned_text)
  return text

df['cleaned_review'] = df['review'].apply(lambda x: clean_data(x))


def convert_sentiment_to_int(text):  #Convert sentiment positive to 1, negative to 0
  if(text.lower() == 'positive'):
    text = 1
  else:
    text = 0
  return text

df['sentiment'] = df['sentiment'].apply(lambda x: convert_sentiment_to_int(x))

result = [len(x) for x in [df['cleaned_review'].iloc[i].split() for i in range(50000)]]
print(np.mean(result)) #Mean no of words in each cleaned review

X_train = [text for text in list(df['cleaned_review'].iloc[:25000])] #Preparation of X,Y
X_test = [text for text in list(df['cleaned_review'].iloc[25000:])]
Y_train = [text for text in list(df['sentiment'].iloc[:25000])]
Y_test = [text for text in list(df['sentiment'].iloc[25000:])]

print(len(np.unique(np.hstack(X_train)))) #No of unique words in cleaned review

#Tokenize and Padding
X = [text for text in list(df['cleaned_review'])] 
max_vocab = 10000  #Max features
max_sent_length = 150  #Max word length of every review
tokenizer = Tokenizer(num_words = max_vocab)
tokenizer.fit_on_texts(X)
X_train_tokenized = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen = max_sent_length) #Tokenization, i.e converting words to int
X_test_tokenized = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen = max_sent_length)
