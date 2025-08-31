import streamlit as st 
import re
import nltk

from nltk.stem import WordNetLemmatizer 
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
import pandas as pd 
from nltk.tokenize import word_tokenize, sent_tokenize

import string
import matplotlib.pylab as plt
from tensorflow.keras.models import load_model
from gensim.models import Doc2Vec
import numpy as np 
import time

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download("punkt_tab")

st.title("Spam Email Classification!!!")

model = load_model('saved_email_classification.keras')

def get_wordnet_pos(tag):
    """Map POS tag to WordNet POS tag for better lemmatization"""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun
    
def custom_tokenizer(text):

    tokens = []

    # replace URLs with tokens
    url_pattern = r'(?:https?\s?:\/\/?|http\s?:\/\/?|https\s?\/\/|http\s?\/\/|www\.)[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    text = re.sub(url_pattern, ' URL_TOKEN ', text)
    tokens.extend(['URL_TOKEN'] * len(urls))  # if more than two url then

    # Replace emails with tokens
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    text = re.sub(email_pattern, ' EMAIL_TOKEN ', text)
    tokens.extend(['EMAIL_TOKEN'] * len(emails))

    # replace phone numbers
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    phones = re.findall(phone_pattern, text)
    text = re.sub(phone_pattern, ' PHONE_TOKEN ', text)
    tokens.extend(['PHONE_TOKEN'] * len(phones))

    # replace currency
    currency_pattern =  r'\$\d+(?:\.\d{2})?'
    currency = re.findall(currency_pattern, text)
    text = re.sub(currency_pattern, ' MONEY_TOKEN ', text)
    tokens.extend(['MONEY_TOKEN'] * len(currency))

    # replace numbers
    number_pattern = r'\b\d+(?:\.\d+)?\b'
    numbers = re.findall(number_pattern, text)
    text = re.sub(number_pattern, ' NUM_TOKEN ', text)
    tokens.extend(['NUM_TOKEN'] * len(numbers))

    # remove _____ spaces
    text = re.sub(r'[_]{3,}', ' ', text)

    # tokenizing common words
    word_pattern = r'\b[A-Za-z]+\b'
    words = re.findall(word_pattern, text.lower())
    tokens.extend(words)

    return text, tokens

def text_preprocessing(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # lowercase
    text = text.lower().strip()

    # clean weird punctuation (like ... or ----)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'-{2,}', ' ', text)

    # tokenize
    words = word_tokenize(text)

    # remove punctuation & empty tokens
    words = [w for w in words if w not in string.punctuation and w.strip() != '']

    # POS tagging
    tokens = pos_tag(words)

    # keep important words even if in stopwords
    words_shouldnot_removed = {
        'is','am','are','was','were','be','been','being',
        'have','has','had','having',
        'do','does','did','doing',
        'will','would','can','could','should','shall','may','might','must',
        'not','no','never','nothing','nobody','nowhere',
        'i','you','he','she','it','we','they','me','him','her','us','them',
        'my','your','his','her','its','our','their',
        'this','that','these','those',
        'who','what','when','where','why','how','which',
        'and','or','but','because','if','when','while','although','though',
        'very','really','quite','rather','too','so','such'
    }

    content_pos_tags = {'NN','NNS','NNP','NNPS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ'}

    lemmatized_sentence = []
    for word, pos in tokens:
        wn_pos = get_wordnet_pos(pos)  # better lemmatization
        lemma = lemmatizer.lemmatize(word, wn_pos)

        if (word in words_shouldnot_removed
            or pos in content_pos_tags
            or word not in stop_words):
            lemmatized_sentence.append(lemma)

    lemm_sent = ' '.join(lemmatized_sentence)

    # final pass with custom tokenizer
    text, tokens = custom_tokenizer(lemm_sent)

    return text

input_text_area = st.text_area("enter the spam or ham email to classify: ")

if input_text_area:

    processed_tokens = text_preprocessing(input_text_area)
    tokens = word_tokenize(processed_tokens)

    doc2vec = Doc2Vec.load("doc_2_vec.model")
    vector =  doc2vec.infer_vector(tokens)
    # print(vector)
    # print(processed_tokens)
    input_text = np.expand_dims(vector, axis=0)
    # print(input_text)
    # print(len(input_text))
    # print(input_text.shape)
    pred_output = model.predict(input_text)
    # print(pred_output)

    'Starting a long computation...'

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
    # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    '...and now we\'re done!'



    if pred_output > 0.5 : 
        st.success("the given email is ham")
    else:
        st.success("The given email is spam")