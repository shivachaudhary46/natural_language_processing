#!/usr/bin/env python
# coding: utf-8

# # Email Spam Detection
# <!--
# *** I'm using markdown "reference style" links for readability.
# *** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
# *** See the bottom of this document for the declaration of the reference variables
# *** This documentation will be finished when this projects finishes.
# -->
# 1. #### Importing libraries
#         this are necessary libraries
#             - pandas
#             - numpy
#             - scikit-learn.preprocessing (LabelEncoder)

# In[1]:


import gensim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pylab as plt
import nltk
import seaborn as sns

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')


# 2. #### Collection of Data
#         I use public datasets:
#             - Email-Spam-Classification-dataset-csv
#             - Enron Email Dataset
#             - deceptive-opinion-spam-corpus
#             - spam-or-not-spam-dataset
#             - spam-email
#             - sms-spam-collection-dataset
#             - email-spam-dataset
#             - email-spam-classification-dataset-csv
#         format: Each row is in text (message/email) and label (spam/ham).

# In[2]:


complete_spam = pd.read_csv("dataset/completeSpamAssassin.csv", index_col=0)
null_rows = complete_spam[complete_spam.isnull().any(axis=1)]
complete_spam = complete_spam.dropna()
complete_spam.rename(columns={"Body": "message", "Label":'label'}, inplace=True)
complete_spam.head()


# In[3]:


enron = pd.read_csv("dataset/enronSpamSubset.csv", index_col=0)
enron = enron.drop(columns=["Unnamed: 0"])
enron.rename(columns={"Body": "message", "Label":'label'}, inplace=True)
enron.head()


# In[4]:


ling_spam = pd.read_csv("dataset/lingSpam.csv", index_col=0)
ling_spam.rename(columns={"Body": "message", "Label":'label'}, inplace=True)
ling_spam.head()


# In[5]:


deceptive = pd.read_csv("dataset/deceptive-opinion.csv")
deceptive = deceptive.iloc[:, [2, 4]]
label = pd.get_dummies(deceptive['polarity'], drop_first=True)
deceptive.drop(columns=['polarity'], inplace=True)
new_deceptive = pd.concat([deceptive, label], axis=1)
new_deceptive.rename(columns={"text": "message", "positive":'label'}, inplace=True)
new_deceptive.head()


# In[6]:


spam = pd.read_csv("dataset/spam_or_not_spam.csv")
spam.rename(columns={'email': 'message'}, inplace=True)
spam.head()


# ##### Shape of Dataset

# In[7]:


print(f"dataset 1 : {complete_spam.shape}")
print(f"dataset 2 : {spam.shape}")
print(f"dataset 3 : {new_deceptive.shape}")
print(f"dataset 4 : {ling_spam.shape}")
print(f"dataset 5 : {enron.shape}")
print(f"total expected shape : ({complete_spam.shape[0] + spam.shape[0] + new_deceptive.shape[0] + ling_spam.shape[0] + enron.shape[0]}, 2)")


# In[8]:


dfs = [complete_spam, spam, new_deceptive, ling_spam, enron]
spam_dataset = pd.concat(dfs, axis=0, ignore_index=True)
spam_dataset.tail()
print(spam_dataset.shape)


# ##### Checking Missing Values

# In[9]:


spam_dataset.isnull().sum()


# ##### Checking Duplicate Values

# In[10]:


spam_dataset.duplicated().sum()


# ##### Remove Duplicate Values

# In[11]:


spam_dataset = spam_dataset.dropna()
spam_dataset = spam_dataset.drop_duplicates(keep='first')
spam_dataset.head()


# ##### Final Shape of Dataset

# In[12]:


print(f"After removing duplicate values. The final shape of dataset : {spam_dataset.shape}")


# 3. #### EDA aka Exploratory Data Analysis
#         The goal is to investigating data, understanding patterns, checking (anomalies, outliers), summarize key structure before applying any types of machine learning or statistical models.
# 

# In[13]:


values = spam_dataset['label'].value_counts()
total = values[0] + values[1]

percentage_0 = (values[0] / total) * 100
percentage_1 = (values[1] / total) * 100

print(f"Percentage of target (ham) : {percentage_0}")
print(f"Percentage of target (spam) : {percentage_1}")


# #### Pie Chart (Email Classification)

# In[14]:


plt.figure(figsize=(12, 7)) # i know i 'm taking huge figure but i like that

colors = ['#FF5733', '#33FF57'] # red-oraney and green are my colors
myexplode = [0, 0.1]    # define explode parameter to create a gap between slices by 10%
mylabels = ['ham', 'spam']  # ham and spam are categories


plt.pie(
    values,
    startangle=90,  # set angle
    autopct='%0.2f%%',  # add percentage value
    labels=mylabels, # labeling pie with ham and spam
    explode=myexplode, # explode ham with 10%
    colors=colors,  # add your colors
    shadow=True, # add shadow
    textprops={'fontsize': 14, 'fontweight': 'bold'}    # add fontsize and weight
    )

plt.axis('equal')   #set axis equal so that pie looks circle
plt.title("Email Classification", fontweight="bold", fontsize=16)
plt.show()


# #### Summary for Text, Chars, Sentences Length

# In[15]:


import re
def count_characters(text):
    words = re.findall(r'[a-zA-Z]', text)
    icount = 0
    for w in words:
        if w.isdigit():
            continue
        icount = icount + 1
    return icount


# In[16]:


spam_dataset['num_characters'] = spam_dataset['message'].apply(count_characters)
spam_dataset['num_words'] = spam_dataset['message'].apply(lambda x: len(nltk.word_tokenize(x)))
spam_dataset['num_sentences'] = spam_dataset['message'].apply(lambda x: len(nltk.sent_tokenize(x)))

spam_dataset[['num_characters', 'num_words', 'num_sentences']].describe()


# ##### Summary Statistics for the legitimate message (emails)

# In[17]:


spam_dataset[spam_dataset['label'] == 0 ][['num_characters', 'num_sentences', 'num_words']].describe()


# ##### Summary Statistics for the spam messages (email)

# In[18]:


spam_dataset[spam_dataset['label'] == 1][['num_characters', 'num_sentences', 'num_words']].describe()


# In[19]:


spam_dataset[['label', 'num_characters', 'num_sentences', 'num_words']].corr()


# In[20]:


corr_matrix = spam_dataset[['label', 'num_characters', 'num_words', 'num_sentences']].corr()

sns.set(font_scale=1.2)
sns.heatmap(corr_matrix, annot=True, fmt='0.2f', linewidths=0.5, cmap='coolwarm')

plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')

plt.xticks(rotation=45)
plt.show()


# #### Data Text Preprocessing
#         Before data preprocessing let's understand how the spam are introduce in the mail and what type of the spam can be added in the email.

# In[21]:


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

text = """Visit https://example.com for more info
        Contact us at support@company.com or call 555-0123
        The price is $29.99 for 10 items
        Check out www.github.com and email me at user@domain.org
        Regular text without special patterns"""

text, tokens = custom_tokenizer(text)
print(tokens)
print(text)


# In[22]:


def text_representation(text):

    # tokenize the word
    tokens = word_tokenize(text)

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(tokens)
    feature_names = tfidf.get_feature_names_out()

    print("TF IDF Vectorizer: ")
    print(f"matrix : {tfidf_matrix}")
    print(f"features : {feature_names}")

text_representation(text)


# In[23]:


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

text = """
Subject: 🎉 CONGRATULATIONS! You've Won $1,000,000! 🎉
From: winner-notification@lottery-international.biz
Body:
DEAR LUCKY WINNER!!!
CONGRATULATIONS! You have been selected as the GRAND PRIZE WINNER of our International Email Lottery! Your email address was randomly selected from millions of addresses worldwide.
YOU HAVE WON: $1,000,000 USD!!!
To claim your prize, you MUST respond within 24 HOURS with the following information:

Full Name
Address
Phone Number
Bank Account Details
Copy of ID/Passport

URGENT: Send processing fee of $500 via Western Union to claim your winnings immediately!
Contact our claims agent: Mr. Johnson Smith
Email: claims.agent.smith@totally-legit-lottery.com
Phone: +234-555-7845
This offer expires TOMORROW! Don't miss this once-in-a-lifetime opportunity!
Best Regards,
International Lottery Commission
"""
processed_tokens = text_preprocessing(text)
print(processed_tokens)


# In[24]:


spam_dataset['cleaned_message'] = spam_dataset['message'].apply(text_preprocessing)
spam_dataset.head()


# In[25]:


# calculating the length of the cleaned message
spam_dataset['cleaned_message'].apply(lambda x: len(x.split(" ")))


# In[26]:


print(f"length of the cleaned preprocessed message : {len(spam_dataset['cleaned_message'])}")


# #### Tokenization

# In[27]:


tokenized_docs = [word_tokenize(doc) for doc in spam_dataset['cleaned_message']]
tokenized_docs[:10]


# In[28]:


print(f"length of the tokenized documents : {len(tokenized_docs)}")


# #### Creating Doc2Vec Document embeddings

# In[29]:


tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(tokenized_docs)]
tagged_data[:10]


# In[30]:


print(f"length of the tagged documents : {len(tagged_data)}")


# In[86]:


# training the doc2vec model.
model = Doc2Vec(vector_size=200, min_count=2, epochs=30)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)


# In[88]:


model.save("doc_2_vec.model")


# In[60]:


# get the vectors for the document.
document_vectors = [model.infer_vector(doc) for doc in tokenized_docs]


# In[61]:


for i, doc in enumerate(spam_dataset['cleaned_message']):
  print(f" original doc {i+1} : {doc[:100]}")
  print(document_vectors[i])
  print()


# In[62]:


print(f"length of the document embedings : {len(document_vectors)}")
print(f"length of the labeled columns : {len(spam_dataset['label'])}")


# #### Splitting the Dataset into the Training and Testing Set

# In[63]:


X_train, X_test, y_train, y_test = train_test_split(document_vectors, spam_dataset['label'], test_size=0.2, random_state=42)


# In[64]:


X_train[:10]


# In[65]:


# shapes of the training and test sets
print(f"Shape of the X train set : {len(X_train)}")
print(f"Shape of the X test set: {len(X_test)}")
print(f"Shape of the y train set: {len(y_train)}")
print(f"Shape of the y test set: {len(y_test)}")


# #### Model Training (Lstm)

# In[66]:


max_len = 200
max_words = 1024
# defining the lstm model
def lstm_model():
  inputs = Input(name='inputs', shape=[max_len])
  layer = Embedding(max_words, 128, input_length=max_len)(inputs)
  layer = LSTM(64)(layer)
  layer = Dense(256, name='FC1')(layer)
  layer = Activation('relu')(layer)
  layer = Dropout(0.5)(layer)
  layer = Dense(1, name='out_layer')(layer)
  layer = Activation('sigmoid')(layer)
  model = Model(inputs=inputs, outputs=layer)
  return model


# #### Converting Array to Matrix

# In[67]:


# Ensure they are NumPy arrays
import numpy as np
X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)


# #### summary and compiling

# In[69]:


model = lstm_model()
model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.00001)
model.fit(X_train, y_train, batch_size=128, epochs=50, validation_split=0.2, verbose=1, callbacks=[early_stopping])


# #### Visualizing Accuracy and Loss

# In[70]:


loss, acc = model.evaluate(X_train, y_train)
print(f"loss of Training is {loss:.2f}")
print(f"accuracy of Training is {acc:.2f}")


# In[71]:


hist = model.history.history
plt.plot(hist['accuracy'], label="Train Accuracy")
plt.plot(hist['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[72]:


plt.plot(hist['loss'], label="Train Loss")
plt.plot(hist['val_loss'], label='Validation loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[73]:


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

pd.DataFrame(y_pred[:10], columns=['Prediction'])


# In[74]:


spam = 0
ham = 0
for i in y_pred:
  if i == 1:
    spam += 1
  else:
    ham += 1
print(f"length of possible prediction spam: {spam}")
print(f"length of possible prediction ham: {ham}")


# #### Visualizing Confusion Matrix

# In[75]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[76]:


cm = confusion_matrix(y_pred, y_test)
print(f"confusion matrix \n {cm}")


# In[77]:


plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, cbar=True, cmap="viridis")
plt.title("confusion Matrix", fontsize=14)
plt.xlabel("Predicted labels", fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.show()


# In[78]:


print("Classification report \n")
print(classification_report(y_pred, y_test))


# #### Single Prediction

# In[79]:


input_text =  X_test[8]
print(f"before expanding dims : {input_text, input_text.shape}")
input_text = np.expand_dims(input_text, axis=0)
print(f"after expanding the col dims : {input_text, input_text.shape}")

# make the prediction
prediction = model.predict(input_text)

if prediction[0][0] > 0.5:
  print("Ham")
else:
  print("Spam")


# In[81]:


# saving model pkl 
model.save('saved_email_classification.keras')

