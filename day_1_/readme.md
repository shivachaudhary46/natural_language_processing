#  Named Entity Recognition. 
Today I, learned named entity recognition model which focus on identifying and categorizing impportant information known as entities in text. These entities can be names of people, places, organizations, dates, etc. 

# Understanding NER 
NER helps to identify specific information and sort them into categories. It plays important role in enhancing other NLP tasks like Part of Speech tagging and parsing. Examples of common Entity types: 
- Person Name : Shiva Chaudhary 
- Organization : Nobel Learning PBC
- Location : Nepal, Kathmandu
- Dates and TImes : 5th oct 2005
- Quantities and Percentages : 50 %, $100

It helps in clearing ambguity by analyzing the surrounding words, structure of sentence and overall context to make the correct classification. It means context can change based on entity's meaning. for example: 
- Amazon sales increase last week. (means company)
- Amazon is the largest tropical forest. (means forest)

# Working of Named Entity Recognition (NER)
1. ***Analyzing the Text:*** first preprocess the text and locate the words that clearly defines entities. Entities are (Name, organization, Location, Dates, Times, Quantities)

2. ***Finding sentence Boundaries:*** Identifying and finding the boundaries of phrases with punctuation and spaces. It helps to maintain meaning of context as well. 

3. ***Tokenizing and Part of Speech Tagging:*** Text is broken into tokens (words) and each token is tagged with Part of Speech and grammatical role. it provide important clues for identifying entities. 

4. ***Entity detection and classification:*** 
Tokens or group of tokens that match patterns of known entities are recognized and classified into categories like (Person Name, Organization, values)

5. ***Machine Learning Model:*** 
The model will be trained on the labeled data and they improve over time learning patterns and relationship between words. 

6. ***Adapting to the new contexts:***
A well trained model can generate to different langugaes, styles and unseen type of context.

I learned to implement the Named Entity Recognition through Geeks for Geeks, here's link, 
https://www.geeksforgeeks.org/nlp/named-entity-recognition/