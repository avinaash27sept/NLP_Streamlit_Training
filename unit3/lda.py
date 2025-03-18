import pandas as pd
import streamlit as st
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt


# Create a sample dataset
data1= {
    'text_column': [
        'The cat sat on 1234 the mat.',
        'Dogs are great pets.',
        'I love to play          football.',
        'Data science is an interdisciplinary field.',
        'Python is a great""""""" programming language.',
        'Machine learning is a subset of artificial intelligence.',
        'Artificial intelligence and machine learning are popular topics.',
        'Deep learning is a type of machine learning.',
        'Natural language processing involves analyzing text data.',
        'I enjoy hiking and outdoor activities.'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data1)
st.info("Before Cleaning")
st.data_editor(df)

# Save DataFrame to CSV
df.to_csv('sample_dataset.csv', index=False)



# Download NLTK stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Load data
data = pd.read_csv('sample_dataset.csv')  # Load the sample dataset

# Preprocess the text data
def preprocess_text(text):
    text = re.sub('\s+', ' ', text)  # Remove extra spaces
    text = re.sub('\S*@\S*\s?', '', text)  # Remove emails
    text = re.sub('\'', '', text)  # Remove apostrophes
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabet characters
    text = text.lower()  # Convert to lowercase
    return text

data['cleaned_text'] = data['text_column'].apply(preprocess_text)  # Replace 'text_column' with your column name

st.info("After Cleaning")
st.dataframe(data)
# Tokenize and remove stopwords
def tokenize(text):
    tokens = gensim.utils.simple_preprocess(text, deacc=True)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

data['tokens'] = data['cleaned_text'].apply(tokenize)

st.info("After Tokenization")
st.dataframe(data)

# Lemmatization using spaCy
nlp = spacy.load('en_core_web_sm')
def lemmatize(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

data['lemmas'] = data['tokens'].apply(lemmatize)
st.info("After Lematization")
st.dataframe(data)

# Create dictionary and corpus
id2word = corpora.Dictionary(data['lemmas'])
texts = data['lemmas']
corpus = [id2word.doc2bow(text) for text in texts]

st.write(f"Dictionary",id2word)
st.write(f"texts",texts)
st.write(f"corpus",corpus)

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=5, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

# Print the topics
topics = lda_model.print_topics(num_words=5)
document = [lda_model.get_document_topics(item) for item in corpus]
for topic in topics:
    st.write(topic)

for doc in document :
    st.write(doc)
    


