import streamlit as st
from nltk.stem import SnowballStemmer
st.title("Stemming")
st.write("Stemming is a method in text processing that eliminates prefixes and suffixes from words, transforming them into their fundamental or root form, The main objective of stemming is to streamline and standardize words, enhancing the effectiveness of the natural language processing tasks. The article explores more on the stemming technique and how to perform stemming in Python.")

tab1, tab2, tab3 = st.tabs(["Porter’s Stemmer", "Snowball Stemmer", "Snowball Stemmer"])

with tab1:
    st.header("Porter Stemmer")
    st.write("Hello")
   
with tab2:
    st.header("Snowball Stemmer")
    st.write("""The Snowball Stemmer, compared to the Porter Stemmer, is multi-lingual as it can handle non-English words. It supports various languages and is based on the ‘Snowball’ programming language, known for efficient processing of small strings.

    The Snowball stemmer is way more aggressive than Porter Stemmer and is also referred to as Porter2 Stemmer. Because of the improvements added when compared to the Porter Stemmer, the Snowball stemmer is having greater computational speed. """)
    st.subheader("Example code ")
    st.code("""from nltk.stem import SnowballStemmer

    # Choose a language for stemming, for example, English
    stemmer = SnowballStemmer(language='english')

    # Example words to stem
    words_to_stem = ['running', 'jumped', 'happily', 'quickly', 'foxes']

    # Apply Snowball Stemmer
    stemmed_words = [stemmer.stem(word) for word in words_to_stem]

    # Print the results
    print("Original words:", words_to_stem)
    print("Stemmed words:", stemmed_words)
    """)
    stemmer = SnowballStemmer(language='english')
    
    # Example words to stem
    words_to_stem = ['running', 'jumped', 'happily', 'quickly', 'foxes']
    
    # Apply Snowball Stemmer
    stemmed_words = [stemmer.stem(word) for word in words_to_stem]
    
    # Print the results
    st.write(f"Original words:", words_to_stem)
    st.write(f"Stemmed words:", stemmed_words)

    
with tab3:
    st.header("Snowball Stemmer")
   