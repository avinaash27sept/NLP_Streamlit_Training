import streamlit as st
pages = {
    "Introduction to Natural Language Processing": [
        st.Page("unit1\\token.py", title="Tokenization"),
        st.Page("unit1\\stem.py", title="Stemming"),
        st.Page("unit1\\lemma.py", title="Lematization"),
        st.Page("unit1\\pos.py", title="Part of speech tagging"),

    ],
    "Language Modelling": [
         st.Page("unit3\\lda.py", title="Latent Dirichlet Allocation")

    ]
}
pg = st.navigation(pages)
pg.run()
