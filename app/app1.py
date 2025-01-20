import streamlit as st
import numpy as np
import pickle
import torch
from models.classes import Skipgram, SkipgramNeg, Glove, Gensim
import nltk
from nltk.corpus import reuters

# Download necessary NLTK data
nltk.download('reuters')
nltk.download('punkt')

# Load and preprocess Reuters dataset
def load_corpus(max_vocab=10000):
    words = [word.lower() for word in reuters.words()]
    vocab = list(set(words))[:max_vocab]
    sentences = [[word for word in sentence if word in vocab]
                 for fileid in reuters.fileids() for sentence in reuters.sents(fileid)]
    return sentences, vocab

corpus, vocabulary = load_corpus(max_vocab=10000)

# Load Models
@st.cache(allow_output_mutation=True)
def load_models():
    try:
        # Skipgram
        skg_args = pickle.load(open('models/skipgram.args', 'rb'))
        model_skipgram = Skipgram(**skg_args)
        model_skipgram.load_state_dict(torch.load('models/skipgram.model', map_location=torch.device('cpu')))

        # SkipgramNeg
        neg_args = pickle.load(open('models/negsampling.args', 'rb'))
        model_neg = SkipgramNeg(**neg_args)
        model_neg.load_state_dict(torch.load('models/negsampling.model', map_location=torch.device('cpu')))

        # Glove
        glove_args = pickle.load(open('models/glove.args', 'rb'))
        model_glove = Glove(**glove_args)
        model_glove.load_state_dict(torch.load('models/glove.model', map_location=torch.device('cpu')))

        # Gensim
        load_model = pickle.load(open('models/gensim.model', 'rb'))
        model_gensim = Gensim(load_model)

        return {
            'skipgram': model_skipgram,
            'neg': model_neg,
            'glove': model_glove,
            'gensim': model_gensim,
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}

models = load_models()
model_names = {'skipgram': 'Skipgram', 'neg': 'SkipgramNeg', 'glove': 'Glove', 'gensim': 'Glove (Gensim)'}

# Function to compute cosine similarity
def find_closest_indices_cosine(vector_list, single_vector, k=10):
    vector_list = np.array(vector_list)
    single_vector = np.array(single_vector)

    # Handle edge cases for empty or inconsistent inputs
    if vector_list.shape[1] != single_vector.shape[0]:
        raise ValueError("Mismatched dimensions between corpus embeddings and query embedding.")

    similarities = np.dot(vector_list, single_vector) / (
        np.linalg.norm(vector_list, axis=1) * np.linalg.norm(single_vector)
    )
    similarities = np.nan_to_num(similarities)  # Handle NaNs
    top_indices = np.argsort(similarities)[-k:][::-1]
    return top_indices

# Streamlit App UI
st.title("NLP A1: Search Engine")

# Sidebar for model selection
st.sidebar.title("Select NLP Model")
selected_model = st.sidebar.selectbox("Choose a model:", list(model_names.keys()), format_func=lambda x: model_names[x])

# Query input
query = st.text_input("Enter your search query:", placeholder="Type your query here.")

# Submit button
if st.button("Search"):
    if query:
        try:
            model = models[selected_model]

            # Compute query embedding
            qwords = query.split()
            qwords_embeds = np.array([model.get_embed(word) for word in qwords if word in vocabulary])
            if len(qwords_embeds) == 0:
                st.error("No valid words in the query match the vocabulary.")
                st.stop()

            qsentence_embeds = np.mean(qwords_embeds, axis=0)

            # Compute corpus embeddings
            corpus_embeds = []
            for each_sent in corpus:
                words_embeds = np.array([model.get_embed(word) for word in each_sent if word in vocabulary])
                if len(words_embeds) > 0:
                    sentence_embeds = np.mean(words_embeds, axis=0)
                    corpus_embeds.append(sentence_embeds)

            corpus_embeds = np.array(corpus_embeds)

            # Find the closest sentences
            result_idxs = find_closest_indices_cosine(corpus_embeds, qsentence_embeds)
            results = [' '.join(corpus[idx]) for idx in result_idxs]

            # Display results
            st.subheader(f"Search Results using {model_names[selected_model]}:")
            for idx, sentence in enumerate(results, 1):
                st.write(f"{idx}. {sentence}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query.")