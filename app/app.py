import streamlit as st
import numpy as np
import torch
import pickle
from numpy import dot
from numpy.linalg import norm
import torch.nn as nn
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# Import missing classes
from models.classes import Skipgram, SkipgramNeg  # Ensure the correct path to classes.py

# Prepare Corpus
@st.cache(allow_output_mutation=True)
def preprocess_corpus():
    # Load pre-saved corpus to avoid accessing the Reuters corpus repeatedly
    with open("corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    vocabs = list(set(corpus))[:10000]  # Limit vocab size to 10,000
    return corpus, vocabs

corpus, vocabs = preprocess_corpus()

# Load GloVe (Gensim) model
@st.cache(allow_output_mutation=True)
def load_gensim_model():
    import os
    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec

    # Use the correct base directory where the file is located
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../test"))  # Moves up one level to the 'test' folder
    glove_file = os.path.join(base_dir, "glove.6B.100d.txt")

    if not os.path.exists(glove_file):
        raise FileNotFoundError(f"GloVe file not found at {glove_file}")

    word2vec_output_file = glove_file + ".word2vec"
    glove2word2vec(glove_file, word2vec_output_file)
    return KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

gensim_model = load_gensim_model()

# Define GloVe model class
class Glove(nn.Module):
    def __init__(self, voc_size, emb_size, word2index):
        super(Glove, self).__init__()
        self.center_embedding = nn.Embedding(voc_size, emb_size)
        self.outside_embedding = nn.Embedding(voc_size, emb_size)
        self.center_bias = nn.Embedding(voc_size, 1)
        self.outside_bias = nn.Embedding(voc_size, 1)
        self.word2index = word2index

    def forward(self, center, outside, coocs, weighting):
        center_embeds = self.center_embedding(center)
        outside_embeds = self.outside_embedding(outside)
        center_bias = self.center_bias(center).squeeze(1)
        target_bias = self.outside_bias(outside).squeeze(1)
        inner_product = outside_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        loss = weighting * torch.pow(inner_product + center_bias + target_bias - coocs, 2)
        return torch.sum(loss)

    def get_vector(self, word):
        try:
            index = self.word2index[word]
        except KeyError:
            index = self.word2index["<UNK>"]
        word = torch.LongTensor([index])
        embed_c = self.center_embedding(word)
        embed_o = self.outside_embedding(word)
        embed = (embed_c + embed_o) / 2
        return embed.squeeze(0)

# Load models
@st.cache(allow_output_mutation=True)
def load_models():
    # Load GloVe model
    glove_args = pickle.load(open('models/glove.args', 'rb'))
    glove_model = Glove(**glove_args)
    glove_model.load_state_dict(torch.load('models/glove.model', map_location=torch.device('cpu')))

    # Skipgram
    skg_args = pickle.load(open('models/skipgram.args', 'rb'))
    model_skipgram = Skipgram(**skg_args)
    model_skipgram.load_state_dict(torch.load('models/skipgram.model', map_location=torch.device('cpu')))

    # Negative Sampling
    neg_args = pickle.load(open('models/negsampling.args', 'rb'))
    model_neg = SkipgramNeg(**neg_args)
    model_neg.load_state_dict(torch.load('models/negsampling.model', map_location=torch.device('cpu')))

    return glove_model, model_skipgram, model_neg

glove_model, skipgram_model, skipgram_neg_model = load_models()

# Cosine similarity function
def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def get_top_similar_words(model, word_input):
    try:
        if len(word_input.split()) == 1:
            # Use appropriate method for the model
            if hasattr(model, "get_vector"):
                # For Glove model
                word_embed = model.get_vector(word_input).detach().numpy().flatten()
            elif hasattr(model, "get_embed"):
                # For Skipgram and SkipgramNeg models
                word_embed = np.array(model.get_embed(word_input))
            else:
                return ["Model does not have a valid method to retrieve embeddings."]

            similarity_dict = {}
            for word in vocabs:
                if hasattr(model, "get_vector"):
                    # For Glove model
                    word_vector = model.get_vector(word).detach().numpy().flatten()
                elif hasattr(model, "get_embed"):
                    # For Skipgram and SkipgramNeg models
                    word_vector = np.array(model.get_embed(word))
                else:
                    continue
                similarity_dict[word] = cos_sim(word_embed, word_vector)

            sorted_words = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            return [f"{i+1}. {word} ({score:.8f})" for i, (word, score) in enumerate(sorted_words)]
        else:
            return ["The system can search with 1 word only."]
    except KeyError:
        return ["The word is not in my corpus. Please enter a new word."]

# Streamlit UI
st.title("NLP-A1 (Word Similarity Search Engine)")

# Input form
search_query = st.text_input("Enter a word to search:", "")

if st.button("Search"):
    if search_query:
        st.subheader(f"Search Results for: {search_query}")
        
        # GloVe Model Results
        st.write("### GloVe Model Results:")
        glove_output = get_top_similar_words(glove_model, search_query)
        st.write(glove_output)

        # Skipgram Model Results
        st.write("### Skipgram Model Results:")
        skipgram_output = get_top_similar_words(skipgram_model, search_query)
        st.write(skipgram_output)

        # Skipgram Negative Sampling Results
        st.write("### Skipgram with Negative Sampling Results:")
        skipgram_neg_output = get_top_similar_words(skipgram_neg_model, search_query)
        st.write(skipgram_neg_output)

        # GloVe (Gensim) Results
        st.write("### GloVe (Gensim) Results:")
        gensim_output = get_top_similar_words_gensim(gensim_model, search_query)
        st.write(gensim_output)
    else:
        st.error("Please enter a valid word.")