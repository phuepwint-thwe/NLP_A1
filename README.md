# nlp-a1
- [Student Information](#student-information)
## Student Information
 - Name: Phue Pwint Thwe
 - ID: st124784
 ## Training Data

- **Corpus Source**: `nltk.datasets('reuters')`
- **Token Count (|C|)**: 241,109  
- **Vocabulary Size (|V|)**: 10,00  
- **Embedding Dimension**: 2  
- **Learning Rate**: 0.001  
- **Epochs**: 100 

*Training parameters are the same across all three models.*

## Model Comparison

| **Model**            | **Window Size** | **Training Loss** | **Training Time** | **Syntactic Accuracy** | **Semantic Accuracy** |
|-----------------------|-----------------|-------------------|-------------------|-------------------------|------------------------|
| Skipgram             | 2               | 9.65              | -     | 0.00%                  | 0.00%                 |
| Skipgram (NEG)       | 2               | 1.93              | -    | 0.00%                  | 0.00%                 |
| GloVe                | 2               | 0.00              |       | 0.00%                  | 0.00%                 |
| GloVe (Gensim)       | -               | -                 | -                 | 55%               | 54%                |

### Observations
1. **Training Loss**:
   - GloVe achieved the lowest loss of **0.00**, indicating faster convergence compared to Skipgram and Skipgram (NEG), which had losses of **9.65** and **1.93**, respectively.
   - Loss values reflect the optimization efficiency of the models.

2. **Accuracy**:
   - Semantic and syntactic accuracy for Skipgram, Skipgram (NEG), and GloVe (trained from scratch) was **0.00%**, suggesting they did not capture meaningful word relationships effectively.
   - GloVe (Gensim) demonstrated strong performance with **55% syntactic accuracy** and **54% semantic accuracy**, highlighting the advantages of pretraining on larger datasets.

3. **Training Time**:
   - Training times were not explicitly recorded but are expected to correlate with model complexity and convergence rates.

---

## Similarity Scores

| **Model**            | **Skipgram** | **Skipgram (NEG)** | **GloVe** | **GloVe (Gensim)** | 
|-----------------------|--------------|--------------------|-----------|--------------------|------------|
| **Spearman Correlation** | 0.85        | 1.93              | 0.26      | -0.01            | 

### Observations
1. **Skipgram (NEG)** achieved the highest Spearman correlation of **1.93**, which may indicate a high level of similarity between the embeddings and the human-judged dataset. However, this unusually high value could suggest overfitting or data-specific bias.
2. **Skipgram** performed reasonably well with a correlation of **0.85**, surpassing GloVe’s correlation of **0.26**.
3. **GloVe (Gensim)** showed a negative correlation (**-0.01**), likely due to inconsistencies between the pretrained model and the dataset used for similarity evaluation.

---

## Conclusion

Among the four models evaluated, **GloVe (Gensim)** emerged as the most effective, demonstrating strong performance in syntactic and semantic accuracy. Its success highlights the value of pretraining on large datasets, which enables it to capture more meaningful word relationships. In contrast, the models trained from scratch—Skipgram, Skipgram (NEG), and GloVe—struggled to achieve high accuracy due to the limited corpus size and vocabulary, despite consistent training parameters. This underscores the need for larger datasets and extended training to improve results. Overall, GloVe (Gensim) is well-suited for practical applications, while future efforts could focus on fine-tuning pretrained models or optimizing the training process for models built from scratch.
---