# NLPSimilarity

Precog Task 1

## Directory Structure

```
├── bonus.ipynb
├── data
│ ├── SimLex-999
│ │ ├── README.txt
│ │ └── SimLex-999.txt
│ └── sv-en
│ └── euro.txt
├── models
│ ├── doc2vec
│ ├── doc2vecsent
│ ├── GoogleNews-vectors-negative300.bin.gz
│ ├── word2vec_model_trained
│ ├── word2vec_model_trained.syn1neg.npy
│ └── word2vec_model_trained.wv.vectors.npy
├── phrases.ipynb
├── README.md
├── sentences.ipynb
├── words_constrained.ipynb
└── words_unconstrained.ipynb
```

## Commands to run

    Clone repository

    Extract moels zip folder

    run corresponding notebooks for each subtask

### Dependency Libraries

This project uses the following Python libraries:

Gensim: For training word embedding models.

Installation:

    pip install gensim

NLTK (Natural Language Toolkit): For text processing tasks.

Installation:

    pip install nltk

scikit-learn: For machine learning algorithms and evaluation metrics.

Installation:

    pip install scikit-learn

Numpy: For numerical computations.

Installation:

    pip install numpy

Datasets: For huggingface datasets

Installation:

    pip install datasets

Spacy: For computing linguistic features for preprocessing

Installation:

    pip install spacy

## Approach

Approach for Word Similarity Task

To predict the similarity between pairs of words, the following approach is followed:

Loading and Preprocessing the Corpus:

- The training data is loaded from a corpus file (corpus_path).
- The corpus is preprocessed to tokenize sentences and convert them into lowercase.

Training Word2Vec Model:

- The Word2Vec model from the Gensim library is used to train word embeddings.
- Parameters for the Word2Vec model training such as vector_size, window, min_count, and workers are defined.

Word Similarity Calculation:

- The word_similarity() function calculates the cosine similarity between embeddings of two input words.
- If both words are in the vocabulary, their embeddings are retrieved and cosine similarity is computed.
- If any of the words are not found in the vocabulary, the function returns -1.

Evaluation on SimLex-999 Dataset:

- The SimLex-999 test set file is read (simlex_path).
- Relevant columns (word pairs and SimLex-999 ratings) are extracted from the dataset.
- For each word pair, the similarity score is calculated using the trained Word2Vec model.
- Model similarities and SimLex-999 ratings are collected for evaluation.

Model Evaluation:

- The Spearman correlation coefficient is calculated to measure the correlation between model predictions and SimLex-999 ratings.
- The Spearman correlation evaluates how well the model's similarity scores align with human-rated word similarities in the SimLex-999 dataset.

Approach for Phrase Similarity Task

`get_phrase_embedding(phrase, word_embeddings)`: This function takes a phrase as input, tokenizes it, and calculates the average embedding of the constituent words using pre-trained word embeddings. If any word is not found in the embeddings, it returns a zero vector.

`phrase_similarity(phrase1, phrase2, word_embeddings)`: This function computes the cosine similarity between the embeddings of two phrases using the get_phrase_embedding function.

`cosine_phrase_similarity(phrase1, phrase2, word_embeddings, threshold=0.5)`: This function predicts whether two phrases are similar based on a cosine similarity threshold. It returns 1 if the similarity score is above the threshold, otherwise 0.

Also used Doc2Vec model.

Approach for Sentence Similarity Task using Doc2Vec

**Data Preparation:**

- For the training and development splits, tagged document objects are created for each pair of preprocessed sentences. Each tagged document contains the preprocessed sentence text along with a unique identifier.

**Model Training:**

- The Doc2Vec model is initialized with a vector size of 300 and trained for 30 epochs.
- The vocabulary is built using the tagged document objects from the training split, and the model is trained on these documents.

**Similarity Computation:**

- A function is defined to compute the similarity score between pairs of sentences using the trained Doc2Vec model.
- For each pair of sentences in the test set, the similarity score is calculated using the Doc2Vec model's `infer_vector` method.

**Classification:**

- Based on the computed similarity score for each pair of sentences, a binary classification label is assigned: 1 if the similarity score is greater than or equal to 0.5, and 0 otherwise.

**Evaluation:**

- The predicted labels are compared against the ground truth labels from the test set to calculate the accuracy of the model.
- The accuracy score is reported as the performance metric for the sentence similarity task.

This approach utilizes the Doc2Vec model to learn fixed-length embeddings for sentences, capturing semantic information that enables computing similarity scores between pairs of sentences. The computed similarity scores are then used for binary classification to determine whether the sentences are similar or dissimilar. Finally, the accuracy of the model's predictions is evaluated on the test set to assess its performance.

Approach for Sentence Similarity using Word2Vec

1. **Sentence Embedding Extraction**:

   - A function `get_sentence_embedding(sentence)` is defined to compute the embedding for a given sentence.
   - The function retrieves word embeddings for each word in the sentence using a pre-trained word2vec model (`word2vec_model_pretrained`).
   - If embeddings are found for any words in the sentence, they are averaged to obtain the sentence embedding. Otherwise, a zero vector is returned.

2. **Similarity Computation**:

   - Another function `compute_similarity(sentence1, sentence2)` is defined to compute the similarity score between two sentences.
   - It calculates the cosine similarity between the embeddings of the two sentences obtained using `get_sentence_embedding()`.

3. **Prediction and Classification**:

   - A function `predict_similarity(sentence1, sentence2, threshold=0.5)` is defined to predict whether two sentences are similar based on a given threshold.
   - It computes the cosine similarity between the embeddings of the two sentences and classifies them as similar if the similarity score is above the threshold, otherwise as dissimilar.

4. **Prediction on Test Data**:

   - The `compute_similarity()` function is applied to each pair of preprocessed sentences in the test dataset to obtain the predicted similarity score.
   - Based on the predicted similarity score, a binary label is assigned to each pair of sentences using a threshold of 0.5.

5. **Evaluation**:
   - The predicted labels are compared against the ground truth labels from the test set to calculate the accuracy of the model.
   - The accuracy score is reported as the performance metric for the sentence similarity task.

This approach leverages pre-trained word embeddings to compute sentence embeddings and then computes similarity scores between pairs of sentences based on cosine similarity. The predicted similarity scores are then used to classify sentences as similar or dissimilar, and the accuracy of the model's predictions is evaluated on the test set.
