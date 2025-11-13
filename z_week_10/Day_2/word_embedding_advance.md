### 📘 Introduction

**What are Word Embeddings?**

Word embeddings are a cornerstone of modern Natural Language Processing (NLP). They are dense numerical representations of words in the form of real-valued vectors in a multi-dimensional space. This technique allows words with similar meanings to have similar vector representations. Essentially, word embeddings capture the semantic and syntactic relationships between words, enabling computers to process and understand language in a more nuanced, human-like way.

**Why Do We Need Embeddings?**

Computers do not understand text in its raw form; they require numerical input. Traditional methods for converting text to numbers, such as one-hot encoding, create very high-dimensional and sparse vectors that fail to capture the relationships between words. Word embeddings solve this by representing words in a lower-dimensional space, which is more computationally efficient and, crucially, encodes semantic meaning. This transformation is a fundamental step for a wide range of NLP tasks, including text classification, machine translation, and sentiment analysis.

**The Core Principle:**

The underlying idea for many word embedding techniques is the **distributional hypothesis**, famously summarized by John Rupert Firth's quote: "You shall know a word by the company it keeps." This means that words that appear in similar contexts are likely to have similar meanings and, therefore, should have similar vector representations.

### 🔍 Deep Explanation

#### **Early Approaches and Their Limitations**

Before the advent of dense embeddings, several methods were used to represent words numerically.

*   **One-Hot Encoding:** In this method, each word in a vocabulary is represented by a unique integer. This integer is then converted into a binary vector that is all zeros except for a single '1' at the index corresponding to that word.
    *   **Problems:**
        *   **High Dimensionality:** The length of the vector is equal to the size of the vocabulary, which can be very large, leading to the "curse of dimensionality."
        *   **Sparsity:** The vectors are mostly composed of zeros, making them computationally inefficient.
        *   **No Semantic Relationship:** The vectors are orthogonal, meaning the dot product between any two distinct word vectors is zero. This implies that there is no inherent similarity between words like "cat" and "dog," which is semantically incorrect.

*   **Bag-of-Words (BoW) and TF-IDF:** These methods represent entire documents as vectors.
    *   **BoW:** Counts the frequency of each word in a document, disregarding grammar and word order.
    *   **TF-IDF (Term Frequency-Inverse Document Frequency):** A more advanced statistical measure that evaluates the relevance of a word to a document in a collection of documents. While an improvement over simple frequency counts, these methods still do not fully capture the semantic relationships between words.

#### **Static Word Embeddings**

Static embeddings assign a single, fixed vector to each word, regardless of the context in which it appears. These embeddings are typically pre-trained on large text corpora.

*   **Word2Vec (Google, 2013):** A pioneering neural network-based model that learns word associations from a large corpus of text. It comes in two main architectures:
    *   **Continuous Bag-of-Words (CBOW):** This model predicts a target word based on its surrounding context words. It is generally faster to train and performs well for frequent words.
    *   **Skip-Gram:** This model does the opposite of CBOW; it predicts the surrounding context words given a target word. It works well with small amounts of training data and is effective at representing rare words.

*   **GloVe (Global Vectors for Word Representation - Stanford, 2014):** GloVe combines the advantages of global matrix factorization and local context window methods. It is trained on aggregated global word-word co-occurrence statistics from a corpus, which allows it to capture global context more effectively than Word2Vec.

*   **FastText (Facebook, 2016):** An extension of Word2Vec, FastText addresses a key limitation of Word2Vec and GloVe: handling out-of-vocabulary (OOV) words. It achieves this by representing each word as a bag of character n-grams. This allows it to generate embeddings for unseen words by summing the embeddings of their constituent n-grams. This makes it particularly effective for morphologically rich languages.

#### **Contextual Word Embeddings**

The major limitation of static embeddings is their inability to handle polysemy (words with multiple meanings). Contextual embeddings address this by generating dynamic word representations that depend on the surrounding text.

*   **ELMo (Embeddings from Language Models, 2018):** ELMo generates deep contextualized word representations using a bidirectional LSTM (Long Short-Term Memory) network. The embeddings are a function of the entire input sentence, so the representation for a word like "bank" will be different in "river bank" versus "money bank."

*   **BERT (Bidirectional Encoder Representations from Transformers, 2018):** A landmark model that uses the Transformer architecture to learn contextual relations between words in a text. BERT's key innovation is its use of a masked language model objective, which allows it to learn deep bidirectional representations by predicting randomly masked words in a sequence.

*   **GPT (Generative Pre-trained Transformer):** Another Transformer-based model, GPT is trained using a causal language modeling objective, which involves predicting the next word in a sequence. This makes it particularly adept at text generation tasks.

### 💡 Examples

#### **Mathematical Example: Semantic Arithmetic**

A famous example demonstrating the semantic properties of word embeddings is the vector arithmetic:

`vector("king") - vector("man") + vector("woman") ≈ vector("queen")`

This shows that the embeddings have captured the gender relationship and can perform logical reasoning through simple vector operations.

#### **Coding Example: Keras Embedding Layer**

The `Embedding` layer in Keras is a powerful tool for working with word embeddings in deep learning models.

**1. Training an Embedding Layer from Scratch:**

This approach is suitable when you have a domain-specific dataset and want to learn embeddings tailored to your task.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
import numpy as np

# Assume you have preprocessed your text into integer sequences
vocab_size = 10000  # Size of your vocabulary
max_length = 50     # Maximum length of your input sequences

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

**2. Using Pre-trained GloVe Embeddings:**

This is a common practice when you want to leverage the knowledge captured from a large corpus.

```python
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# 1. Load GloVe embeddings
embeddings_index = {}
with open('glove.6B.100d.txt', encoding="utf-8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

# 2. Create an embedding matrix for your vocabulary
tokenizer = Tokenizer() # Assume this is fit on your training text
word_index = tokenizer.word_index
num_tokens = len(word_index) + 1
embedding_dim = 100
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 3. Create the Keras Embedding layer
embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=False,  # Freeze the pre-trained weights
)

# 4. Build your model
model = Sequential([
    embedding_layer,
    Flatten(),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

### 🧩 Related Concepts

*   **Vector Space Models (VSMs):** A broader algebraic model for representing text documents as vectors of identifiers.
*   **Dimensionality Reduction:** Techniques like t-SNE (t-Distributed Stochastic Neighbor Embedding) are often used to visualize high-dimensional word embeddings in 2D or 3D space.
*   **Cosine Similarity:** A common metric used to measure the similarity between two word vectors. It calculates the cosine of the angle between them.
*   **Transfer Learning:** The practice of using pre-trained embeddings is a form of transfer learning, where knowledge gained from one task is applied to a different but related problem.

### 📝 Assignments / Practice Questions

1.  **MCQ:** What is the primary advantage of FastText over Word2Vec and GloVe?
    *   A) It is faster to train.
    *   B) It can handle out-of-vocabulary words.
    *   C) It produces lower-dimensional vectors.
    *   D) It uses a Transformer architecture.

2.  **MCQ:** Which of the following models generates contextualized word embeddings?
    *   A) Word2Vec
    *   B) GloVe
    *   C) BERT
    *   D) Skip-Gram

3.  **Short Question:** Explain the difference between the CBOW and Skip-Gram architectures in Word2Vec. Which one is generally better for representing rare words?

4.  **Short Question:** Why is one-hot encoding not ideal for representing words in NLP tasks? Mention at least two reasons.

5.  **Problem-Solving Task:** You are given the following word vectors: `v_apple`, `v_fruit`, `v_car`, `v_vehicle`. How would you expect the cosine similarity to compare between (`v_apple`, `v_fruit`) and (`v_apple`, `v_car`)? Justify your answer.

6.  **Case Study:** A company wants to build a sentiment analysis model for customer reviews of their new electronic product. They have a relatively small dataset of 5,000 reviews. Would you recommend they train their own word embeddings from scratch or use pre-trained embeddings like GloVe? Explain the trade-offs of each approach in this scenario.

### 📈 Applications

Word embeddings have become integral to a wide array of NLP applications across various industries:

*   **Text Classification and Sentiment Analysis:** By converting text into meaningful vectors, embeddings serve as powerful features for machine learning models to classify documents (e.g., spam detection, topic categorization) and determine sentiment (positive, negative, neutral).
*   **Machine Translation:** Embeddings help in mapping words from a source language to a target language by representing them in a language-agnostic way.
*   **Information Retrieval and Search Engines:** By understanding the semantic meaning of queries, search engines can retrieve more relevant documents, even if they don't contain the exact keywords.
*   **Named Entity Recognition (NER):** Contextual embeddings are particularly effective at disambiguating entities, such as distinguishing between "Apple" the company and "apple" the fruit.
*   **Recommendation Engines:** E-commerce platforms can recommend products based on semantic similarity derived from product descriptions and user reviews.
*   **Question Answering and Chatbots:** Embeddings help systems understand the intent behind user questions and provide relevant answers by matching the query with information in a knowledge base.

### 🔗 Related Study Resources

*   **Original Papers (Google Scholar):**
    *   Word2Vec: [Efficient Estimation of Word Representations in Vector Space](https://scholar.google.com/scholar?q=efficient+estimation+of+word+representations+in+vector+space)
    *   GloVe: [GloVe: Global Vectors for Word Representation](https://scholar.google.com/scholar?q=glove+global+vectors+for+word+representation)
    *   BERT: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://scholar.google.com/scholar?q=bert+pre-training+of+deep+bidirectional+transformers)

*   **Documentation and Tutorials:**
    *   **TensorFlow:** [Word embeddings tutorial](https://www.tensorflow.org/text/guide/word_embeddings)
    *   **Keras:** [Using pre-trained word embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings/)

*   **Online Courses:**
    *   **Coursera:** [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing) by DeepLearning.AI
    *   **Stanford University:** [CS224n: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)

### 🎯 Summary / Key Takeaways

*   **What they are:** Dense vector representations of words that capture semantic meaning.
*   **Why they matter:** They enable machines to understand text by converting words into a numerical format that preserves their relationships.
*   **How they are learned:** Based on the distributional hypothesis ("a word is known by the company it keeps").
*   **Static vs. Contextual:**
    *   **Static (Word2Vec, GloVe, FastText):** One fixed vector per word. Computationally efficient but cannot handle words with multiple meanings.
    *   **Contextual (ELMo, BERT, GPT):** Dynamic vectors that change based on the sentence context, providing a more nuanced understanding of language.
*   **Key Models:**
    *   **Word2Vec:** Predicts context words (Skip-Gram) or a target word from its context (CBOW).
    *   **GloVe:** Learns from global word co-occurrence statistics.
    *   **FastText:** Uses subword information to handle out-of-vocabulary words.
    *   **BERT:** A powerful Transformer-based model that learns deep bidirectional representations.
*   **Practical Use:** The Keras `Embedding` layer allows for both training custom embeddings and leveraging powerful pre-trained models like GloVe.
