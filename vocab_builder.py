from data_helpers import *
import itertools
from collections import Counter
import csv

"""
Reads the dataset and creates two csv files: one with the vocabulary used in the dataset, and one with the vocabulary's integer mapping (sorted from most to least used).
"""


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


# Load and preprocess data
print 'vocab_builder: loading...'
sentences, labels = load_data_and_labels(1)  # 1 is passed so that load_data_and_labels() will parse the whole dataset
print 'vocab_builder: padding...'
sentences_padded = pad_sentences(sentences)
print 'vocab_builder: building vocabularies...'
vocabulary, vocabulary_inv = build_vocab(sentences_padded)

print 'vocab_builder: writing csv...'
voc = csv.writer(open('twitter-sentiment-dataset/vocab.csv', 'w'))
voc_inv = csv.writer(open('twitter-sentiment-dataset/vocab_inv.csv', 'w'))

for key, val in vocabulary.items():
    voc.writerow([key, val])
for val in vocabulary_inv:
    voc_inv.writerow([val])
