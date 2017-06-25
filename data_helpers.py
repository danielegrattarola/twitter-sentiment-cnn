import numpy as np
import re
import random, csv

POS_DATASET_PATH = 'twitter-sentiment-dataset/tw-data.pos'
NEG_DATASET_PATH = 'twitter-sentiment-dataset/tw-data.neg'
VOC_PATH = 'twitter-sentiment-dataset/vocab.csv'
VOC_INV_PATH = 'twitter-sentiment-dataset/vocab_inv.csv'


def clean_str(string):
    """
    Tokenizes common abbreviations and punctuation, removes unwanted characters. 
    Returns the clean string.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r'(.)\1+', r'\1\1', string) 
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def sample_list(list, dividend):
    """
    Returns 1/dividend-th of the given list, randomply sampled. 
    """
    return random.sample(list, len(list)/dividend)


def load_data_and_labels(reduced_dataset):
    """
    Loads data from files, processes the data and creates two lists, one of
    strings and one of labels.
    Returns the lists. 
    """
    print "\tdata_helpers: loading positive examples..."
    positive_examples = list(open(POS_DATASET_PATH).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    print "\tdata_helpers: [OK]"
    print "\tdata_helpers: loading negative examples..."
    negative_examples = list(open(NEG_DATASET_PATH).readlines())
    negative_examples = [s.strip() for s in negative_examples]
    print "\tdata_helpers: [OK]"

    positive_examples = sample_list(positive_examples, reduced_dataset)
    negative_examples = sample_list(negative_examples, reduced_dataset)

    # Split by words
    x_text = positive_examples + negative_examples
    print "\tdata_helpers: cleaning strings..."
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    print "\tdata_helpers: [OK]"

    # Generate labels
    print "\tdata_helpers: generating labels..."
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    print "\tdata_helpers: [OK]"
    print "\tdata_helpers: concatenating labels..."
    y = np.concatenate([positive_labels, negative_labels], 0)
    print "\tdata_helpers: [OK]"
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest
    sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def pad_sentences_to(sentences, pad_to, padding_word="<PAD/>"):
    """
    Pads all sentences to the pad_to lenght. 
    Returns the padded senteces.
    """
    sequence_length = pad_to
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab():
    """
    Reads the vocabulary and its inverse mapping from the csv in the dataset
    folder.
    Returns a list with the vocabulary and the inverse mapping.
    """
    voc = csv.reader(open(VOC_PATH))
    voc_inv = csv.reader(open(VOC_INV_PATH))
    # Mapping from index to word
    vocabulary_inv = [x for x in voc_inv]
    # Mapping from word to index
    vocabulary = {x: i for x, i in voc}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    Returns the mapped lists. 
    """
    x = np.array([[vocabulary[word] for word in sentence]
                  for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def string_to_int(sentence, vocabulary, max_len):
    """
    Converts the given string to the corresponding string encoded in integers.
    Returns the encoded string.
    """
    # Reads dataset in order to create the vocabulary
    base = [sentence]
    base = [s.strip() for s in base]
    x_text = base
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    padded_x_text = pad_sentences_to(x_text, max_len)
    try: 
        x = np.array([[vocabulary[word] for word in sentence]
                      for sentence in padded_x_text])
        return x
    except KeyError, e:
        print "The following word is unknown to the network: %s" % str(e)
        quit()


def load_data(reduced_dataset):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(reduced_dataset)
    print "\tdata_helpers: padding strings..."
    sentences_padded = pad_sentences(sentences)
    print "\tdata_helpers: [OK]"
    print "\tdata_helpers: building vocabulary..."
    vocabulary, vocabulary_inv = build_vocab()
    print "\tdata_helpers: [OK]"
    print "\tdata_helpers: building processed datasets..."
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    print "\tdata_helpers: [OK]"
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
