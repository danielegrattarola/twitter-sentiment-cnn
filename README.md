# Twitter sentiment classification by Daniele Grattarola
This is a TensorFlow implementation of a convolutional neural
network (CNN) to perform sentiment classification on tweets.

This code is meant to have an educational value, to train the model by
yourself and play with different configurations, and was not developed
to be deployed as-is (although it has been used in [professional
contexts](https://linkedin.com/pulse/real-time-twitter-sentiment-analytics-tensorflow-spring-tzolov/)).
The dataset used for training is taken from [here](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/)
(someone reported to me that the link to the dataset appears to be dead
sometimes, so `dataset_downloader.py` **might** not work. I successfully
ran the script on January 20, 2018, but please report it to me if
you have any problems).

**NOTE: this script is for Python 2.7 only**


## Setup
You'll need Tensorflow >=1.1.0 and its dependecies installed in order
for the script to work (see [here](https://www.tensorflow.org/)).

Once you've installed and configured Tensorflow, download the source
files and `cd` into the folder:
```sh
$ git clone https://gitlab.com/danielegrattarola/twitter-sentiment-cnn.git
$ cd twitter-sentiment-cnn
```
Before being able to use the script, some setup is needed; download the
dataset from the link above by running:
```sh
$ python dataset_downloader.py
```
Read the dataset from the CSV into two files (.pos and .neg) with:
```sh
$ python csv_parser.py
```
And generate a CSV with the vocabulary (and its inverse mapping) with:
```sh
$ python vocab_builder.py
```
The files will be created in the `twitter-sentiment-dataset/` folder.
Finally, create an `output/` folder that will contain all session
checkpoints needed to restore the trained models:
```sh
mkdir output
```
Now everything is set up and you're ready to start training the model.

## Usage
The simplest way to run the script is:
```sh
$ python twitter-sentiment-cnn.py
```
which will load the dataset in memory, create the computation graph, and
quit. Try to run the script like this to see if everything is set up
correctly.
To run a training session on the full dataset (and save the result so
that we can reuse the network later, or perform more training) run:
```sh
python twitter-sentiment-cnn.py --train --save
```
After training, we can test the network as follows:
```sh
$ python twitter-sentiment-cnn.py --load path/to/ckpt/folder/ --custom_input 'I love neural networks!'
```
which will eventually output: 
```
...
Processing custom input: I love neural networks!
Custom input evaluation: POS
Actual output: [ 0.19249919  0.80750078]
...
```

By running: 
```sh 
$ python twitter-sentiment-cnn.py -h
```
the script will output a list of all customizable flags and parameters.
The parameters are:
- `train`: train the network;
- `save`: save session checkpoints;
- `save_protobuf`: save model as binary protobuf;
- `evaluate_batch`: evaluate the network on a held-out batch from the
    dataset and print the results (for debugging/educational purposes);
- `load`: restore a model from the given path;
- `custom_input`: evaluate the model on the given string;
- `filter_sizes`: comma-separated filter sizes for the convolutional
    layers (default: '3,4,5');
- `dataset_fraction`: fraction of the dataset to load in memory, to
    reduce memory usage (default: 1.0; uses all dataset);
- `embedding_size`: size of the word embeddings (default: 128);
- `num_filters`: number of filters per filter size (default: 128);
- `batch_size`: batch size (default: 128);
- `epochs`: number of training epochs (default: 3);
- `valid_freq`: how many times per epoch to perform validation testing
    (default: 1);
- `checkpoint_freq`: how many times per epoch to save the model
    (default: 1);
- `test_data_ratio`: fraction of the dataset to use for validation
    (default: 0.1);
- `device`: device to use for running the model (can be either 'cpu' or
    'gpu').

## Pre-trained model
User [@Horkyze](https://github.com/Horkyze) kindly trained the model
for three epochs on the full dataset and shared the summary folder for
quick deploy.
The folder is available on [Mega](https://mega.nz/#!xVg0ARYK!oVyBZatotQGOD_FFSzZl5gTS1Z49048vjFEbyzftcFY),
to load the model simply unpack the zip file and use the `--load` flag
as follows:

```sh
# Current directoty: twitter-sentiment-cnn/
$ unzip path/to/run20180201-231509.zip
$ python twitter-sentiment-cnn.py --load path/to/run20180201-231509/ --custom_input "I love neural networks!"
```

Running this command should give you something like:

```
======================= START! ========================
	data_helpers: loading positive examples...
	data_helpers: [OK]
	data_helpers: loading negative examples...
	data_helpers: [OK]
	data_helpers: cleaning strings...
	data_helpers: [OK]
	data_helpers: generating labels...
	data_helpers: [OK]
	data_helpers: concatenating labels...
	data_helpers: [OK]
	data_helpers: padding strings...
	data_helpers: [OK]
	data_helpers: building vocabulary...
	data_helpers: [OK]
	data_helpers: building processed datasets...
	data_helpers: [OK]

Flags:
	batch_size = 128
	checkpoint_freq = 1
	custom_input = I love neural networks!
	dataset_fraction = 0.001
	device = cpu
	embedding_size = 128
	epochs = 3
	evaluate_batch = False
	filter_sizes = 3,4,5
	load = output/run20180201-231509/
	num_filters = 128
	save = False
	save_protobuf = False
	test_data_ratio = 0.1
	train = False
	valid_freq = 1

Dataset:
	Train set size = 1421
	Test set size = 157
	Vocabulary size = 274562
	Input layer size = 36
	Number of classes = 2

Output folder: /home/phait/dev/twitter-sentiment-cnn/output/run20180208-112402
Data processing OK, loading network...
Evaluating custom input: I love neural networks!
Custom input evaluation: POS
Actual output: [0.04109644 0.95890355]
```

**NOTE: loading this model won't work if you change anything in the
default network architecture, so don't set the `--filter_sizes` flag**.

According to the `log.log` file provided by [@Horkyze](https://github.com/Horkyze),
the model had a final validation accuracy of 0.80976, and a validation
loss of 53.3314.

I sincerely thank [@Horkyze](https://github.com/Horkyze) for providing
the computational power and sharing the model with me.

## Model description
The network implemented in this script is a single layer CNN structured
as follows:
- **Embedding layer**: takes as input the tweets (as strings) and maps
    each word to an n-dimensional space so that it is represented as a
    sparse vector (see [word2vec](https://en.wikipedia.org/wiki/Word2vec)).
- **Convolution layers**: a set of parallel 1D convolutional layers
    with the given filter sizes and 128 output channels.
    A filter's size is the number of embedded words that the
    filter covers.
- **Pooling layers**: a set of pooling layers associated to each of the
    convolutional layers.
- **Concat layer**: concatenates the output of the different pooling
    layers into a single tensor.
- **Dropout layer**: performs neuron dropout (some neurons are randomly
    not considered during training).
- **Output layer**: fully connected layer with a softmax activation
    function to perform classification.

The script will automatically log the session with Tensorboard. To
visualize the computation graph and training metrics run:
```sh
$ tensorboard --logdir output/path/to/summaries/
```
and then navigate to `localhost:6006` from your browser (you'll see the
computation graph in the *Graph* section).