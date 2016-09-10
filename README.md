# Twitter sentiment classification by Daniele Grattarola
This is an implementation in TensorFlow of a convolutional neural network (CNN) to perform sentiment classification on tweets.   
Rather than a pre-packaged tool to simply perform classification, this is a model that needs to be trained and fine tuned by hand and has more of an educational value.    
The dataset used in this example is taken from [here](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/) (someone reported to me that the link to the dataset appears to be dead sometimes, so dataset_downloader.py **might** not work. I succesfully ran the script on September 10, 2016, but feel free to contact me if you have any problems).    
You'll obviously need TensorFlow >=0.7.0 and its dependecies installed in order for the script to work (see [here](https://www.tensorflow.org/)).   
Download the source files and `cd` into the folder:
```sh
git clone https://gitlab.com/danielegrattarola/twitter-sentiment-cnn.git
cd twitter-sentiment-cnn
```
Before being able to use the script, some setup is needed; download the dataset from the link above by running: 
```sh
# Creates the twitter-sentiment-dataset folder and downloads the dataset csv
python dataset_downloader.py
```
Read the dataset from the CSV into two files (.pos and .neg) with:
```sh
# Creates tw-data.neg and tw-data.pos inside the twitter-sentiment-dataset folder
python csv_parser.py
```
And generate a CSV with the vocabulary (and its inverse mapping) with:
```sh
# Creates vocab.csv and vocab_inv.csv inside the twitter-sentiment-dataset folder
# This is needed to sucessfully restore saved sessions with different hyperparameters
python vocab_builder.py
```
The files will be generated into the `twitter-sentiment-dataset` folder. 
After the first data preprocessing, create an `output` folder which will contain all the data associated to different sessions of the network:
```sh
mkdir output
```
Now everything is set up and you're ready to start using the network. 

### Usage
The simplest way to run the script is:
```sh
python twitter-sentiment-cnn.py
```
which will load the dataset in memory, create the computation graph and quit. Try to run the script like this to see if everything is set up correctly.  
To run a training session on the full dataset (and save the result so that we can reuse the network later, or perform more training) run:
```sh
python twitter-sentiment-cnn.py --train --save
```
When the training in finished, we can test the network as follows:
```sh
# replace 'runYYYYMMDD-HHmmss' with the actual name of the run folder (not the path, just the name of the folder)
python twitter-sentiment-cnn.py --load runYYYYMMDD-HHmmss --custom_input 'I love neural networks!'
```
which will eventually output: 
```
...
Processing custom input: I love neural networks!
Custom input evaluation: POS
Actual output: [ 0.19249919  0.80750078]
...
```
We can also fine tune the parameters to have different performance.   
By running: 
```sh 
python twitter-sentiment-cnn.py -h
```
the script will output a list of all customizable session parameters. The parameters are: 
- `train`: whether the network should perform training (default: False)
- `save`: save session checkpoints (default: False)
- `evaluate_batch`: print the network output on a batch from the dataset (for debugging/educational purposes)
- `load`: restore the given session if it exists (Pass the name of the session folder: runYYYMMDD-hhmmss)
- `custom_input`: the program will print the network output for the given input string.
- `filter_sizes`: comma-separated filter sizes for the convolution layer (default: '3,4,5')
- `reduced_dataset`: use a reduced dataset to improve memory usage (default: 1; uses all dataset)
- `embedding_size`: size of character embedding (default: 128)
- `num_filters`: number of filters per filter size (default: 128)
- `batch_size`: batch Size (default: 100)
- `epochs`: number of training epochs (default: 3)
- `valid_freq`: set how many times per epoch to perform validation testing (default: 1)
- `checkpoint_freq`: set how many times per epoch to save the model (default: 1)
- `test_data_ratio`: percentual of the dataset to be used for validation (default: 10)

### Network description
The network implemented in this script is a single layer CNN structured as follows: 
- **Embedding layer**: takes in input the tweets (as strings) and maps each word to an n-space so that it is represented as a sparse vector (see [word2vec](https://en.wikipedia.org/wiki/Word2vec)).
- **Convolution layers**: take as input the output of the previous layer and perform a convolution using an arbitrary number of filters of arbitrary size.   
Each layer is associated to a filter size and outputs to a single pooling layer (by defalut: 128 filters per filter size, sizes 3,4,5; this means that the network has 3 parallel conv+pool sections).    
Note that a filter size is the number of words that the filter covers. 
- **Pooling layers**: pool the output of a convolution layer using max-pooling. 
- **Concat layer**: concatenates the output of the different pooling layers into a single tensor. 
- **Dropout layer**: performs neuron dropout (some random neurons are not considered during the computation).
- **Output layer**: fully connected layer associated to a weight and bias matrix that uses softmax as activation function to perform classification. 

To have a better visualization of the network, first run a training session, then:
```sh
tensorboard --logdir output/runYYYYMMDD-HHmmss/summaries/
```
and then navigate to `localhost:6006` from your browser (you'll see the TF computation graph in the *Graph* section). 