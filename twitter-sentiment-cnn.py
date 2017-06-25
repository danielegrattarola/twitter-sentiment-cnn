import tensorflow as tf
import numpy as np
from random import randint
from data_helpers import batch_iter, load_data, string_to_int
import os
import time
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tqdm import tqdm


def log(*string, **kwargs):
    output = ' '.join(string)
    if kwargs.pop('verbose', True):
        print output
    LOG_FILE.write(''.join(['\n', output]))


def weight_variable(shape, name):
    """
    Creates a new Tf weight variable with the given shape and name.
    Returns the new variable.
    """
    var = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(var, name=name)


def bias_variable(shape, name):
    """
    Creates a new Tf bias variable with the given shape and name.
    Returns the new variable.
    """
    var = tf.constant(0.1, shape=shape)
    return tf.Variable(var, name=name)


def human_readable_output(a_batch):
    """
    Feeds a batch to the network and prints in a human readable format a
    comparison between the batch's labels and the network output.
    Outputs comparison to stdout.
    """
    log('Network output on random data...')
    sentences = zip(*a_batch)[0]
    word_sentence = []
    network_result = sess.run(tf.argmax(network_out, 1),
                              feed_dict={data_in: zip(*a_batch)[0],
                                         dropout_keep_prob: 1.0})
    actual_result = sess.run(tf.argmax(data_out, 1),
                             feed_dict={data_out: zip(*a_batch)[1]})
    # Translate the string to ASCII (remove <PAD/> symbols)
    for s in sentences:
        output = ''
        for w in s:
            output += vocabulary_inv[w.astype(np.int)][0] + ' '
        output = output.translate(None, '<PAD/>')
        word_sentence.append(output)
    # Output the network result
    for idx, item in enumerate(network_result, start=0):
        network_sentiment = 'POS' if item == 1 else 'NEG'
        actual_sentiment = 'POS' if actual_result[idx] == 1 else 'NEG'

        if item == actual_result[idx]:
            status = '\033[92mCORRECT\033[0m'
        else:
            status = '\033[91mWRONG\033[0m'

        log('\n%s\nLABEL: %s - OUTPUT %s | %s' %
            (word_sentence[idx], actual_sentiment, network_sentiment, status))


def evaluate_sentence(sentence, vocabulary):
    """
    Translates a string to its equivalent in the integer vocabulary and feeds it
    to the network.
    Outputs result to stdout.
    """
    x_to_eval = string_to_int(sentence, vocabulary, max(len(_) for _ in x))
    result = sess.run(tf.argmax(network_out, 1),
                      feed_dict={data_in: x_to_eval,
                                 dropout_keep_prob: 1.0})
    unnorm_result = sess.run(network_out, feed_dict={data_in: x_to_eval,
                                                     dropout_keep_prob: 1.0})
    network_sentiment = 'POS' if result == 1 else 'NEG'
    log('Custom input evaluation:', network_sentiment)
    log('Actual output:', str(unnorm_result[0]))

# Hyperparameters
tf.flags.DEFINE_boolean('train', False,
                        'Should the network perform training? (default: False)')
tf.flags.DEFINE_boolean('save', False,
                        'Save session checkpoints (default: False)')
tf.flags.DEFINE_boolean('save_protobuf', False,
                        'Save session as binary protobuf (default: False)')
tf.flags.DEFINE_boolean('evaluate_batch', False,
                        'Print the network output on a batch from the dataset '
                        '(for debugging/educational purposes')
tf.flags.DEFINE_string('load', None,
                       'Load a previous run from the given path (must '
                       'contain a log and checkpoint file).')
tf.flags.DEFINE_string('device', 'cpu', 'Type of device to run the network on.'
                                        '(Can be either \'cpu\' or \'gpu\')')
tf.flags.DEFINE_string('custom_input', '',
                       'The program will print the network output for the '
                       'given input string.')
tf.flags.DEFINE_string('filter_sizes', '3,4,5',
                       'Comma-separated filter sizes for the convolution layer '
                       '(default: \'3,4,5\')')
tf.flags.DEFINE_integer('reduced_dataset', 1,
                        'Use 1/[REDUCED_DATASET]-th of the dataset to reduce '
                        'memory usage (default: 1; uses all dataset)')
tf.flags.DEFINE_integer('embedding_size', 128,
                        'Size of character embedding (default: 128)')
tf.flags.DEFINE_integer('num_filters', 128,
                        'Number of filters per filter size (default: 128)')
tf.flags.DEFINE_integer('batch_size', 100, 'Batch Size (default: 100)')
tf.flags.DEFINE_integer('epochs', 3, 'Number of training epochs (default: 3)')
tf.flags.DEFINE_integer('valid_freq', 1,
                        'Check model accuracy on validation set '
                        '[VALIDATION_FREQ] times per epoch (default: 1)')
tf.flags.DEFINE_integer('checkpoint_freq', 1,
                        'Save model [CHECKPOINT_FREQ] times per epoch '
                        '(default: 1)')
tf.flags.DEFINE_integer('test_data_ratio', 10,
                        'Percentual of the dataset to be used for validation '
                        '(default: 10)')
FLAGS = tf.flags.FLAGS

# File paths
OUT_DIR = os.path.abspath(os.path.join(os.path.curdir, 'output'))
if FLAGS.load is not None:
    # Use logfile and checkpoint from given path
    RUN_DIR = FLAGS.load
    LOG_FILE_PATH = os.path.abspath(os.path.join(RUN_DIR, 'log.log'))
    CHECKPOINT_FILE_PATH = os.path.abspath(os.path.join(RUN_DIR, 'ckpt.ckpt'))
else:
    RUN_ID = time.strftime('run%Y%m%d-%H%M%S')
    RUN_DIR = os.path.abspath(os.path.join(OUT_DIR, RUN_ID))
    LOG_FILE_PATH = os.path.abspath(os.path.join(RUN_DIR, 'log.log'))
    CHECKPOINT_FILE_PATH = os.path.abspath(os.path.join(RUN_DIR, 'ckpt.ckpt'))
    os.mkdir(RUN_DIR)
SUMMARY_DIR = os.path.join(RUN_DIR, 'summaries')
LOG_FILE = open(LOG_FILE_PATH, 'a', 0)


log('======================= START! ========================')
# Load data
x, y, vocabulary, vocabulary_inv = load_data(FLAGS.reduced_dataset)

# Randomly shuffle data
np.random.seed(123)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
text_percent = FLAGS.test_data_ratio / 100.0
test_index = int(len(x) * text_percent)
x_train, x_test = x_shuffled[:-test_index], x_shuffled[-test_index:]
y_train, y_test = y_shuffled[:-test_index], y_shuffled[-test_index:]

# Parameters
sequence_length = x_train.shape[1]
num_classes = y_train.shape[1]
vocab_size = len(vocabulary)
filter_sizes = map(int, FLAGS.filter_sizes.split(','))
validate_every = len(y_train) / (FLAGS.batch_size * FLAGS.valid_freq)
checkpoint_every = len(y_train) / (FLAGS.batch_size * FLAGS.checkpoint_freq)

# Set computation device
if FLAGS.device == 'gpu':
    device = '/gpu:0'
else:
    device = '/cpu:0'

# Log run data
log('\nFlags:')
for attr, value in sorted(FLAGS.__flags.iteritems()):
    log('\t%s = %s' % (attr, value))
log('\nDataset:')
log('\tTrain set size = %d\n'
    '\tTest set size = %d\n'
    '\tVocabulary size = %d\n'
    '\tInput layer size = %d\n'
    '\tNumber of classes = %d' %
    (len(y_train), len(y_test), len(vocabulary), sequence_length, num_classes))
log('\nOutput folder:', RUN_DIR)

# Session
sess = tf.InteractiveSession()

# Network
with tf.device(device):
    # Placeholders
    data_in = tf.placeholder(tf.int32, [None, sequence_length], name='data_in')
    data_out = tf.placeholder(tf.float32, [None, num_classes], name='data_out')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    # Stores the accuracy of the model for each batch of the validation testing
    valid_accuracies = tf.placeholder(tf.float32)
    # Stores the loss of the model for each batch of the validation testing
    valid_losses = tf.placeholder(tf.float32)

    # Embedding layer
    with tf.name_scope('embedding'):
        W = tf.Variable(tf.random_uniform([vocab_size, FLAGS.embedding_size],
                                          -1.0, 1.0),
                        name='embedding_matrix')
        embedded_chars = tf.nn.embedding_lookup(W, data_in)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    # Convolution + ReLU + Pooling layer
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope('conv-maxpool-%s' % filter_size):
            # Convolution Layer
            filter_shape = [filter_size,
                            FLAGS.embedding_size,
                            1,
                            FLAGS.num_filters]
            W = weight_variable(filter_shape, name='W_conv')
            b = bias_variable([FLAGS.num_filters], name='b_conv')
            conv = tf.nn.conv2d(embedded_chars_expanded,
                                W,
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name='conv')
            # Activation function
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            # Maxpooling layer
            ksize = [1,
                     sequence_length - filter_size + 1,
                     1,
                     1]
            pooled = tf.nn.max_pool(h,
                                    ksize=ksize,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='pool')
        pooled_outputs.append(pooled)

    # Combine the pooled feature tensors
    num_filters_total = FLAGS.num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Dropout
    with tf.name_scope('dropout'):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    # Output layer
    with tf.name_scope('output'):
        W_out = weight_variable([num_filters_total, num_classes], name='W_out')
        b_out = bias_variable([num_classes], name='b_out')
        network_out = tf.nn.softmax(tf.matmul(h_drop, W_out) + b_out)

    # Loss function
    cross_entropy = -tf.reduce_sum(data_out * tf.log(network_out))

    # Training algorithm
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Testing operations
    correct_prediction = tf.equal(tf.argmax(network_out, 1),
                                  tf.argmax(data_out, 1))
    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Validation ops
    valid_mean_accuracy = tf.reduce_mean(valid_accuracies)
    valid_mean_loss = tf.reduce_mean(valid_losses)

# Init session
if FLAGS.load is not None:
    log('Data processing OK, loading network...')
    saver = tf.train.Saver()
    try:
        saver.restore(sess, CHECKPOINT_FILE_PATH)
    except:
        log('Couldn\'t restore the session properly, falling back to default '
            'initialization.')
        sess.run(tf.global_variables_initializer())
else:
    log('Data processing OK, creating network...')
    sess.run(tf.global_variables_initializer())

# Summaries for loss and accuracy
loss_summary = tf.summary.scalar('Training loss', cross_entropy)
valid_loss_summary = tf.summary.scalar('Validation loss', valid_mean_loss)
valid_accuracy_summary = tf.summary.scalar('Validation accuracy',
                                           valid_mean_accuracy)
summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
tf.summary.merge_all()

# Training
if FLAGS.train:
    # Batches
    batches = batch_iter(zip(x_train, y_train), FLAGS.batch_size, FLAGS.epochs)
    test_batches = list(batch_iter(zip(x_test, y_test), FLAGS.batch_size, 1))
    my_batch = batches.next()  # To use with human_readable_output()

    # Pretty-printing variables
    global_step = 0
    batches_in_epoch = len(y_train) / FLAGS.batch_size
    batches_in_epoch = batches_in_epoch if batches_in_epoch != 0 else 1
    total_num_step = FLAGS.epochs * batches_in_epoch

    batches_progressbar = tqdm(batches, total=total_num_step,
                               desc='Starting training...')

    for batch in batches_progressbar:
        global_step += 1
        x_batch, y_batch = zip(*batch)

        # Run the training step
        feed_dict = {data_in: x_batch,
                     data_out: y_batch,
                     dropout_keep_prob: 0.5}
        train_result, loss_summary_result = sess.run([train_step, loss_summary],
                                                     feed_dict=feed_dict)

        # Print training accuracy
        feed_dict = {data_in: x_batch,
                     data_out: y_batch,
                     dropout_keep_prob: 1.0}
        accuracy_result = accuracy.eval(feed_dict=feed_dict)
        current_loss = cross_entropy.eval(feed_dict=feed_dict)
        current_epoch = (global_step / batches_in_epoch)

        batches_progressbar.set_description('Epoch: %s - loss: %s - acc: %s' %
                                            (current_epoch, current_loss,
                                             accuracy_result))

        # Write loss summary
        summary_writer.add_summary(loss_summary_result, global_step)

        # Validation testing
        # Evaluate accuracy as (correctly classified samples) / (all samples)
        # For each batch, evaluate the loss
        if global_step % validate_every == 0:
            accuracies = []
            losses = []
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                feed_dict = {data_in: x_test_batch,
                             data_out: y_test_batch,
                             dropout_keep_prob: 1.0}
                accuracy_result = accuracy.eval(feed_dict=feed_dict)
                current_loss = cross_entropy.eval(feed_dict=feed_dict)
                accuracies.append(accuracy_result)
                losses.append(current_loss)

            # Evaluate the mean accuracy of the model using the test accuracies
            mean_accuracy_result, accuracy_summary_result = sess.run(
                [valid_mean_accuracy, valid_accuracy_summary],
                feed_dict={valid_accuracies: accuracies})
            # Evaluate the mean loss of the model using the test losses
            mean_loss_result, loss_summary_result = sess.run(
                [valid_mean_loss, valid_loss_summary],
                feed_dict={valid_losses: losses})

            valid_msg = 'Step %d of %d (epoch %d), validation accuracy: %g, ' \
                        'validation loss: %g' % \
                        (global_step, total_num_step, current_epoch,
                         mean_accuracy_result, mean_loss_result)
            batches_progressbar.write(valid_msg)
            log(valid_msg, verbose=False)  # Write only to file

            # Write summaries
            summary_writer.add_summary(accuracy_summary_result, global_step)
            summary_writer.add_summary(loss_summary_result, global_step)

        if FLAGS.save and global_step % checkpoint_every == 0:
            batches_progressbar.write('Saving checkpoint...')
            log('Saving checkpoint...', verbose=False)
            saver = tf.train.Saver()
            saver.save(sess, CHECKPOINT_FILE_PATH)

    # Final validation testing
    accuracies = []
    losses = []
    for test_batch in test_batches:
        x_test_batch, y_test_batch = zip(*test_batch)
        feed_dict = {data_in: x_test_batch,
                     data_out: y_test_batch,
                     dropout_keep_prob: 1.0}
        accuracy_result = accuracy.eval(feed_dict=feed_dict)
        current_loss = cross_entropy.eval(feed_dict=feed_dict)
        accuracies.append(accuracy_result)
        losses.append(current_loss)

    mean_accuracy_result, accuracy_summary_result = sess.run(
        [valid_mean_accuracy, valid_accuracy_summary],
        feed_dict={valid_accuracies: accuracies})
    mean_loss_result, loss_summary_result = sess.run(
        [valid_mean_loss, valid_loss_summary], feed_dict={valid_losses: losses})
    log('End of training, validation accuracy: %g, validation loss: %g' %
        (mean_accuracy_result, mean_loss_result))

    # Write summaries
    summary_writer.add_summary(accuracy_summary_result, global_step)
    summary_writer.add_summary(loss_summary_result, global_step)

# Evaluate custom input
if FLAGS.custom_input != '':
    log('Evaluating custom input:', FLAGS.custom_input)
    evaluate_sentence(FLAGS.custom_input, vocabulary)

# Evaluate held-out batch
if FLAGS.evaluate_batch:
    if not FLAGS.train:
        _batches = list(batch_iter(zip(x_test, y_test), FLAGS.batch_size, 1))
        my_batch = _batches[randint(0, len(_batches))]
    human_readable_output(my_batch)

# Save final checkpoint
if FLAGS.save:
    log('Saving checkpoint...')
    saver = tf.train.Saver()
    saver.save(sess, CHECKPOINT_FILE_PATH)

# Save as binary Protobuffer
if FLAGS.save_protobuf:
    log('Saving Protobuf...')
    minimal_graph = convert_variables_to_constants(sess,
                                                   sess.graph_def,
                                                   ['output/Softmax'])
    tf.train.write_graph(minimal_graph, RUN_DIR, 'minimal_graph.proto',
                         as_text=False)
    tf.train.write_graph(minimal_graph, RUN_DIR, 'minimal_graph.txt',
                         as_text=True)
