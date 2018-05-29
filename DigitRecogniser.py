from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports:
import numpy as np
import tensorflow as tf
import csv
import os
import matplotlib.pyplot as plt
#import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # in order to omit extra warnings from TensorFlow
tf.logging.set_verbosity( tf.logging.INFO )

DROPOUT_RATE = 0.42 #0.4 in baseline solution

# Convolutional Neural Network model function
def cnn_model_fn( features, labels, mode ):
    # Input layer:
    input_layer = tf.reshape( features["x"], [-1, 28, 28, 1] )
    # Convolutional layer #1:
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu )
    # Pooling layer #1:
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2 )
    # Convolutional layer #2:
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu )
    # Pooling layer #2:
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2 )
    # Dense layer:
    pool2_flat = tf.reshape( pool2, [-1, 7 * 7 * 64 ] )
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu )
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=DROPOUT_RATE,
        training=( mode == tf.estimator.ModeKeys.TRAIN ) )
    # Logits layer:
    logits = tf.layers.dense( 
        inputs=dropout, 
        units=10 )

    predictions = {
        "classes": tf.argmax( input=logits, axis=1 ),
        "probabilities": tf.nn.softmax( logits, name="softmax_tensor" )
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec( mode=mode, predictions=predictions )

    # Calculate loss ( for both TRAIN and EVAL modes )   ???
    onehot_labels = tf.one_hot( indices=tf.cast( labels, tf.int32 ), depth=10 )
    loss = tf.losses.softmax_cross_entropy( onehot_labels=onehot_labels, logits=logits )
    # same?
    #loss = tf.losses.sparse_softmax_cross_entropy( labels=labels, logits=logits )

    # Configure the training op ( for TRAIN mode )
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer( learning_rate=0.001 ) # alpha?
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step() )
        return tf.estimator.EstimatorSpec( mode=mode, loss=loss, train_op=train_op )

    # Add evaluation metrics ( for EVAL mode )
    eval_metric_ops = {
        "accuracy" : tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"] ) }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops )

def train( classifier, train_data, train_labels, steps ):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True )
    classifier.train( input_fn=train_input_fn, steps=steps )
    return

def evaluate( classifier, eval_data, eval_labels ):
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False )
    return classifier.evaluate( input_fn=eval_input_fn )

def predict( classifier, test_data ):
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        batch_size=100, #??? will it work?
        num_epochs=1,
        shuffle=False )
    return classifier.predict( input_fn=test_input_fn )


def small_fast_test( classifier, train_data, train_labels ):
    SMALL_STEPS=1000
    small_data_len = len( train_labels ) // 20
    small_data = train_data[ :small_data_len, : ]
    small_labels = train_labels[ :small_data_len ]

    train( classifier=classifier, train_data=small_data, train_labels=small_labels, steps=SMALL_STEPS )
    eval_results = evaluate( classifier=classifier, eval_data=small_data, eval_labels=small_labels )
    print( eval_results )
    return

def main_calculation( mnist_classifier, train_data, train_labels ):

    STEPS_TOTAL=20000 # should be 20000
    STEPS_PER_EVAL=1250 # should be 1250
    EVAL_NUM = STEPS_TOTAL // STEPS_PER_EVAL

    x = np.linspace( 0, STEPS_TOTAL, EVAL_NUM + 1 )         # graph
    y = np.linspace( 0.10, 0.10, EVAL_NUM + 1 )             # graph
    y[ 0 ] = 0.10
    for curr_eval in range( EVAL_NUM ):
        train( classifier=mnist_classifier, train_data=train_data, train_labels=train_labels, steps=STEPS_PER_EVAL )
        eval_results = evaluate( classifier=mnist_classifier, eval_data=train_data, eval_labels=train_labels )
        print( eval_results ) # later make a graph from 'em
        y[ curr_eval + 1 ] = eval_results[ "accuracy" ]     # graph

    fig, ax = plt.subplots()                                # graph
    ax.plot( x, y, color="blue", label="Accuracy( Step )" ) # graph
    ax.set_xlabel( "Steps" )                                # graph
    ax.set_ylabel( "Accuracy" )                             # graph
    ax.legend()                                             # graph
    plt.show()                                              # graph
    fig.savefig( "Kaggle/report.png" )                      # graph

    with open("./Kaggle/test.csv", "r") as f:
        raw_data = list( csv.reader( f ) )
    data = np.array( raw_data[ 1: ] )
    test_data = np.array( data ).astype("float") / 255.0

    predictions = predict( classifier=mnist_classifier, test_data=test_data )

    with open("./Kaggle/submission.csv", "w") as f:
         fieldnames = ["ImageId", "Label"]
         writer = csv.DictWriter( f, fieldnames=fieldnames )
         writer.writeheader()
         i = 1
         for curr_prediction in predictions:
              predicted_digit = curr_prediction["classes"]
              writer.writerow( {
                  "ImageId": i, 
                  "Label": predicted_digit} )
              i += 1
    return


#===MAIN============================================================
def main( unused_argv ):
    with open("./Kaggle/train.csv", "r") as f:
        raw_data = list( csv.reader( f ) )
    data = np.array( raw_data[ 1: ] )
    train_labels = data[ :, 0 ].astype("float") 
    train_data = data[ :, 1: ].astype("float") / 255.0

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="/tmp/mnist_convnet_model" )

    #small_fast_test( classifier=mnist_classifier, train_data=train_data, train_labels=train_labels )
    main_calculation( mnist_classifier=mnist_classifier, train_data=train_data, train_labels=train_labels )

if __name__ == "__main__":
    tf.app.run()

