# Convolutional Neural Network Estimator for 2048 solver, built with tf.layers.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # The boards are 4x4 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 4, 4, 1])

    # Convolutional Layer #1
    # Computes 10 features using a 2x2 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 4, 4, 1]
    # Output Tensor Shape: [batch_size, 4, 4, 10]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=10,
        kernel_size=[2,2],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2
    # Computes 20 features using a 2x2 filter.
    # Padding is added to preserve width and height.
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=20,
        kernel_size=[2,2],
        padding="same",
        activation=tf.nn.relu)

    # Flatten tensor into a batch of vectors
    conv2_flat = tf.reshape(conv2, [-1, 4 * 4 * 20])

    # Dense Layer
    dense1 = tf.layers.dense(inputs=conv2_flat, units=100, activation=tf.nn.relu)

    # Add dropout operation; 0.05 probability that element will be kept
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.7, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense Layer
    dense2 = tf.layers.dense(inputs=dropout1, units=100, activation=tf.nn.relu)

    # Add dropout operation; 0.05 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense2, rate=0.7, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 4]
    logits = tf.layers.dense(inputs=dropout, units=4)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.uint8), depth=4)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def row_cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # The boards are 4x4 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 4, 4, 1])

    # Convolutional Layer #1
    # Computes 10 features using a 2x2 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 4, 4, 1]
    # Output Tensor Shape: [batch_size, 4, 4, 10]
    # conv1 = tf.layers.conv2d(
    #     inputs=input_layer,
    #     filters=10,
    #     kernel_size=[2,2],
    #     padding="same",
    #     activation=tf.nn.relu)

    # Convolutional Layer #2
    # Computes 20 features using a 2x2 filter.
    # Padding is added to preserve width and height.
    # conv2 = tf.layers.conv2d(
    #     inputs=conv1,
    #     filters=20,
    #     kernel_size=[2,2],
    #     padding="same",
    #     activation=tf.nn.relu)

    # Row filter
    conv3 = tf.layers.conv2d(
        inputs=input_layer,
        filters=40,
        kernel_size=[1,4],
        padding="valid",
        activation=tf.nn.relu)

    # Column filter
    conv4 = tf.layers.conv2d(
        inputs=input_layer,
        filters=40,
        kernel_size=[4,1],
        padding="valid",
        activation=tf.nn.relu)

    # Flatten tensor into a batch of vectors
    # conv2_flat = tf.reshape(conv2, [-1, 4 * 4 * 20])
    conv3_flat = tf.reshape(conv3, [-1, 4 * 40])
    conv4_flat = tf.reshape(conv4, [-1, 4 * 40])

    # Concatenate of 3 tensors
    conv_flat = tf.concat([conv3_flat, conv4_flat], axis=1)

    # Dense Layer
    dense1 = tf.layers.dense(inputs=conv_flat, units=1000, activation=tf.nn.relu)

    # Add dropout operation; 0.05 probability that element will be kept
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.95, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense Layer
    # dense2 = tf.layers.dense(inputs=dropout1, units=100, activation=tf.nn.relu)

    # Add dropout operation; 0.05 probability that element will be kept
    # dropout = tf.layers.dropout(
    #    inputs=dense2, rate=0.7, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 4]
    logits = tf.layers.dense(inputs=dropout1, units=4)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.uint8), depth=4)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def shallow_nn(features, labels, mode):
    # input_layer = tf.reshape(features["x"], [-1, 4, 4, 1])

    # full connected layer
    dense = tf.layers.dense(
        inputs=features["x"],
        units=100,
        activation=tf.nn.relu
    )
    # dropout = tf.layers.dropout(
    #     inputs=dense,
    #     rate=0,
    #     training=(mode == tf.estimator.ModeKeys.TRAIN)
    # )

    # logits layer
    logits = tf.layers.dense(
    #    inputs=dropout,
        inputs=dense,
        units=4
    )

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits,
            name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                    predictions=predictions)

    # calculate loss for both train and eval modes
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),
                               depth=4)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits
    )

    # training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss, train_op=train_op)

    # add evaluation metrics
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    cut_label = 7000
    task = input('input mode: ')
    with open("output.txt", 'r') as input_fo:
        data = []
        label = []
        for line in input_fo:
            if line != '\n':
                temp = line.split('   ')
                temp_data = []
                for i in range(16):
                    x = np.uint16(temp[i])
                    if x > 0:
                        temp_data.append(np.float32(np.log2(x)))
                    else:
                        temp_data.append(np.float32(0))
                data.append(temp_data)
                if temp[16] == 'a':
                    temp_label = 0
                elif temp[16] == 'd':
                    temp_label = 1
                elif temp[16] == 'w':
                    temp_label = 2
                elif temp[16] == 's':
                    temp_label = 3
                label.append(temp_label)

    train_data = np.array(data[:cut_label])
    train_labels = np.array(label[:cut_label])
    eval_data = np.array(data[cut_label:])
    eval_labels = np.array(label[cut_label:])

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        # model_fn=shallow_nn, model_dir="machines/shallow_nn")
        model_fn=row_cnn_model_fn, model_dir="machines/cnn")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    if task == 'T':
        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
        classifier.train(
            input_fn=train_input_fn,
            steps=10000,
            hooks=[logging_hook])

    # Evaluate the performance of the model
    # on test and train set
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        num_epochs=1,
        shuffle=False)
    train_accuracy = classifier.evaluate(input_fn=eval_input_fn)
    print('train score: {}'.format(train_accuracy))
    print('test score: {}'.format(eval_results))

if __name__ == "__main__":
    tf.app.run()
