import tensorflow as tf
import numpy as np

# Hyperparameters
input_layer_size = 28 * 28
hidden_layer_size_1 = 300
hidden_layer_size_2 = 300
hidden_layer_size_3 = 300
output_layer_size = 10
learning_rate = 0.001
dropout_rate = 0.4;

tf.logging.set_verbosity(tf.logging.INFO)

def custom_model_fn(features, labels, mode):
    """Model function for PA1"""

    # Write your custom layer

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1]) # You also can use 1 x 784 vector, or [batch_size, image_height, image_width, channels]
    #input_layer = features["x"]

    layer_conv_1 = tf.layers.conv2d(input_layer, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    max_pooling_1 = tf.layers.max_pooling2d(layer_conv_1, pool_size=[2, 2], strides=2)
    dropout1 = tf.layers.dropout(inputs = max_pooling_1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    layer_conv_2 = tf.layers.conv2d(dropout1, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    max_pooling_2 = tf.layers.max_pooling2d(layer_conv_2, pool_size=[2, 2], strides=2)
    dropout2 = tf.layers.dropout(inputs = max_pooling_2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    layer_conv_3 = tf.layers.conv2d(dropout2, filters=128, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    dropout3 = tf.layers.dropout(inputs = layer_conv_3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    layer_conv_4 = tf.layers.conv2d(dropout3, filters=128, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    dropout4 = tf.layers.dropout(inputs = layer_conv_4, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    layer_conv_5 = tf.layers.conv2d(dropout4, filters=320, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    dropout5 = tf.layers.dropout(inputs = layer_conv_5, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    layer_conv_6 = tf.layers.conv2d(dropout5, filters=320, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    dropout6 = tf.layers.dropout(inputs = layer_conv_6, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    resize = tf.reshape(dropout6, [-1, 7 * 7 * 320])
    hidden7 = tf.layers.dense(resize, 1024, name="hidden3", activation=tf.nn.relu)
    dropout7 = tf.layers.dropout(inputs = hidden7, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Output logits Layer
    logits = tf.layers.dense(inputs=dropout7, units=output_layer_size, name="output_logits", activation=None) #Size [batch_size, 10]

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # In predictions, return the prediction value, do not modify
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Select your loss and optimizer from tensorflow API
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) # Refer to tf.losses

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True) # Refer to tf.train
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode) ==> if mode == tf.estimator.ModeKeys.EVAL
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
    # Write your dataset path
    dataset_train = np.load('./train.npy')
    dataset_eval =  np.load('./valid.npy')
    test_data =  np.load('./test.npy')

    train_data = dataset_train[:, :784] #from 0 to 783
    train_labels = dataset_train[:, 784].astype(np.int32) #VALUE 784 Convert to np.int32 to get TRAIN Label (01..9)(Copy of the array, cast to a specified type)
    eval_data = dataset_eval[:, :784] #from 0 to 783
    eval_labels = dataset_eval[:, 784].astype(np.int32) # VALUE 784 Convert to np.int32 to get VALIDATION Label (01..9) (Copy of the array, cast to a specified type)

    # Save model and checkpoint
    classifier = tf.estimator.Estimator(model_fn = custom_model_fn, model_dir="./model_150")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model. You can train your model with specific batch size and epoches
    train_input = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y = train_labels, batch_size=100, num_epochs=None, shuffle=True)
    classifier.train(input_fn=train_input, steps=20000, hooks=[logging_hook])

    # Eval the model. You can evaluate your trained model with validation data
    eval_input = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y = eval_labels, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input)


    ## ----------- Do not modify!!! ------------ ##
    # Predict the test dataset
    pred_input = tf.estimator.inputs.numpy_input_fn(x={"x": test_data}, shuffle=False)
    pred_results = classifier.predict(input_fn=pred_input)
    pred_list = list(pred_results)
    result = np.asarray([list(x.values())[1] for x in pred_list])
    ## ----------------------------------------- ##

    np.save('20186029_network_7.npy', result)
