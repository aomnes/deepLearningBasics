import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def custom_model_fn(features, labels, mode):
    """Model function for PA2 Extra2"""

    # Write your custom layer

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3]) # cifar-10
    # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1]) # mnist

    layer_conv1 = tf.layers.conv2d(input_layer, filters=96, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)

    layer_conv2 = tf.layers.conv2d(layer_conv1, filters=96, kernel_size=[3, 3], strides=2, padding='same', activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs = layer_conv2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    layer_conv3 = tf.layers.conv2d(dropout1, filters=192, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)

    layer_conv4 = tf.layers.conv2d(layer_conv3, filters=192, kernel_size=[3, 3], strides=2, padding='same', activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs = layer_conv4, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    layer_conv5 = tf.layers.conv2d(dropout2, filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    max_pooling_1 = tf.layers.max_pooling2d(layer_conv5, pool_size=[2, 2], strides=2)

    layer_conv6 = tf.layers.conv2d(max_pooling_1, filters=256, kernel_size=[3, 3], strides=2, padding='same', activation=tf.nn.relu)
    max_pooling_2 = tf.layers.max_pooling2d(layer_conv6, pool_size=[2, 2], strides=2)
    dropout3 = tf.layers.dropout(inputs = max_pooling_2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    resize = tf.reshape(dropout3, [-1, 1 * 1 * 256])
    #norm = tf.nn.batch_normalization(resize, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense6 = tf.layers.dense(resize, 1024 , name="dense1", activation=tf.nn.relu)


    # Output logits Layer
    logits = tf.layers.dense(inputs=dense6, units=10)

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
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9, use_nesterov=True) # Refer to tf.train
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
    # Write your dataset path
    dataset_train = np.load('./extra2-train.npy')
    dataset_eval =  np.load('./extra2-valid.npy')
    test_data =  np.load('./extra2-test_img.npy')

    f_dim = dataset_train.shape[1] - 1
    train_data = dataset_train[:,:f_dim].astype(np.float32)
    train_labels = dataset_train[:,f_dim].astype(np.int32)
    eval_data = dataset_eval[:,:f_dim].astype(np.float32)
    eval_labels = dataset_eval[:,f_dim].astype(np.int32)
    test_data = test_data.astype(np.float32)

    # Save model and checkpoint
    mnist_classifier = tf.estimator.Estimator(model_fn=custom_model_fn, model_dir="./model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model. You can train your model with specific batch size and epoches
    train_input = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
        y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
    mnist_classifier.train(input_fn=train_input, steps=20000, hooks=[logging_hook])

    # Eval the model. You can evaluate your trained model with validation data
    eval_input = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
        y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input)


    ## ----------- Do not modify!!! ------------ ##
    # Predict the test dataset
    pred_input = tf.estimator.inputs.numpy_input_fn(x={"x": test_data}, shuffle=False)
    pred_results = mnist_classifier.predict(input_fn=pred_input)
    pred_list = list(pred_results)
    result = np.asarray([list(x.values())[1] for x in pred_list])
    ## ----------------------------------------- ##

    np.save('extra2_20186029_network_7.npy', result)
