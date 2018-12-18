import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from image_corrector import imageprepare

def NeuralNetwork():

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]
    m, n = X_train.shape


    n_inputs = 28*28
    n_hidden_layer1 = 400
    n_hidden_layer2 = 100
    n_outputs = 10
    learning_rate = 0.03
    n_epochs = 10001
    batch_size = 50
    n_batches = int(np.ceil(m / batch_size))

    checkpoint_path = "tmp/deep_mnist_model.ckpt"
    checkpoint_epoch_path = checkpoint_path + ".epoch"
    final_model_path = "./deep_mnist_model.ckpt"

    best_loss = np.infty
    epochs_without_progress = 0
    max_epochs_without_progress = 50

    def reset_graph(seed=42):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)

    data = imageprepare('./test.jpg')

    reset_graph()
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")


    with tf.name_scope("dnn"):
        hidden_layer1 = tf.layers.dense(X, n_hidden_layer1, name="hidden_layer1", activation=tf.nn.relu)
        hidden_layer2 = tf.layers.dense(hidden_layer1, n_hidden_layer2, name="hidden_layer2", activation=tf.nn.relu)
        logits = tf.layers.dense(hidden_layer2, n_outputs, name="outputs")


    with tf.name_scope("loss"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(cross_entropy, name="loss")
        loss_summary = tf.summary.scalar('loss', loss)


    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)


    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    def log_dir(prefix=""):
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = "tf_logs"
        if prefix:
            prefix += "-"
        name = prefix + "run-" + now
        return "{}/{}/".format(root_logdir, name)


    logdir = log_dir("mnist_dnn")

    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    def shuffle_batch(X, y, batch_size):
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            yield X_batch, y_batch

    with tf.Session() as sess:
        saver.restore(sess, final_model_path)
        X_new = data
        Z = logits.eval(feed_dict = {X: X_new})
        y_pred = np.argmax(Z, axis=1)
        return y_pred

if __name__ == "__main__":
    NeuralNetwork()