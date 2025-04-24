import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DropConnect(layers.Layer):
    def __init__(self, units, rate, activation=None, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.dense = layers.Dense(units, activation=activation)
        self.units = units
        self.rate = rate
        self.activation = layers.Activation(activation)

    def call(self, inputs, training=False):
        if not self.dense.built:
            self.dense.build(inputs.shape)
        kernel = self.dense.kernel
        bias = self.dense.bias

        if training:
            kernel_scale = kernel / self.rate
            mask = tf.random.uniform(tf.shape(kernel)) < self.rate
            kernel_mask = tf.where(mask, kernel_scale, 0)
            output = tf.matmul(inputs, kernel_mask) + bias
            if self.activation:
                output = self.activation(output)
        else:
            output = tf.matmul(inputs, kernel) + bias
            if self.activation:
                output = self.activation(output)

        return output


class LeNet_DropConnect(Model):
    def __init__(self):
        super(LeNet_DropConnect, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(6, 5, activation='relu', input_shape=(28, 28, 1))
        self.pool1 = tf.keras.layers.MaxPooling2D(2)
        self.conv2 = tf.keras.layers.Conv2D(16, 5, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = DropConnect(units=120, rate=1, activation='relu')
        self.fc2 = DropConnect(units=84, rate=0.7, activation='relu')  # keep rate
        self.fc3 = DropConnect(units=10, rate=1, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def load_mnist_data(batch_size):
    (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x = x.reshape((60000, 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
    x1 = np.concatenate((x, x_test))
    y1 = np.concatenate((y, y_test))
    np.random.seed(2023)
    train_size = 0.3
    index = np.random.rand(len(x1)) < train_size
    x, x_test = x1[index], x1[~index]
    y, y_test = y1[index], y1[~index]
    x_test = x_test[0:10000]
    y_test = y_test[0:10000]
    x = x[0:3000]
    y = y[0:3000]
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset.shuffle(1000)
    test_dataset.shuffle(1000)
    train_labels = y.astype('int32')
    test_labels = y_test.astype('int32')
    train_dataset = tf.data.Dataset.from_tensor_slices((x, train_labels)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, test_labels)).batch(batch_size)
    return train_dataset, test_dataset


if __name__ == '__main__':
    batch_size = 256
    train_dataset, test_dataset = load_mnist_data(batch_size)
    model = LeNet_DropConnect()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    train_accuracies = []
    test_accuracies = []

    for epoch in range(300):
        train_accuracy_metric = tf.keras.metrics.Accuracy()
        test_accuracy_metric = tf.keras.metrics.Accuracy()
        for x_batch_train, y_batch_train in train_dataset:
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            predictions = tf.argmax(logits, axis=1)
            train_accuracy_metric.update_state(y_batch_train, predictions)

        for x_batch_test, y_batch_test in test_dataset:
            logits = model(x_batch_test, training=False)
            predictions = tf.argmax(logits, axis=1)
            test_accuracy_metric.update_state(y_batch_test, predictions)

        train_accuracy = train_accuracy_metric.result().numpy()
        test_accuracy = test_accuracy_metric.result().numpy()
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print(f'Epoch {epoch + 1}: Training Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
        train_accuracy_metric.reset_states()
        test_accuracy_metric.reset_states()

with open('accuracy of LeNet_DropConnect.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Training Accuracy', 'Test Accuracy'])
    for train_acc, test_acc in zip(train_accuracies, test_accuracies):
        writer.writerow([train_acc, test_acc])
