"""
In-situ stochastic computing strategy (Corresponding to Noisy DropConnect method)
The stochasticity only being turned on in the forward phase, which perfectly implements the idea of Dropout.
"""
import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DropConnect_noise(layers.Layer):
    def __init__(self, units, rate, activation=None, **kwargs):
        super(DropConnect_noise, self).__init__(**kwargs)
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
            kernel_noise = tf.cast(kernel / 6, tf.float32)
            kernel_mask = tf.where(mask, kernel_scale, kernel_noise)
            output = tf.matmul(inputs, kernel_mask) + bias
            if self.activation:
                output = self.activation(output)
        else:
            output = tf.matmul(inputs, kernel) + bias
            if self.activation:
                output = self.activation(output)

        return output


class AlexNet_This_work(Model):
    def __init__(self):
        super(AlexNet_This_work, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(96, 3, activation='relu', padding='same', input_shape=(28, 28, 1))
        self.pool1 = tf.keras.layers.MaxPooling2D(2)
        self.conv2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(2)
        self.conv3 = tf.keras.layers.Conv2D(384, 3, activation='relu', padding='same')
        self.conv4 = tf.keras.layers.Conv2D(384, 3, activation='relu', padding='same')
        self.conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.pool3 = tf.keras.layers.MaxPooling2D(2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = DropConnect_noise(units=512, rate=1, activation='relu')
        self.fc2 = DropConnect_noise(units=128, rate=0.1, activation='relu')  # keep rate
        self.fc3 = DropConnect_noise(units=10, rate=1, activation='softmax')

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
    (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # tf.keras.datasets.fashion_mnist.load_data()
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
    x = x[0:5000]
    y = y[0:5000]
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
    model = AlexNet_This_work()
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

with open('accuracy of AlexNet_This work.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Training Accuracy', 'Test Accuracy'])
    for train_acc, test_acc in zip(train_accuracies, test_accuracies):
        writer.writerow([train_acc, test_acc])
