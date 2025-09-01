import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LeakySurrogate(tf.keras.layers.Layer):
    def __init__(self, beta, p_neuron, threshold=1.0):
        super(LeakySurrogate, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.p_neuron = p_neuron

    def call(self, inputs, mem):
        spk = tf.cast(tf.math.greater(mem, self.threshold), tf.float32)
        reset = tf.stop_gradient(self.beta * spk * self.threshold)
        mem = self.beta * mem + inputs - reset
        mask = tf.random.uniform(tf.shape(mem)) < self.p_neuron
        spk *= tf.cast(mask, tf.float32)
        return spk, mem

    def get_config(self):
        base_config = super(LeakySurrogate, self).get_config()
        base_config['beta'] = self.beta
        base_config['threshold'] = self.threshold
        base_config['p_neuron'] = self.p_neuron
        return base_config


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


class SNN_DropConnect(Model):
    def __init__(self):
        super(SNN_DropConnect, self).__init__()
        self.flatten = layers.Flatten()
        self.fc1 = DropConnect(units=2000, rate=0.5, activation='relu')  # keep rate
        self.lif1 = LeakySurrogate(beta=0.9, p_neuron=1)
        self.fc2 = DropConnect(units=10, rate=1, activation='softmax')
        self.lif2 = LeakySurrogate(beta=0.9, p_neuron=1)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        mem1 = tf.zeros([tf.shape(x)[0], 2000])
        mem2 = tf.zeros([tf.shape(x)[0], 10])

        for step in range(8):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
        return cur2, mem2


(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x1 = np.concatenate((x, x_test))
y1 = np.concatenate((y, y_test))
np.random.seed(2023)
train_size = 0.3
index = np.random.rand(len(x1)) < train_size
x, x_test = x1[index], x1[~index]
y, y_test = y1[index], y1[~index]
train_labels = y[0:5000]
test_labels = y_test[0:10000]
train_images = x[0:5000]
test_images = x_test[0:10000]
train_images = train_images[..., tf.newaxis] / 255.0
test_images = test_images[..., tf.newaxis] / 255.0
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(2023).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(2023).batch(128)

model = SNN_DropConnect()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])


class PrintAccuracy(tf.keras.callbacks.Callback):
    def __init__(self):
        super(PrintAccuracy, self).__init__()
        self.train_accuracies = []
        self.test_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_accuracy = round(logs.get('output_2_accuracy', 0), 4)
        self.train_accuracies.append(train_accuracy)
        test_accuracy = round(logs.get('val_output_2_accuracy', 0), 4)
        self.test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch + 1}: Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")


print_accuracy = PrintAccuracy()
history = model.fit(train_dataset, epochs=300, callbacks=[print_accuracy], validation_data=test_dataset, verbose=0)


def save_accuracies_to_csv(save_path, train_accuracies, test_accuracies):
    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Training Accuracy', 'Test Accuracy'])
        for epoch, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies), start=1):
            writer.writerow([round(train_acc, 4), round(test_acc, 4)])


save_path = 'accuracy of SNN_DropConnect.csv'
save_accuracies_to_csv(save_path, print_accuracy.train_accuracies, print_accuracy.test_accuracies)
print(f"Training and test accuracies have been saved to {save_path}")
