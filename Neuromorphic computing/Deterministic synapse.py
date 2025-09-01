"""
Deterministic computing based on the deterministic synapse
No stochasticity is introduced into the model (line 124).
Optimization using L1/L2 regularization (line 142/line 147)
"""

import os
import csv
import math
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set the random seeds
tf.random.set_seed(2023)
np.random.seed(2023)

# hyperparameters
batch_num = 2
batch_size = 2500
error = 1e-4
lr = 2.7e-3
epoch = 300
network_node = [784, 600, 300, 100, 10]

# model parameters
# g_max and g_min are the normalized maximum conductance and minimum conductance, respectively.
# Ap and Ad are the parameters that govern the nonlinear correlation between the conductance and pulse number for LTP and LTD stages, respectively.
g_max_100, g_min_100, Ap_100, Ad_100 = 1, 0.32, 0.43, 1.40  # when p equals 1
Bp_100 = (g_max_100 - g_min_100) / (1 - math.exp(-1 * Ap_100))
Bd_100 = (g_max_100 - g_min_100) / (1 - math.exp(-1 * Ad_100))


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def write_csv(name, matrix):
    if type(matrix) == list:
        if not os.path.exists('%s.csv' % name):
            with open('%s.csv' % name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['training accuracy', 'test accuracy', 'w1', 'w2', 'w3', 'w4', 'Total update count'])
        with open('%s.csv' % name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(matrix)
    else:
        with open('%s.csv' % name, 'w') as file:
            writer = csv.writer(file)
            matrix = tf.convert_to_tensor(matrix)
        np.savetxt('%s.csv' % name, matrix, delimiter=",")


def get_mnist_data():
    (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
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
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_db.shuffle(1000, seed=2023)
    test_db.shuffle(1000, seed=2023)
    train_db = train_db.map(preprocess).batch(batch_size)
    test_db = test_db.map(preprocess).batch(batch_size)
    return train_db, test_db


class Model:
    def __init__(self, network_node):
        self.wp1_changes, self.wp2_changes, self.wp3_changes, self.wp4_changes,  = 0, 0, 0, 0
        self.wm1_changes, self.wm2_changes, self.wm3_changes, self.wm4_changes = 0, 0, 0, 0
        self.w1_changes, self.w2_changes, self.w3_changes, self.w4_changes = 0, 0, 0, 0
        self.output = network_node[4]
        self.layer_num = len(network_node)
        self.save_path = 'model_Deterministic synapse\\device_network_{0}_{1}_{2}_{3}'.format(network_node[0], network_node[1], network_node[2], network_node[3], network_node[4])
        if os.path.exists(self.save_path + '\\wp1.csv'):  # If trained before, load the existing parameters directly.
            self.wp1 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\wp1.csv", delimiter=","), dtype=tf.float32))
            self.wm1 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\wm1.csv", delimiter=","), dtype=tf.float32))
            self.bp1 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\bp1.csv", delimiter=","), dtype=tf.float32))
            self.bm1 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\bm1.csv", delimiter=","), dtype=tf.float32))
            self.wp2 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\wp2.csv", delimiter=","), dtype=tf.float32))
            self.wm2 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\wm2.csv", delimiter=","), dtype=tf.float32))
            self.bp2 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\bp2.csv", delimiter=","), dtype=tf.float32))
            self.bm2 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\bm2.csv", delimiter=","), dtype=tf.float32))
            self.wp3 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\wp3.csv", delimiter=","), dtype=tf.float32))
            self.wm3 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\wm3.csv", delimiter=","), dtype=tf.float32))
            self.bp3 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\bp3.csv", delimiter=","), dtype=tf.float32))
            self.bm3 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\bm3.csv", delimiter=","), dtype=tf.float32))
            self.wp4 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\wp4.csv", delimiter=","), dtype=tf.float32))
            self.wm4 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\wm4.csv", delimiter=","), dtype=tf.float32))
            self.bp4 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\bp4.csv", delimiter=","), dtype=tf.float32))
            self.bm4 = tf.Variable(tf.cast(np.genfromtxt(self.save_path + "\\bm4.csv", delimiter=","), dtype=tf.float32))
            print('model load secessfully!!')
        else:
            self.wp1 = tf.Variable(tf.random.uniform([network_node[0], network_node[1]], minval=g_min_100, maxval=g_max_100))
            self.wm1 = tf.Variable(tf.random.uniform([network_node[0], network_node[1]], minval=g_min_100, maxval=g_max_100))
            self.bp1 = tf.Variable(tf.ones(network_node[1]) / network_node[1])
            self.bm1 = tf.Variable(tf.ones(network_node[1]) / network_node[1])
            self.wp2 = tf.Variable(tf.random.uniform([network_node[1], network_node[2]], minval=g_min_100, maxval=g_max_100))
            self.wm2 = tf.Variable(tf.random.uniform([network_node[1], network_node[2]], minval=g_min_100, maxval=g_max_100))
            self.bp2 = tf.Variable(tf.ones(network_node[2]) / network_node[2])
            self.bm2 = tf.Variable(tf.ones(network_node[2]) / network_node[2])
            self.wp3 = tf.Variable(tf.random.uniform([network_node[2], network_node[3]], minval=g_min_100, maxval=g_max_100))
            self.wm3 = tf.Variable(tf.random.uniform([network_node[2], network_node[3]], minval=g_min_100, maxval=g_max_100))
            self.bp3 = tf.Variable(tf.ones(network_node[3]) / network_node[3])
            self.bm3 = tf.Variable(tf.ones(network_node[3]) / network_node[3])
            self.wp4 = tf.Variable(tf.random.uniform([network_node[3], network_node[4]], minval=g_min_100, maxval=g_max_100))
            self.wm4 = tf.Variable(tf.random.uniform([network_node[3], network_node[4]], minval=g_min_100, maxval=g_max_100))
            self.bp4 = tf.Variable(tf.ones(network_node[4]) / network_node[4])
            self.bm4 = tf.Variable(tf.ones(network_node[4]) / network_node[4])

    # forward propagation
    def enter_network(self, x, y, l1_lambda=1E-5):  # fully connected network. l2_lambda for L2 regularization
        global w1, w2, w3, w4
        w1 = self.wp1 - self.wm1
        b1 = self.bp1 - self.bm1
        w2 = self.wp2 - self.wm2
        b2 = self.bp2 - self.bm2
        w3 = self.wp3 - self.wm3
        b3 = self.bp3 - self.bm3
        w4 = self.wp4 - self.wm4
        b4 = self.bp4 - self.bm4
        h1 = tf.nn.relu(x @ w1 + b1)
        h2 = tf.nn.relu(h1 @ w2 + b2)
        h3 = tf.nn.relu(h2 @ w3 + b3)
        out = h3 @ w4 + b4
        y_onehot = tf.one_hot(y, depth=self.output)
        loss_mse = tf.reduce_mean(tf.losses.MSE(out, y_onehot))
        loss_ce = tf.reduce_sum(tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True))

        # Optimization using L1 regularization
        l_reg = l1_lambda * (tf.reduce_sum(tf.abs(w1)) + tf.reduce_sum(tf.abs(w2)) + tf.reduce_sum(tf.abs(w3)) + tf.reduce_sum(tf.abs(w4)))
        loss_mse += l_reg
        loss_ce += l_reg
        """
        # Optimization using L2 regularization
        l_reg = l2_lambda * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4))
        loss_mse += l_reg
        loss_ce += l_reg
        """
        return out, loss_mse, loss_ce

    def save_weights(self, epoch):
        def save_csv(csv_list, epoch_for_mode, file_name):
            mode = ""
            if epoch_for_mode == 0:
                mode = "w"
            else:
                mode = "a"
            with open(self.save_path + "\\" + file_name + ".csv", mode, newline="", encoding="utf-8") as f:
                write = csv.writer(f, delimiter=",")
                for item in csv_list:
                    write.writerow([item])
        if epoch == 299:
            save_csv(w1.numpy().flatten().tolist(), epoch, "w1-300")
            save_csv(w2.numpy().flatten().tolist(), epoch, "w2-300")
            save_csv(w3.numpy().flatten().tolist(), epoch, "w3-300")
            save_csv(w4.numpy().flatten().tolist(), epoch, "w4-300")

    def loss_gradient(self, x, y):
        def cal_gradient_100w(variable, grad):
            increase = tf.cast(grad > tf.ones([grad.shape[0], grad.shape[1]]) * error, dtype=tf.float32)
            decrease = tf.cast(grad < tf.ones([grad.shape[0], grad.shape[1]]) * -error, dtype=tf.float32)
            increase_gradient = Ap_100 * (Bp_100 - variable + g_min_100)
            decrease_gradient = Ad_100 * (Bd_100 + variable - g_max_100)
            return increase_gradient * increase - decrease_gradient * decrease

        def cal_gradient_100b(variable, grad):
            increase = tf.cast(grad > tf.ones([grad.shape[0]]) * error, dtype=tf.float32)
            decrease = tf.cast(grad < tf.ones([grad.shape[0]]) * -error, dtype=tf.float32)
            increase_gradient = Ap_100 * (Bp_100 - variable + g_min_100)
            decrease_gradient = Ad_100 * (Bd_100 + variable - g_max_100)
            return increase_gradient * increase - decrease_gradient * decrease

        with tf.GradientTape() as tape:
            _, _, loss = self.enter_network(x, y)

        grad_variables = tape.gradient(loss, [self.wp1, self.bp1, self.wp2, self.bp2, self.wp3, self.bp3, self.wp4, self.bp4])

        self.wp1.assign_sub(lr * cal_gradient_100w(self.wp1, grad_variables[0]))
        self.wp2.assign_sub(lr * cal_gradient_100w(self.wp2, grad_variables[2]))
        self.wp3.assign_sub(lr * cal_gradient_100w(self.wp3, grad_variables[4]))
        self.wp4.assign_sub(lr * cal_gradient_100w(self.wp4, grad_variables[6]))

        self.wm1.assign_sub(-lr * cal_gradient_100w(self.wm1, grad_variables[0]))
        self.wm2.assign_sub(-lr * cal_gradient_100w(self.wm2, grad_variables[2]))
        self.wm3.assign_sub(-lr * cal_gradient_100w(self.wm3, grad_variables[4]))
        self.wm4.assign_sub(-lr * cal_gradient_100w(self.wm4, grad_variables[6]))

        self.bp1.assign_sub(lr * cal_gradient_100b(self.bp1, grad_variables[1]))
        self.bp2.assign_sub(lr * cal_gradient_100b(self.bp2, grad_variables[3]))
        self.bp3.assign_sub(lr * cal_gradient_100b(self.bp3, grad_variables[5]))
        self.bp4.assign_sub(lr * cal_gradient_100b(self.bp4, grad_variables[7]))

        self.bm1.assign_sub(-lr * cal_gradient_100b(self.bm1, grad_variables[1]))
        self.bm2.assign_sub(-lr * cal_gradient_100b(self.bm2, grad_variables[3]))
        self.bm3.assign_sub(-lr * cal_gradient_100b(self.bm3, grad_variables[5]))
        self.bm4.assign_sub(-lr * cal_gradient_100b(self.bm4, grad_variables[7]))

        self.wp1_changes += tf.math.count_nonzero(cal_gradient_100w(self.wp1, grad_variables[0]))
        self.wp2_changes += tf.math.count_nonzero(cal_gradient_100w(self.wp2, grad_variables[2]))
        self.wp3_changes += tf.math.count_nonzero(cal_gradient_100w(self.wp3, grad_variables[4]))
        self.wp4_changes += tf.math.count_nonzero(cal_gradient_100w(self.wp4, grad_variables[6]))

        self.wm1_changes += tf.math.count_nonzero(cal_gradient_100w(self.wm1, grad_variables[0]))
        self.wm2_changes += tf.math.count_nonzero(cal_gradient_100w(self.wm2, grad_variables[2]))
        self.wm3_changes += tf.math.count_nonzero(cal_gradient_100w(self.wm3, grad_variables[4]))
        self.wm4_changes += tf.math.count_nonzero(cal_gradient_100w(self.wm4, grad_variables[6]))

        self.w1_changes = self.wp1_changes if self.wp1_changes > self.wm1_changes else self.wm1_changes
        self.w2_changes = self.wp2_changes if self.wp2_changes > self.wm2_changes else self.wm2_changes
        self.w3_changes = self.wp3_changes if self.wp3_changes > self.wm3_changes else self.wm3_changes
        self.w4_changes = self.wp4_changes if self.wp4_changes > self.wm4_changes else self.wm4_changes

        return loss

    def record(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        write_csv(self.save_path + '\\wp1', self.wp1)
        write_csv(self.save_path + '\\wm1', self.wm1)
        write_csv(self.save_path + '\\bp1', self.bp1)
        write_csv(self.save_path + '\\bm1', self.bm1)
        write_csv(self.save_path + '\\wp2', self.wp2)
        write_csv(self.save_path + '\\wm2', self.wm2)
        write_csv(self.save_path + '\\bp2', self.bp2)
        write_csv(self.save_path + '\\bm2', self.bm2)
        write_csv(self.save_path + '\\wp3', self.wp3)
        write_csv(self.save_path + '\\wm3', self.wm3)
        write_csv(self.save_path + '\\bp3', self.bp3)
        write_csv(self.save_path + '\\bm3', self.bm3)
        write_csv(self.save_path + '\\wp4', self.wp4)
        write_csv(self.save_path + '\\wm4', self.wm4)
        write_csv(self.save_path + '\\bp4', self.bp4)
        write_csv(self.save_path + '\\bm4', self.bm4)


def model_train(epoch, train_db):
    train_db = train_db.shuffle(100)
    epoch_loss = []
    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, (-1, 784))
        loss = model.loss_gradient(x, y)
        epoch_loss.append(float(loss))
    average_loss = np.mean(epoch_loss)
    average_loss = '%.4f' % average_loss
    model.record()
    return average_loss


def model_test(test_db):
    total_correct, total_num = 0, 0
    for (x, y) in test_db:
        x = tf.reshape(x, [-1, 784])
        out, _, _, = model.enter_network(x, y)
        prob = tf.nn.softmax(out, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_num += x.shape[0]
    accuracy = total_correct / total_num
    accuracy = '%.4f' % accuracy
    return accuracy


if __name__ == '__main__':
    train_db, test_db = get_mnist_data()
    model = Model(network_node)
    (x_test, y_test) = next(iter(test_db))
    for epoch in range(epoch):
        x_test = tf.reshape(x_test, (-1, 784))
        y_predict, _, _ = model.enter_network(x_test, y_test)
        accuracy1 = model_test(train_db)
        accuracy2 = model_test(test_db)
        loss = model_train(epoch, train_db)
        print('Epoch {0}: training accuracy：{1}，test accuracy：{2}'.format(epoch + 1, accuracy1, accuracy2))
        write_csv(model.save_path + '\\benchmark_Deterministic synapse', [accuracy1, accuracy2, model.w1_changes.numpy(), model.w2_changes.numpy(), model.w3_changes.numpy(), model.w4_changes.numpy(), model.w1_changes.numpy()+model.w2_changes.numpy()+model.w3_changes.numpy()+model.w4_changes.numpy()])
        model.save_weights(epoch)

x_test = tf.reshape(x_test, (-1, 784))
y_predict, _, _ = model.enter_network(x_test, y_test)
y_pred_labels = np.argmax(y_predict, axis=1)
confusion_mtx = confusion_matrix(y_test, y_pred_labels)
np.savetxt("model_Deterministic synapse/confusion_matrix.txt", confusion_mtx, fmt="%d")
total = 0
for index in range(10):
    total += confusion_mtx[index][index]
print(total)
