'''
Stochastic computing with Strategy 2
The retention probabilities of weights for each layer are set to 0.45, 0.45, 0.8 and 1, respectively. (line 287)
The stochasticity is turned on during the forward (line 131, line 134, line 137) and backpropagation (line 221-223, line 224-226).
'''
import os
import csv
import time
import math
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# hyperparameters
batch_num = 2
batch_size = 2500
error = 1e-4
lr = 2.6e-3
learning_epoch = 400
network_node = [784, 600, 300, 100, 10]

# model parameters
g_max, g_min, Ap, Ad = 1, 0.32, 0.51, 0.94
# g_max and g_min are the normalized maximum conductance and minimum conductance, respectively.
# Ap and Ad are the parameters that govern the nonlinear correlation between the conductance and pulse number for LTP and LTD stages, respectively.
Bp = (g_max - g_min) / (1 - math.exp(-1 * Ap))
Bd = (g_max - g_min) / (1 - math.exp(-1 * Ad))

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def write_csv(name, matrix):
    if type(matrix) == list:
        if not os.path.exists('%s.csv' % name):
            with open('%s.csv' % name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['train accuracy', 'test accuracy', 'w1', 'w2', 'w3', 'w4 update count'])
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
    train_db.shuffle(1000)
    test_db.shuffle(1000)
    train_db = train_db.map(preprocess).batch(batch_size)
    test_db = test_db.map(preprocess).batch(batch_size)
    return train_db, test_db

class Model:
    def __init__(self, network_node, p1, p2, p3, p4):
        self.p1, self.p2, self.p3, self.p4 = p1, p2, p3, p4
        self.wp1_changes, self.wp2_changes, self.wp3_changes, self.wp4_changes,  = 0, 0, 0, 0
        self.wm1_changes, self.wm2_changes, self.wm3_changes, self.wm4_changes = 0, 0, 0, 0
        self.w1_changes, self.w2_changes, self.w3_changes, self.w4_changes = 0, 0, 0, 0
        self.output = network_node[4]
        self.layer_num = len(network_node)
        self.save_path = 'model_SC2\\device_network_{0}_{1}_{2}_{3}'.format(network_node[0], network_node[1], network_node[2], network_node[3], network_node[4])
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
        else:
            self.wp1 = tf.Variable(tf.random.uniform([network_node[0], network_node[1]], minval=g_min, maxval=g_max))
            self.wm1 = tf.Variable(tf.random.uniform([network_node[0], network_node[1]], minval=g_min, maxval=g_max))
            self.bp1 = tf.Variable(tf.ones(network_node[1]) / network_node[1])
            self.bm1 = tf.Variable(tf.ones(network_node[1]) / network_node[1])
            self.wp2 = tf.Variable(tf.random.uniform([network_node[1], network_node[2]], minval=g_min, maxval=g_max))
            self.wm2 = tf.Variable(tf.random.uniform([network_node[1], network_node[2]], minval=g_min, maxval=g_max))
            self.bp2 = tf.Variable(tf.ones(network_node[2]) / network_node[2])
            self.bm2 = tf.Variable(tf.ones(network_node[2]) / network_node[2])
            self.wp3 = tf.Variable(tf.random.uniform([network_node[2], network_node[3]], minval=g_min, maxval=g_max))
            self.wm3 = tf.Variable(tf.random.uniform([network_node[2], network_node[3]], minval=g_min, maxval=g_max))
            self.bp3 = tf.Variable(tf.ones(network_node[3]) / network_node[3])
            self.bm3 = tf.Variable(tf.ones(network_node[3]) / network_node[3])
            self.wp4 = tf.Variable(tf.random.uniform([network_node[3], network_node[4]], minval=g_min, maxval=g_max))
            self.wm4 = tf.Variable(tf.random.uniform([network_node[3], network_node[4]], minval=g_min, maxval=g_max))
            self.bp4 = tf.Variable(tf.ones(network_node[4]) / network_node[4])
            self.bm4 = tf.Variable(tf.ones(network_node[4]) / network_node[4])

    # forward propagation
    def enter_network(self, x, y):  # drop-connected network
        w1 = self.wp1 - self.wm1
        b1 = self.bp1 - self.bm1
        w2 = self.wp2 - self.wm2
        b2 = self.bp2 - self.bm2
        w3 = self.wp3 - self.wm3
        b3 = self.bp3 - self.bm3
        w4 = self.wp4 - self.wm4
        b4 = self.bp4 - self.bm4
        global mask1
        global mask2
        global mask3
        global mask4
        mask1 = tf.cast(tf.random.uniform([w1.shape[0], w1.shape[1]]) <= self.p1, tf.float32)
        w1 = w1 * mask1 / self.p1  # the stochasticity is on during forward propagation (p1=0.45, p2=0.45, p3=0.8, p4=1)
        h1 = tf.nn.relu(x @ w1 + b1)
        mask2 = tf.cast(tf.random.uniform([w2.shape[0], w2.shape[1]]) <= self.p2, tf.float32)
        w2 = w2 * mask2 / self.p2
        h2 = tf.nn.relu(h1 @ w2 + b2)
        mask3 = tf.cast(tf.random.uniform([w3.shape[0], w3.shape[1]]) <= self.p3, tf.float32)
        w3 = w3 * mask3 / self.p3
        h3 = tf.nn.relu(h2 @ w3 + b3)
        mask4 = tf.cast(tf.random.uniform([w4.shape[0], w4.shape[1]]) <= self.p4, tf.float32)
        w4 = w4 * mask4 / self.p4
        out = h3 @ w4 + b4
        y_onehot = tf.one_hot(y, depth=self.output)
        loss_mse = tf.reduce_mean(tf.losses.MSE(out, y_onehot))
        loss_ce = tf.reduce_sum(tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True))
        return out, loss_mse, loss_ce

    def enter_network2(self, x, y):  # fully connected network
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
        if epoch == 50:
            w2 = (self.wp2 - self.wm2) * mask2
            save_csv(w2.numpy().flatten().tolist(), epoch, "w2-50")
        elif epoch == 100:
            w2 = (self.wp2 - self.wm2) * mask2
            save_csv(w2.numpy().flatten().tolist(), epoch, "w2-100")
        elif epoch == 150:
            w2 = (self.wp2 - self.wm2) * mask2
            save_csv(w2.numpy().flatten().tolist(), epoch, "w2-150")
        elif epoch == 199:
            w2 = (self.wp2 - self.wm2) * mask2
            save_csv(w2.numpy().flatten().tolist(), epoch, "w2-200")

    def loss_gradient(self, x, y)  :
        def cal_gradient(variable, grad):
            increase = tf.cast(grad > tf.ones([grad.shape[0], grad.shape[1]])*error, dtype=tf.float32)
            decrease = tf.cast(grad < tf.ones([grad.shape[0], grad.shape[1]])*-error, dtype=tf.float32)
            increase_gradient = Ap*(Bp - variable + g_min)
            decrease_gradient = Ad*(Bd + variable - g_max)
            return increase_gradient * increase - decrease_gradient * decrease
        def cal_gradient2(variable, grad):
            increase = tf.cast(grad > tf.ones([grad.shape[0]])*error, dtype=tf.float32)
            decrease = tf.cast(grad < tf.ones([grad.shape[0]])*-error, dtype=tf.float32)
            increase_gradient = Ap*(Bp - variable + g_min)
            decrease_gradient = Ad*(Bd + variable - g_max)
            return increase_gradient * increase - decrease_gradient * decrease
        with tf.GradientTape() as tape:
            _, _, loss = self.enter_network(x, y)

        grad_variables = tape.gradient(loss, [self.wp1, self.bp1, self.wp2, self.bp2, self.wp3, self.bp3, self.wp4, self.bp4])
        self.wp1_changes += tf.math.count_nonzero(cal_gradient(self.wp1, grad_variables[0]))
        self.wp2_changes += tf.math.count_nonzero(cal_gradient(self.wp2, grad_variables[2]))
        self.wp3_changes += tf.math.count_nonzero(cal_gradient(self.wp3, grad_variables[4]))
        self.wp4_changes += tf.math.count_nonzero(cal_gradient(self.wp4, grad_variables[6]))
        self.wm1_changes += tf.math.count_nonzero(cal_gradient(self.wm1, grad_variables[0]))
        self.wm2_changes += tf.math.count_nonzero(cal_gradient(self.wm2, grad_variables[2]))
        self.wm3_changes += tf.math.count_nonzero(cal_gradient(self.wm3, grad_variables[4]))
        self.wm4_changes += tf.math.count_nonzero(cal_gradient(self.wm4, grad_variables[6]))
        self.w1_changes = self.wp1_changes if self.wp1_changes > self.wm1_changes else self.wm1_changes
        self.w2_changes = self.wp2_changes if self.wp2_changes > self.wm2_changes else self.wm2_changes
        self.w3_changes = self.wp3_changes if self.wp3_changes > self.wm3_changes else self.wm3_changes
        self.w4_changes = self.wp4_changes if self.wp4_changes > self.wm4_changes else self.wm4_changes

        # back propagation
        self.wp1.assign_sub(lr * cal_gradient(self.wp1, grad_variables[0])*0.45)  # the stochasticity is turned on during backpropagation
        self.wp2.assign_sub(lr * cal_gradient(self.wp2, grad_variables[2])*0.45)
        self.wp3.assign_sub(lr * cal_gradient(self.wp3, grad_variables[4])*0.8)
        self.wp4.assign_sub(lr * cal_gradient(self.wp4, grad_variables[6]))
        self.wm1.assign_sub(-lr * cal_gradient(self.wm1, grad_variables[0])*0.45)
        self.wm2.assign_sub(-lr * cal_gradient(self.wm2, grad_variables[2])*0.45)
        self.wm3.assign_sub(-lr * cal_gradient(self.wm3, grad_variables[4])*0.8)
        self.wm4.assign_sub(-lr * cal_gradient(self.wm4, grad_variables[6]))
        self.bp1.assign_sub(lr * cal_gradient2(self.bp1, grad_variables[1]))
        self.bp2.assign_sub(lr * cal_gradient2(self.bp2, grad_variables[3]))
        self.bp3.assign_sub(lr * cal_gradient2(self.bp3, grad_variables[5]))
        self.bp4.assign_sub(lr * cal_gradient2(self.bp4, grad_variables[7]))
        self.bm1.assign_sub(-lr * cal_gradient2(self.bm1, grad_variables[1]))
        self.bm2.assign_sub(-lr * cal_gradient2(self.bm2, grad_variables[3]))
        self.bm3.assign_sub(-lr * cal_gradient2(self.bm3, grad_variables[5]))
        self.bm4.assign_sub(-lr * cal_gradient2(self.bm4, grad_variables[7]))
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
    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, (-1, 784))
        loss = model.loss_gradient(x, y)
        if step % 500 == 0:
            print(epoch + 1, step+1, 'loss:', float(loss))
    model.record()

def model_test(test_db):
    total_correct, total_num = 0, 0
    # inference
    for (x, y) in test_db:
        x = tf.reshape(x, [-1, 784])
        out, _, _, = model.enter_network2(x, y)  # the stochasticity is turned off during inference
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
    model = Model(network_node, 0.45, 0.45, 0.8, 1)  # stochastic computing
    (x_test, y_test) = next(iter(test_db))
    time0 = time.time()
    for epoch in range(learning_epoch):
        time1 = time.time()
        model_train(epoch, train_db)
        time2 = time.time()
        print('training time：{0}， total time：{1}'.format(time2 - time1, time2 - time0))
        accuracy1 = model_test(train_db)
        accuracy2 = model_test(test_db)
        print('training accuracy：{0}，test accuracy：{1}'.format(accuracy1, accuracy2))
        write_csv(model.save_path + '\\accuracy', [accuracy1, accuracy2, model.w1_changes.numpy(), model.w2_changes.numpy(), model.w3_changes.numpy(), model.w4_changes.numpy()])
        model.save_weights(epoch)

x_test = tf.reshape(x_test, (-1, 28 * 28))
y_predict, _, _ = model.enter_network(x_test, y_test)

# confusion matrix
y_pred_labels = np.argmax(y_predict, axis=1)
confusion_mtx = confusion_matrix(y_test, y_pred_labels)
np.savetxt("confusion_matrix_SC2.txt", confusion_mtx, fmt="%d")
total = 0
for index in range(10):
    total += confusion_mtx[index][index]
print(total)
print("The confusion matrix is saved as a text file：confusion_matrix_SC2.txt")

plt.figure(figsize=(10, 8))
plt.imshow(confusion_mtx, cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
