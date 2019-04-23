import tensorflow as tf
import cnn_construction as cnn
import data_processing as dp

# 将所有的图片resize成100*100
w = 100
h = 100
c = 3
path = 'data'

x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

logits, pred = cnn.simple_cnn(x)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

data, label = dp.read_img(path)
x_train, y_train, x_val, y_val = dp.suffer(data, label)

# 训练和测试数据
n_epoch = 21
batch_size = 16


def train():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch in range(n_epoch):
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in dp.minibatches(x_train, y_train, batch_size, shuffle=True):
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch += 1

        print('Epoch %d - train loss: %f' % (epoch, (train_loss / n_batch)))
        print('Epoch %d - train acc: %f' % (epoch, train_acc / n_batch))

        # validation
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in dp.minibatches(x_val, y_val, batch_size, shuffle=False):
            err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err
            val_acc += ac
            n_batch += 1
        print('Epoch %d - Validation loss: %f' % (epoch, val_loss / n_batch))
        print('Epoch %d - Validation Accuracy: %f' % (epoch, (val_acc / n_batch)))
        # 每隔5次保存一次模型
        if epoch % 5 == 0:
            saver.save(sess, "./model/save_net.ckpt", epoch)
            print('Trained Model Saved.')


train()
