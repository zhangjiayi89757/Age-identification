import tensorflow as tf

# CNN网络由4个卷积层，两个全连接层，一个softmax层组成
# 在每一层的卷积后面加入了batch_normalization，relu和池化
# batch_normalization层，有效的预防了梯度消逝和爆炸，还加速了收敛

def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
    return tf.layers.batch_normalization(x,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         scale=True,
                                         training=train,
                                         name=name)


def simple_cnn(x):
    # 第一个卷积层（100——>50)
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    conv1 = batch_norm(conv1, name='pw_bn1')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二个卷积层(50->25)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    conv2 = batch_norm(conv2, name='pw_bn2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 第三个卷积层(25->12)
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    conv3 = batch_norm(conv3, name='pw_bn3')

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # 第四个卷积层(12->6)
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)) # 0.01
    conv4 = batch_norm(conv4, name='pw_bn4')

    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2) # 2

    re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),# 0.01
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
    dense2 = tf.layers.dense(inputs=dense1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
    logits = tf.layers.dense(inputs=dense2,
                             units=5,
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
    pred = tf.nn.softmax(logits, name='prob')
    return logits, pred
