import tensorflow as tf
from PIL import Image
from cnn_construction import simple_cnn
import numpy as np
import cv2
import os

def use(path_test = "pictureForTest/test.jpeg"):
    # 将所有的图片resize成100*100
    w = 100
    h = 100
    c = 3
    classes = ['baby', 'teenager', 'younger', 'adult', 'older']

    image_test = Image.open(path_test)
    resized_image = image_test.resize((w, h), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')

    imgs_holder = tf.placeholder(tf.float32, shape=[1, w, h, c])

    logits, pred = simple_cnn(imgs_holder)

    saver = tf.train.Saver()
    ckpt_dir = './model/'

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        classes_ = sess.run(pred, feed_dict={imgs_holder: np.reshape(image_data, [1, w, h, c])})

    num = np.argmax(classes_)

    # 在图片上添加文字 1
    var = classes[int(num)]

    # 加载背景图片
    bk_img = cv2.imread(path_test)
    # 在图片上添加文字信息
    cv2.putText(bk_img, var, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 1, cv2.LINE_AA)
    # 显示图片
    cv2.imshow("add_text", bk_img)
    cv2.waitKey()
    os.remove(path_test)


# use()
