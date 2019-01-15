import tensorflow as tf
import CifarData
import network
import numpy as np

# 要运行这个模型需要先初始化变量
init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100
# 通过sess运行
with tf.Session() as sess:
    sess.run(init)  # 运行初始化参数
    for i in range(train_steps):
        batch_data, batch_labels = CifarData.train_data.next_batch(batch_size)
        # 要计算的参数，feed_dict是要传递的参数的值x和y的值
        loss_val, acc_val, _ = sess.run(
            [network.loss, network.accuracy, network.train_op],
            feed_dict={network.x: batch_data, network.y: batch_labels,network.is_training:True})
        if (i + 1) % 500 == 0:
            print('[Train] Step: %d,loss:%4.5f,acc:%4.5f' % (i + 1, loss_val, acc_val))
        if (i + 1) % 5000 == 0:
            test_data = CifarData.CifarData(CifarData.test_filenames, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run([network.accuracy],
                                        feed_dict={network.x: test_batch_data, network.y: test_batch_labels,network.is_training:False})
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print("[Test ] Step: %d,acc: %4.5f" % (i + 1, test_acc))
