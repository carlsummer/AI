import tensorflow as tf
import CifarData
import network
import numpy as np
import os

# 要运行这个模型需要先初始化变量
init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100

output_summary_every_steps = 100
output_model_every_steps = 100

saver = tf.train.Saver()
model_name = "okp-04300"
model_path = os.path.join(network.model_dir,model_name)


# 通过sess运行
with tf.Session() as sess:
    sess.run(init)  # 运行初始化参数
    train_writer = tf.summary.FileWriter(network.train_log_dir,sess.graph)
    test_writer = tf.summary.FileWriter(network.test_log_dir)

    fixed_test_batch_data,fixed_test_batch_labels = CifarData.test_data.next_batch(batch_size)

    if os.path.exists(model_path + ".index"):
        saver.restore(sess,model_path)
        print("model restored from %s" % model_path)
    else:
        print("model %s does not exist" % model_path)

    for i in range(train_steps):
        batch_data, batch_labels = CifarData.train_data.next_batch(batch_size)
        eval_ops = [network.loss, network.accuracy, network.train_op]
        should_output_summary = ((i+1) % output_summary_every_steps == 0)
        if should_output_summary:
            eval_ops.append(network.merged_summary)
        # 要计算的参数，feed_dict是要传递的参数的值x和y的值
        eval_ops_results = sess.run(
            eval_ops,
            feed_dict={network.x: batch_data, network.y: batch_labels})
        loss_val,acc_val = eval_ops_results[0:2]
        if should_output_summary:
            train_summary_str = eval_ops_results[-1]
            train_writer.add_summary(train_summary_str,i+1)
            test_summary_str = sess.run([network.merged_summary_test],
                                        feed_dict={
                                            network.x:fixed_test_batch_data,
                                            network.y:fixed_test_batch_labels
                                        })[0]
            test_writer.add_summary(test_summary_str,i+1)

        if (i + 1) % 500 == 0:
            print('[Train] Step: %d,loss:%4.5f,acc:%4.5f' % (i + 1, loss_val, acc_val))
        if (i + 1) % 5000 == 0:
            test_data = CifarData.CifarData(CifarData.test_filenames, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run([network.accuracy],
                                        feed_dict={network.x: test_batch_data, network.y: test_batch_labels})
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print("[Test ] Step: %d,acc: %4.5f" % (i + 1, test_acc))
        if(i+1) % output_model_every_steps == 0:
            saver.save(sess,os.path.join(network.model_dir,'okp-%05d' % (i+1)))
            print("model saved to ckp-%05d" % (i+1))