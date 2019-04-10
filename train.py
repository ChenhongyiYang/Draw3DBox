import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np
import time
import build_model
import data_process



DIRECTIONS = 2
H, W  = 224, 224
BATCH_SIZE = 8
epochs = 101
ckpt_path = 'vgg_16.ckpt'


inputs = build_model.inputs
c_label = build_model.c_label
t1_label = build_model.t1_label
t2_label = build_model.t2_label
rs_label = build_model.rs_label

def get_n_cores():
  nslots = os.getenv('NSLOTS')
  if nslots is not None:
    return int(nslots)
  raise ValueError('Environment variable NSLOTS is not defined.')


def train(save_path,image_dir,label_dir):
    all_objs = data_process.parse_annotation(label_dir, image_dir)
    all_exams = len(all_objs)
    np.random.shuffle(all_objs)
    train_gen = data_process.data_gen(all_objs, BATCH_SIZE)
    train_num = int(np.ceil(all_exams / BATCH_SIZE))

    rs, theta1, theta2, probability, total_loss, optimizer, loss_r, loss_theta1, loss_theta2, loss_c = build_model.build_model()

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=get_n_cores() - 1,
        inter_op_parallelism_threads=1,
        allow_soft_placement=True,
        log_device_placement=True)
    #session_conf.gpu_options.allow_growth = True

    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)

    with tf.Session(config=session_conf) as sess:
        variables_to_restore = slim.get_variables()[:26]
        saver = tf.train.Saver()

        ckpt_list = tf.contrib.framework.list_variables(ckpt_path)[1:-7]
        for name in range(1, len(ckpt_list), 2):
            tf.contrib.framework.init_from_checkpoint(ckpt_path, {ckpt_list[name - 1][0]: variables_to_restore[name]})
            tf.contrib.framework.init_from_checkpoint(ckpt_path, {ckpt_list[name][0]: variables_to_restore[name - 1]})
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(epochs):
            epoch_loss = np.zeros((train_num), dtype=float)
            epoch_loss_r = np.zeros((train_num), dtype=float)
            epoch_loss_t1 = np.zeros((train_num), dtype=float)
            epoch_loss_t2 = np.zeros((train_num), dtype=float)
            epoch_loss_c = np.zeros((train_num), dtype=float)

            tStart_epoch = time.time()
            batch_loss = 0.0
            for num_iters in range(train_num):
                train_img, train_label = train_gen.__next__()
                feed_dict = {inputs:train_img,
                             rs_label:train_label[0],
                             t1_label:train_label[1],
                             t2_label:train_label[2],
                             c_label:train_label[3]}
                _loss_r,_loss_t1,_loss_t2,_loss_c,_loss,_ = sess.run([loss_r, loss_theta1, loss_theta2, loss_c,total_loss,optimizer],feed_dict=feed_dict)

                epoch_loss[num_iters] = _loss
                epoch_loss_r[num_iters] = _loss_r
                epoch_loss_t1[num_iters] = _loss_t1
                epoch_loss_t2[num_iters] = _loss_t2
                epoch_loss_c[num_iters] = _loss_c

            if (epoch + 1) % 5 == 0:
                saver.save(sess, save_path + "/model", global_step=epoch + 1)

            fl = open('log2.txt', 'a')
            tStop_epoch = time.time()
            fl.write('Epoch:%d\tLoss:%.5f\tLoss_rs:%.5f\tLoss_t1:%.5f\tLoss_t2:%.5f\tLoss_c:%.5f\tTime_cost:%.2f\n' % (epoch + 1, np.mean(epoch_loss,axis=0), np.mean(epoch_loss_r,axis=0),np.mean(epoch_loss_t1,axis=0),np.mean(epoch_loss_t2,axis=0),np.mean(epoch_loss_c,axis=0),round(tStop_epoch - tStart_epoch, 2)))
            fl.close()



if __name__ == '__main__':
    save_path = 'model2/'
    image_dir = 'training/image_2'
    label_dir = 'label/new_label'



    train(save_path,image_dir,label_dir)

































