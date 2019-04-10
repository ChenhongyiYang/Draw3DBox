import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim



DIRECTIONS = 2
learning_rate = 0.0001

inputs = tf.placeholder(tf.float32,[None,224,224,3],'img_input')
c_label = tf.placeholder(tf.int32,[None,DIRECTIONS],'confidence_label')
t1_label = tf.placeholder(tf.float32,[None,DIRECTIONS,2],'theta1_label')
t2_label = tf.placeholder(tf.float32,[None,DIRECTIONS,2],'theta2_label')
rs_label = tf.placeholder(tf.float32,[None,DIRECTIONS],'rs_label')




def build_model():
    def leaky_relu(x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
        #####
        # Build Graph

    def loss_angle_l2(y_true,y_pred,c_true):
        c_mask = tf.cast(tf.greater(c_true, 0), tf.float32)
        l2_loss = tf.square(tf.subtract(y_true, y_pred))
        l2_loss = tf.multiply(c_mask, l2_loss)
        l2_loss = tf.reduce_sum(l2_loss, axis=1)
        l2_loss_mean = tf.reduce_mean(l2_loss, axis=0)
        return l2_loss_mean

    def loss_angle_cos(y_true,y_pred,c_true):
        c_mask = tf.cast(tf.greater(c_true,0),tf.float32)
        cos_loss = tf.reduce_sum(tf.multiply(y_true,y_pred),axis=2)
        cos_loss = tf.multiply(c_mask,cos_loss)
        cos_loss = tf.reduce_sum(cos_loss,axis=1)
        cos_loss_mean = -1 * tf.reduce_mean(cos_loss,axis=0)
        return cos_loss_mean

    def loss_confidence(c_true,c_pred):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=c_true,logits=c_pred)
        loss_mean = tf.reduce_mean(loss,axis=0)
        return loss_mean

    def loss_rs(rs_true,rs_pred,c_true):
        c_mask = tf.cast(tf.greater(c_true, 0),tf.float32)
        l2_loss = tf.square(tf.subtract(rs_true,rs_pred))
        l2_loss = tf.multiply(c_mask,l2_loss)
        l2_loss = tf.reduce_sum(l2_loss,axis=1)
        l2_loss_mean = tf.reduce_mean(l2_loss,axis=0)
        return l2_loss_mean




    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        conv5 = tf.contrib.layers.flatten(net)

        rs = slim.fully_connected(conv5,512,activation_fn=None,scope='fc7_rs')
        rs = leaky_relu(rs,0.1)
        rs = slim.dropout(rs, 0.5, scope='dropout7_rs')
        rs = slim.fully_connected(rs, DIRECTIONS, activation_fn=None, scope='fc8_rs')
        loss_r = loss_rs(rs_label,rs,c_label)

        theta1 = slim.fully_connected(conv5,256,activation_fn=None,scope='fc7_theta1')
        theta1 = leaky_relu(theta1,0.1)
        theta1 = slim.dropout(theta1, 0.5, scope='dropout7_theta1')
        theta1 = slim.fully_connected(theta1, DIRECTIONS*2, activation_fn=None, scope='fc8_theta1')
        theta1 = tf.reshape(theta1,[-1,DIRECTIONS,2])
        theta1 = tf.nn.l2_normalize(theta1, dim=2)
        loss_theta1 = loss_angle_cos(t1_label,theta1,c_label)

        theta2 = slim.fully_connected(conv5, 256, activation_fn=None, scope='fc7_theta2')
        theta2 = leaky_relu(theta2, 0.1)
        theta2 = slim.dropout(theta2, 0.5, scope='dropout7_theta2')
        theta2 = slim.fully_connected(theta2, DIRECTIONS*2, activation_fn=None, scope='fc8_theta2')
        theta2 = tf.reshape(theta2, [-1, DIRECTIONS, 2])
        theta2 = tf.nn.l2_normalize(theta2, dim=2)
        loss_theta2 = loss_angle_cos(t2_label, theta2, c_label)

        confidence = slim.fully_connected(conv5, 256, activation_fn=None, scope='fc7_confidence')
        confidence = leaky_relu(confidence, 0.1)
        confidence = slim.dropout(confidence, 0.5, scope='dropout7_confidence')
        confidence = slim.fully_connected(confidence, DIRECTIONS, activation_fn=None, scope='fc8_confidence')
        loss_c = loss_confidence(c_label,confidence)

        probability = tf.nn.softmax(confidence)


        total_loss = loss_r + loss_theta1 + loss_theta2 + loss_c
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        return rs, theta1, theta2, probability, total_loss, optimizer, loss_r, loss_theta1, loss_theta2, loss_c

































