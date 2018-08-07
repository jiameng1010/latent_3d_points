import tensorflow as tf
layers = tf.contrib.layers

def generate_cloud(feature, noise):
    feature = tf.tile(feature, [1, 1024, 1])#2

    #noise = tf.concat([noise, noise], axis=1)#2
    #noise = tf.concat([noise, noise], axis=1)#4
    #noise = tf.concat([noise, noise], axis=1)#8
    #noise = tf.concat([noise, noise], axis=1)#16
    #noise = tf.concat([noise, noise], axis=1)#32
    #noise = tf.concat([noise, noise], axis=1)#64
    #noise = tf.concat([noise, noise], axis=1)#128
    #noise = tf.concat([noise, noise], axis=1)#256
    #noise = tf.concat([noise, noise], axis=1)#512
    #noise = tf.concat([noise, noise], axis=1)#1024

    feature = tf.concat([feature, noise], axis=2)
    point = layers.fully_connected(feature, 256)#, activation_fn=tf.nn.leaky_relu)
    point = layers.dropout(point, keep_prob=0.8)
    point = layers.fully_connected(point, 128)#, activation_fn=tf.nn.leaky_relu)
    point = layers.dropout(point, keep_prob=0.8)
    #point = layers.fully_connected(point, 128)#, activation_fn=tf.nn.leaky_relu)
    #point = layers.dropout(point, keep_prob=0.8)
    #point = layers.fully_connected(point, 32)#, activation_fn=tf.nn.leaky_relu)
    point = layers.fully_connected(point, 128)#, activation_fn=tf.nn.leaky_relu)
    #point = layers.fully_connected(point, 16)#, activation_fn=tf.nn.leaky_relu)
    point = layers.fully_connected(point, 3, activation_fn=tf.nn.tanh)
    point = tf.reshape(point, [point.get_shape()[0].value, 3*point.get_shape()[1].value, -1])
    point = tf.squeeze(point)

    return point

def decoder_generator(inputs):
    noise = inputs
    #with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
    with tf.contrib.framework.arg_scope(
            [layers.fully_connected, layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm):
        net = layers.fully_connected(noise, 64)#, activation_fn=tf.nn.leaky_relu)
        net = layers.fully_connected(net, 128)#, activation_fn=tf.nn.leaky_relu)
        net = layers.fully_connected(net, 256)#, activation_fn=tf.nn.leaky_relu)
        net = layers.fully_connected(net, 512)#, activation_fn=tf.nn.leaky_relu)
        feature = layers.fully_connected(net, 1024)

    noise2 = tf.random_normal([noise.get_shape()[0].value, 1024, 16])
    #noise2 = tf.to_float(tf.constant(np.zeros([32, 1024, 128])))
    cloud = generate_cloud(tf.expand_dims(feature, axis=1), noise2)

    return cloud