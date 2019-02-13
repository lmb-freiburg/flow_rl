import tensorflow as tf


class Layers:
    def __init__(self, trainable=True):
        self.layer = 0
        self.trainable = trainable

    def conv(self, t, filters, stride=1, kernel_size=3, padding='same', name=None):
        self.layer += 1
        return tf.layers.conv2d(
            inputs=t,
            filters=filters,
            kernel_size=[kernel_size, kernel_size],
            strides=[stride, stride],
            activation=lambda x: tf.nn.leaky_relu(x, 0.1),
            padding=padding,
            name="conv_{}".format(self.layer) if name is None else name,
            trainable=self.trainable
        )

    def upconv(self, t, filters, padding='same', name=None):
        self.layer += 1
        return tf.layers.conv2d_transpose(
            inputs=t,
            filters=filters,
            kernel_size=[4, 4],
            strides=[2, 2],
            activation=lambda x: tf.nn.leaky_relu(x, 0.1),
            padding=padding,
            name="upconv_{}".format(self.layer) if name is None else name,
            trainable=self.trainable
        )


def flow_network(t, trainable=True, size=None):
    output_size = 2
    if t.get_shape().as_list()[-1] == 12:
        output_size = 4

    l = Layers(trainable)

    t = l.conv(t, 64)
    skip1 = t
    t = l.conv(t, 64, 2)
    t = l.conv(t, 128)
    skip05 = t
    t = l.conv(t, 128, 2)
    t = l.conv(t, 128)

    t = l.upconv(t, 32)
    t = tf.concat([t, skip05], 3)
    flow05 = l.conv(t, output_size, name='flow05')
    flow05res = flow05.get_shape().as_list()[2]

    t = l.upconv(t, 16)
    upflow = tf.image.resize_images(
        flow05, [2 * flow05res, 2 * flow05res],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    t = tf.concat([t, skip1, upflow], 3)
    flow = l.conv(t, output_size, name='flow')
    return flow, upflow


def flow_clip_network(t, trainable=True, size=200):
    output_size = 2
    if t.get_shape().as_list()[-1] == 12:
        output_size = 4

    flow, upflow = flow_network(t, trainable, size)

    zeroes = tf.zeros([size, 84, 84, output_size], dtype=tf.float32)
    ones = tf.constant(1., shape=[size, 84, 84, output_size], dtype=tf.float32)

    flow = tf.where(tf.abs(flow) > 0.1, ones, zeroes)

    return flow, upflow


flow_dict = dict(normal=flow_network, clip=flow_clip_network)
