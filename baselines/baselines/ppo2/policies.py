import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype


def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """

    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.

    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


def mujoco_cnn(unscaled_images, name, nbatch, add_flownet,
               unscaled_previous_images, flownet,
               train_from_scratch,
               large_cnn, diff_frames):
    """
    CNN from Nature paper.
    """
    # scaling image input
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.

    if add_flownet:
        # concatenating scaled_images with flow from flownet

        assert unscaled_previous_images is not None
        assert flownet is not None

        scaled_previous_images = tf.cast(unscaled_previous_images, tf.float32) / 255.

        img_stack = tf.concat([scaled_previous_images, scaled_images], axis=3)

        flow, _ = flownet(img_stack, trainable=train_from_scratch, size=nbatch)

        if not train_from_scratch:
            flow = tf.stop_gradient(flow)

        scaled_images = tf.concat([flow, scaled_images], axis=3)

    if diff_frames:
        half_size = scaled_images.get_shape().as_list()[-1] // 2
        img, pre_img = tf.split(scaled_images, [half_size, half_size], axis=3)
        pre_img = pre_img - img
        scaled_images = tf.concat([img, pre_img], axis=3)

    activ = tf.nn.relu

    if not large_cnn:
        h = activ(conv(scaled_images, name + '_c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
        h2 = activ(conv(h, name + '_c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
        h3 = activ(conv(h2, name + '_c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
        return conv_to_fc(h3)
    else:
        h = activ(conv(scaled_images, name + '_c1', nf=32, rf=3, stride=1, init_scale=np.sqrt(2)))
        skip = conv(h, name + '_c2', nf=64, rf=3, stride=2, init_scale=np.sqrt(2))
        h = activ(skip)
        h = activ(conv(h, name + '_c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), pad='SAME'))
        h = activ(conv(h, name + '_c4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), pad='SAME') + skip)

        skip = conv(h, name + '_c5', nf=64, rf=3, stride=2, init_scale=np.sqrt(2))
        h = activ(skip)

        h = activ(conv(h, name + '_c6', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), pad='SAME'))
        h = activ(conv(h, name + '_c7', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), pad='SAME') + skip)

        h = activ(conv(h, name + '_c8', nf=64, rf=3, stride=2, init_scale=np.sqrt(2)))
        h = conv_to_fc(h)

        return activ(fc(h, 'fc1', nh=110, init_scale=np.sqrt(2)))


def get_flow_vec(unscaled_images, name, nbatch, add_flownet,
                 unscaled_previous_images, flownet,
                 train_from_scratch,
                 large_cnn, diff_flow):
    assert add_flownet
    assert unscaled_previous_images is not None
    assert flownet is not None

    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.

    scaled_previous_images = tf.cast(unscaled_previous_images, tf.float32) / 255.

    img_stack = tf.concat([scaled_previous_images, scaled_images], axis=3)

    flow, _ = flownet(img_stack, trainable=train_from_scratch, size=nbatch)

    shape = flow.get_shape().as_list()
    flow = tf.transpose(flow, [0, 3, 1, 2])
    flow = tf.reshape(flow, [shape[0] * shape[-1], 84 * 84])

    n = 6

    _, max_flow_ids = tf.nn.top_k(tf.abs(flow), n)

    r = tf.transpose(tf.tile(tf.reshape(tf.range(flow.get_shape().as_list()[0]), [1, -1]), (n, 1)))

    ids = tf.stack([r, max_flow_ids], -1)

    max_5_flow = tf.gather_nd(flow, ids)

    max_flow = tf.reduce_mean(max_5_flow, -1)

    return tf.stop_gradient(tf.reshape(max_flow, [shape[0], shape[-1]]))





class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class LstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy(object):
    # pylint: disable=W0613
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            pi = fc(h, 'pi', nact, init_scale=0.01)
            vf = fc(h, 'v', 1)[:, 0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class MlpPolicy(object):
    # pylint: disable=W0613
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:, 0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                                     initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class MImVecPolicy(object):
    # pylint: disable=W0613
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, add_flownet,
                 reuse=False,
                 flownet=None, train_from_scratch=False,
                 recurrent=None,
                 large_cnn=False, nlstm=64, add_predicted_flow_to_vec=False, diff_frames=False):
        ob_shape_vec = (nbatch,) + ob_space["vector"].shape
        nh, nw, nc = ob_space["image"].shape
        ob_shape_im = (nbatch, nh, nw, nc)

        actdim = ac_space.shape[0]
        X_vec = tf.placeholder(tf.float32, ob_shape_vec, name='Ob_vec')  # obs
        X_im = tf.placeholder(tf.uint8, ob_shape_im, name='Ob_im')

        if add_flownet:
            # adding previous image placeholder:
            X_p = tf.placeholder(tf.uint8, ob_shape_im, name='Ob_p')  # obs t-1
        else:
            X_p = None

        if recurrent:
            nenv = nbatch // nsteps
            M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
            S = tf.placeholder(tf.float32, [nenv, nlstm*2])  # states

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h_im = mujoco_cnn(
                X_im, 'pi', nbatch, add_flownet and not add_predicted_flow_to_vec,
                X_p, flownet,
                train_from_scratch,
                large_cnn, diff_frames)

            if add_predicted_flow_to_vec:
                flow_vec = get_flow_vec(
                    X_im, 'pi', nbatch, add_flownet,
                    X_p, flownet,
                    train_from_scratch,
                    large_cnn, diff_frames)
                h_vec = tf.concat([X_vec, flow_vec], axis=-1)
                h_vec = activ(fc(h_vec, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            else:
                h_vec = activ(fc(X_vec, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h1 = tf.concat([h_im, h_vec], 1)

            if recurrent:
                xs = batch_to_seq(h1, nenv, nsteps)
                ms = batch_to_seq(M, nenv, nsteps)
                if recurrent == 'lstm':
                    h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
                else:
                    assert recurrent == 'lnlstm'
                    h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
                h2 = seq_to_batch(h5)
            else:
                h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)

            vf = fc(h2, 'vf', 1)
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                                     initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        a0_r = self.pd.mode()
        neglogp0 = self.pd.neglogp(a0)
        if not recurrent:
            self.initial_state = None
        else:
            self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)


        self.placeholder_dict = {
            "image": X_im,
            "vector": X_vec
        }
        if add_flownet:
            self.placeholder_dict["last_image"] = X_p

        if not recurrent:
            def step(ob, *_args, remove_noise=False, **_kwargs):
                feed_dict = {}
                for key, value in self.placeholder_dict.items():
                    feed_dict[value] = ob[key]
                if not remove_noise:
                    a, v, neglogp = sess.run([a0, v0, neglogp0], feed_dict=feed_dict)
                else:
                    a, v, neglogp = sess.run([a0_r, v0, neglogp0], feed_dict=feed_dict)
                return a, v, self.initial_state, neglogp

            def value(ob, *_args, **_kwargs):
                feed_dict = {}
                for key, value in self.placeholder_dict.items():
                    feed_dict[value] = ob[key]
                return sess.run(v0, feed_dict=feed_dict)
        else:
            def step(ob, state, mask, remove_noise=False):
                feed_dict = {}
                for key, value in self.placeholder_dict.items():
                    feed_dict[value] = ob[key]
                feed_dict[S] = state
                feed_dict[M] = mask
                if not remove_noise:
                    a, v, s, neglogp = sess.run([a0, v0, snew, neglogp0], feed_dict=feed_dict)
                else:
                    a, v, s, neglogp = sess.run([a0_r, v0, snew, neglogp0], feed_dict=feed_dict)
                return a, v, s, neglogp

            def value(ob, state, mask):
                feed_dict = {}
                for key, value in self.placeholder_dict.items():
                    feed_dict[value] = ob[key]
                feed_dict[S] = state
                feed_dict[M] = mask
                return sess.run(v0, feed_dict=feed_dict)

        self.X_im = X_im
        self.X_vec = X_vec
        self.X_p = X_p
        self.pi = pi
        if not recurrent:
            self.vf = v0
        else:
            self.vf = vf
            self.M = M
            self.S = S
        self.step = step
        self.value = value


class MImVecLstmPolicy(MImVecPolicy):
    def __init__(self, *args, **kwargs):
        super(MImVecLstmPolicy, self).__init__(*args, **kwargs, recurrent='lstm')


class MImVecLnLstmPolicy(MImVecPolicy):
    def __init__(self, *args, **kwargs):
        super(MImVecLnLstmPolicy, self).__init__(*args, **kwargs, recurrent='lnlstm')
