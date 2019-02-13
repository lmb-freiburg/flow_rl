import os
import os.path as osp
import time
from collections import deque
from os.path import join as p_join

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from baselines import logger
from baselines.common import explained_variance
from baselines.flow_rl_utils import network
from baselines.ppo2.policies import MImVecPolicy, MImVecLstmPolicy, MImVecLnLstmPolicy


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm, add_flownet,
                 flownet_path, flownet,
                 train_from_scratch, large_cnn,
                 add_predicted_flow_to_vec, diff_frames):
        if policy not in (MImVecPolicy, MImVecLstmPolicy, MImVecLnLstmPolicy):
            raise NotImplementedError

        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False,
                           add_flownet=add_flownet,
                           flownet=flownet,
                           train_from_scratch=train_from_scratch,
                           large_cnn=large_cnn,
                           add_predicted_flow_to_vec=add_predicted_flow_to_vec,
                           diff_frames=diff_frames)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True,
                             add_flownet=add_flownet,
                             flownet=flownet,
                             train_from_scratch=train_from_scratch,
                             large_cnn=large_cnn,
                             add_predicted_flow_to_vec=add_predicted_flow_to_vec,
                             diff_frames=diff_frames)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(
            train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            if isinstance(obs, dict):
                td_map = {A: actions, ADV: advs, R: returns, LR: lr,
                          CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values}

                for key, value in train_model.placeholder_dict.items():
                    td_map[value] = obs[key]
            else:
                td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                          CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        self.saver = tf.train.Saver(max_to_keep=20)

        def save(save_path, id):
            self.saver.save(sess, p_join(save_path, "model{}.ckpt".format(id)))

        def load(load_path, id):
            self.saver.restore(sess, p_join(load_path, "model{}.ckpt".format(id)))

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)  # pylint: disable=E1101
        if add_flownet:
            if not train_from_scratch:
                assert flownet_path != ""
                checkpoint_path = p_join(flownet_path, "flow.ckpt")
                reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
                var_to_shape_map = reader.get_variable_to_shape_map()
                l = [var for var in var_to_shape_map
                     if 'Adam' not in var and 'step' not in var and 'beta' not in var]
                lp = [[var, reader.get_tensor(var)[1].shape] for var in var_to_shape_map
                      if 'Adam' not in var and 'step' not in var and 'beta' not in var]

                d = {var: tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + var)[0]
                     for var in l}

                flow_saver = tf.train.Saver(d)

                flow_saver.restore(sess, checkpoint_path)

            if train_from_scratch:
                tf.global_variables_initializer().run(session=sess)


class ImVecRunner(object):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        self.nenv = env.num_envs
        self.obs = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(self.nenv)]

        self.highest_reward = - float('inf')

    def run(self, update_fraction):
        mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], []
        mb_obs = {}
        for key in self.obs:
            mb_obs[key] = []
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states,
                                                                       self.dones)
            for key, value in self.obs.items():
                mb_obs[key].append(value.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            self.obs, rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        for key, value in mb_obs.items():
            mb_obs[key] = np.asarray(value, dtype=value[0].dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        # mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(dsf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)


def dsf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if isinstance(arr, dict):
        for key, value in arr.items():
            s = value.shape
            arr[key] = value.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
        return arr
    else:
        s = arr.shape
        return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
          vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=0, add_flownet=False, flownet_path="",
          flow_key=None,
          train_from_scratch=False,
          large_cnn=False, add_predicted_flow_to_vec=False,
          diff_frames=False):

    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    if flow_key is not None:
        flownet = network.flow_dict[flow_key]
    else:
        flownet = None

    def make_model():
        return Model(
            policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
            nbatch_train=nbatch_train, nsteps=nsteps, ent_coef=ent_coef,
            vf_coef=vf_coef, max_grad_norm=max_grad_norm, add_flownet=add_flownet,
            flownet_path=flownet_path,
            flownet=flownet,
            train_from_scratch=train_from_scratch,
            large_cnn=large_cnn, add_predicted_flow_to_vec=add_predicted_flow_to_vec,
            diff_frames=diff_frames
        )

    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if isinstance(ob_space, dict):
        runner = ImVecRunner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    else:
        raise NotImplementedError

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    model_save_interval = nupdates // (total_timesteps // int(1e6))

    saved_model_id = 0

    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        # pylint: disable=E0632:
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(
            np.float(update) / np.float(nupdates))
        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None:  # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]

                    slices = (get_part(arr, mbinds) for arr in (obs, returns, masks, actions,
                                                                values, neglogpacs))

                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else:  # recurrent version
            assert nenvs % nminibatches == 0
            # envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (get_part(arr, mbflatinds) for arr in (obs, returns, masks, actions,
                                                                    values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        if (update - 1) % model_save_interval == 0:
            if isinstance(ob_space, dict):
                save_path = p_join(logger.get_dir(), "saves")
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                model.save(save_path, saved_model_id)
                env.save_norm(save_path, saved_model_id)
                saved_model_id += 1

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if isinstance(ob_space, dict):
                logger.logkv("highest_reward", runner.highest_reward)
            logger.dumpkvs()

    env.close()


def get_part(l, mb):
    if isinstance(l, dict):
        out = {}
        for key, value in l.items():
            out[key] = value[mb]
        return out
    else:
        return l[mb]


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
