
import os
import numpy as np
import tensorflow as tf
from copy import deepcopy
import math

tf.config.run_functions_eagerly(True)

def dense(x, units, name=None, activation=None):
    return tf.keras.layers.Dense(units, activation=activation, kernel_initializer='he_normal', name=name)(x)

def build_positional_encoding(length, dim):
    pos = np.arange(length)[:, None]
    i = np.arange(dim)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dim))
    angle_rads = pos * angle_rates
    pe = np.zeros((length, dim))
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(pe, dtype=tf.float32)

class GraphAggregation(tf.keras.layers.Layer):
    def __init__(self, neighbors_list, out_dim, name=None):
        super().__init__(name=name)
        self.neighbors_list = neighbors_list
        self.out_dim = out_dim
        self.linear = tf.keras.layers.Dense(out_dim, activation=None, kernel_initializer='he_normal')

        N = len(neighbors_list)
        A = np.zeros((N, N), dtype=np.float32)
        for i, neigh in enumerate(neighbors_list):
            for j in neigh:
                A[i, j] = 1.0

        for i in range(N):
            A[i, i] = 1.0

        row_sum = A.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        A = A / row_sum
        self.A = tf.constant(A, dtype=tf.float32)

    def call(self, node_feats):

        return self.linear(tf.matmul(self.A, node_feats))

def masked_softmax(logits, mask, axis=-1, eps=1e-8):

    big_neg = -1e9
    logits_masked = tf.where(mask > 0, logits, big_neg * tf.ones_like(logits))
    probs = tf.nn.softmax(logits_masked, axis=axis)
    probs = probs * mask
    denom = tf.reduce_sum(probs, axis=axis, keepdims=True) + eps
    probs = probs / denom
    return probs

def reward_shaping(node_rewards, demand_index, shaping_coef=0.8, orr_boost=2.0):

    shaped = node_rewards + shaping_coef * demand_index

    if hasattr(reward_shaping, 'current_orr') and reward_shaping.current_orr > 0.85:
        shaped = shaped * (1.0 + orr_boost * (reward_shaping.current_orr - 0.85))

    return shaped

reward_shaping.current_orr = 0.0

class Estimator:
    def __init__(self,
                 action_dim,
                 state_dim,
                 env,
                 predict_time,
                 scope="improved_estimator",
                 summaries_dir=None,
                 hidden_dim=128,
                 dropout_rate=0.1,
                 use_mixed_precision=False):
        self.scope = scope
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.env = env
        self.predict_time = predict_time
        self.n_nodes = env.n_valid_grids
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        if use_mixed_precision:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            self.use_mixed_precision = True
        else:
            self.use_mixed_precision = False

        self.neighbors_list = []
        for idx, node_id in enumerate(env.target_grids):
            neighbor_indices = env.nodes[node_id].layers_neighbors_id[0]
            neighbor_ids = [env.target_grids.index(env.nodes[item].get_node_index()) for item in neighbor_indices]
            neighbor_ids.append(idx)
            self.neighbors_list.append(neighbor_ids)

        self.gcn = GraphAggregation(self.neighbors_list, out_dim=self.hidden_dim)

        self.pos_enc = build_positional_encoding(self.predict_time + 16, self.hidden_dim)

        self._build_models()

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        print(f"Policy network weights check:")
        for i, layer in enumerate(self.policy_model.layers):
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                weights = layer.kernel.numpy()
                print(f"  Layer {i} ({layer.name}): shape={weights.shape}, range=[{weights.min():.6f}, {weights.max():.6f}], mean={weights.mean():.6f}")
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias = layer.bias.numpy()
                print(f"  Layer {i} bias: shape={bias.shape}, range=[{bias.min():.6f}, {bias.max():.6f}], mean={bias.mean():.6f}")

        self.ckpt = tf.train.Checkpoint(policy=self.policy_model, value=self.value_model,
                                        policy_opt=self.policy_optimizer, value_opt=self.value_optimizer)
        self.manager = tf.train.CheckpointManager(self.ckpt, directory='./improved_checkpoints', max_to_keep=5)

    def _node_feature_from_state(self, s):

        N = self.n_nodes
        batch = tf.shape(s)[0]
        max_feature_dim = 100

        if len(s.shape) == 3 and s.shape[1] == N:

            if s.shape[2] > max_feature_dim:
                return s[:, :, :max_feature_dim]
            elif s.shape[2] < max_feature_dim:
                return tf.pad(s, [[0, 0], [0, 0], [0, max_feature_dim - s.shape[2]]])
            else:
                return s

        feature_dim_per_node = min(s.shape[-1] // N, max_feature_dim) if s.shape[-1] >= N else 1

        if s.shape[-1] >= N * feature_dim_per_node:

            node_feats = tf.reshape(s[:, :N * feature_dim_per_node], (batch, N, feature_dim_per_node))
        else:

            padded_s = tf.pad(s, [[0, 0], [0, N * feature_dim_per_node - s.shape[-1]]])
            node_feats = tf.reshape(padded_s, (batch, N, feature_dim_per_node))

        if node_feats.shape[2] < max_feature_dim:
            node_feats = tf.pad(node_feats, [[0, 0], [0, 0], [0, max_feature_dim - node_feats.shape[2]]])
        elif node_feats.shape[2] > max_feature_dim:
            node_feats = node_feats[:, :, :max_feature_dim]

        return node_feats

    def _build_encoder(self):

        max_feature_dim = 100
        node_input = tf.keras.Input(shape=(None, max_feature_dim), name="node_input")

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.hidden_dim, activation='relu')
        )(node_input)

        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.hidden_dim, activation='relu')
        )(x)

        return tf.keras.Model(inputs=node_input, outputs=x, name="encoder")

    def _build_models(self):

        self.encoder = self._build_encoder()
        max_feature_dim = 100

        node_input = tf.keras.Input(shape=(None, max_feature_dim), name='state_node_input')
        node_repr = self.encoder(node_input)
        global_repr = tf.reduce_mean(node_repr, axis=1)

        per_node = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(node_repr)
        per_node = tf.keras.layers.Dense(1, activation=None)(per_node)
        per_node = tf.reshape(per_node, (-1, tf.shape(node_repr)[1]))

        self.value_model = tf.keras.Model(inputs=node_input, outputs=per_node, name='value_model')

        node_input_p = tf.keras.Input(shape=(None, max_feature_dim), name='policy_node_input')
        node_repr_p = self.encoder(node_input_p)
        global_repr_p = tf.reduce_mean(node_repr_p, axis=1)

        policy_hidden = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(node_repr_p)
        logits = tf.keras.layers.Dense(self.action_dim, activation=None)(policy_hidden)

        self.policy_model = tf.keras.Model(inputs=node_input_p, outputs=logits, name='policy_model')

    def predict(self, s, v_feature=None, order_predict=None):
        node_feats = self._node_feature_from_state(s)
        return self.value_model(node_feats)

    def action(self, s, context, order_predict, next_v, epsilon):

        debug_enabled = os.environ.get('DEBUG_ESTIMATOR_ACTION', 'False').lower() == 'true'

        single_input = False
        if s.ndim == 1:
            s = np.expand_dims(s, 0)
            single_input = True

        s_tf = tf.convert_to_tensor(s, dtype=tf.float32)
        node_feats = self._node_feature_from_state(s_tf)

        try:

            logits = self.policy_model(node_feats).numpy()

        except Exception as e:

            raise e

        batch = logits.shape[0]
        probs = np.zeros_like(logits)

        if hasattr(self.env, 'valid_action_mask'):
            valid_mask = self.env.valid_action_mask
        else:

            valid_mask = np.ones((self.n_nodes, self.action_dim), dtype=np.float32)

        for b in range(batch):
            for n in range(self.n_nodes):
                logit = logits[b, n]
                mask = valid_mask[n]

                big_neg = -1e9
                logit_masked = np.where(mask > 0, logit, big_neg)
                p = np.exp(logit_masked - np.max(logit_masked))
                p = p * mask
                ssum = np.sum(p)
                if ssum <= 0:
                    p = mask / (np.sum(mask) + 1e-8)
                else:
                    p = p / ssum
                probs[b, n] = p

        action_tuple = []
        valid_prob_list = []
        policy_state = []
        action_chosen_mat = []
        curr_state_values = []
        next_state_ids = []
        policy_order_predict = []
        policy_next_v = []

        if np.isscalar(context) or (hasattr(context, 'shape') and (len(context.shape) == 0 or context.shape[0] == 1)):
            context_arr = np.ones(self.n_nodes, dtype=np.int32) * int(context)
        else:
            context_arr = np.array(context).reshape(-1)
            if context_arr.shape[0] != self.n_nodes:
                context_arr = np.ones(self.n_nodes, dtype=np.int32)

        action_count = 0
        for n in range(self.n_nodes):
            p = probs[0, n]
            valid_prob_list.append(p)
            if context_arr[n] == 0:
                continue

            if np.random.rand() < epsilon:
                a = np.random.choice(self.action_dim, p=p)

            else:
                a = np.random.choice(self.action_dim, p=p)

            num = int(context_arr[n])
            start_node_id = self.env.target_grids[n]

            try:
                if a < len(self.env.nodes[self.env.target_grids[n]].neighbors) and self.env.nodes[self.env.target_grids[n]].neighbors[a] is not None:
                    neighbor_node = self.env.nodes[self.env.target_grids[n]].neighbors[a]
                    neighbor_node_id = neighbor_node.get_node_index()
                    if neighbor_node_id in self.env.target_grids:
                        end_node_id = neighbor_node_id
                    else:
                        end_node_id = start_node_id
                else:
                    end_node_id = start_node_id
            except (IndexError, AttributeError):
                end_node_id = start_node_id

            if end_node_id != start_node_id:
                action_tuple.append((start_node_id, end_node_id, num))
                action_count += 1

            else:

                pass

            one_hot = np.zeros(self.action_dim, dtype=np.float32)
            one_hot[a] = 1.0
            action_chosen_mat.append(one_hot)
            policy_state.append(s[0])
            curr_state_values.append(self.predict(s_tf).numpy().flatten()[n])
            next_state_ids.append(int(a))
            policy_order_predict.append(order_predict[n] if (order_predict is not None and hasattr(order_predict, '__len__')) else np.zeros((self.predict_time,)))
            policy_next_v.append(next_v[n] if (next_v is not None and hasattr(next_v, '__len__')) else np.zeros((self.n_nodes,)))

        valid_prob_mat = np.stack(valid_prob_list) if len(valid_prob_list) > 0 else np.zeros((self.n_nodes, self.action_dim))

        curr_neighbor_mask = []
        for n in range(self.n_nodes):
            if n < len(action_chosen_mat):
                curr_neighbor_mask.append(valid_mask[n])
            else:
                curr_neighbor_mask.append(np.zeros(self.action_dim))
        curr_neighbor_mask = np.array(curr_neighbor_mask)

        if single_input:

            return action_tuple, valid_prob_mat, np.array(policy_state), np.array(action_chosen_mat), np.array(curr_state_values), \
                   curr_neighbor_mask, np.array(next_state_ids), np.array(policy_order_predict), np.array([]), np.array(policy_next_v)
        else:
            return action_tuple, valid_prob_mat, np.array(policy_state), np.array(action_chosen_mat), np.array(curr_state_values), \
                   curr_neighbor_mask, np.array(next_state_ids), np.array(policy_order_predict), np.array([]), np.array(policy_next_v)

    @tf.function
    def train_value_step(self, node_state_batch, targets):
        with tf.GradientTape() as tape:
            preds = self.value_model(node_state_batch, training=True)
            loss = tf.reduce_mean(tf.keras.losses.MSE(targets, preds))
        grads = tape.gradient(loss, self.value_model.trainable_variables)

        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)

        self.value_optimizer.apply_gradients(zip(grads, self.value_model.trainable_variables))
        return loss

    @tf.function
    def train_policy_step(self, node_state_batch, actions_onehot, advantages, valid_mask):
        with tf.GradientTape() as tape:

            logits = self.policy_model(node_state_batch, training=True)

            logits_shape = tf.shape(logits)
            actions_shape = tf.shape(actions_onehot)

            if len(actions_onehot.shape) == 2:
                batch_size = logits_shape[0]
                n_nodes = logits_shape[1]
                action_dim = logits_shape[2]

                if actions_shape[0] == batch_size and actions_shape[1] == action_dim:

                    actions_onehot = tf.expand_dims(actions_onehot, axis=1)
                    actions_onehot = tf.tile(actions_onehot, [1, n_nodes, 1])
                else:

                    total_elements = actions_shape[0] * actions_shape[1]
                    expected_elements = batch_size * n_nodes * action_dim

                    if total_elements == expected_elements:

                        actions_onehot = tf.reshape(actions_onehot, [batch_size, n_nodes, action_dim])
                    else:

                        if actions_shape[0] == batch_size and actions_shape[1] == action_dim:
                            actions_onehot = tf.expand_dims(actions_onehot, axis=1)
                            actions_onehot = tf.tile(actions_onehot, [1, n_nodes, 1])
                        else:

                            actions_onehot = tf.zeros([batch_size, n_nodes, action_dim])

            if logits.shape[-1] != actions_onehot.shape[-1]:
                target_action_dim = logits.shape[-1]
                current_action_dim = actions_onehot.shape[-1]

                if current_action_dim < target_action_dim:

                    padding = tf.zeros([tf.shape(actions_onehot)[0], tf.shape(actions_onehot)[1], 
                                      target_action_dim - current_action_dim])
                    actions_onehot = tf.concat([actions_onehot, padding], axis=-1)
                else:

                    actions_onehot = actions_onehot[:, :, :target_action_dim]

            logits_2d = tf.reshape(logits, [-1, tf.shape(logits)[-1]])
            actions_2d = tf.reshape(actions_onehot, [-1, tf.shape(actions_onehot)[-1]])

            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=actions_2d, 
                logits=logits_2d
            )

            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.policy_model.trainable_variables)

        if grads is not None:
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
            self.policy_optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables))

        return loss

    def compute_advantage(self, curr_state_value, next_state_ids, next_state, node_reward, next_order_predict, next_v,
                          p_kl, gamma):
        try:

            ns_tf = tf.convert_to_tensor(next_state, dtype=tf.float32)
            q_next = self.value_model(self._node_feature_from_state(ns_tf)).numpy().flatten()
            advantage = []

            if hasattr(p_kl, 'numpy'):
                p_kl = p_kl.numpy()
            p_kl = np.array(p_kl).flatten()

            node_reward = np.array(node_reward).flatten()

            for idx, nid in enumerate(next_state_ids):
                nid = int(nid)
                if nid < len(q_next) and nid < len(node_reward) and nid < len(p_kl):
                    temp = - p_kl[nid] + node_reward[nid] + gamma * q_next[nid] - curr_state_value[idx]
                else:
                    temp = 0.0
                advantage.append(temp)
            return np.array(advantage)
        except Exception as e:
            print(f"Error in compute_advantage: {e}")
            return np.zeros(len(curr_state_value))

    def compute_targets(self, valid_prob, next_state, node_reward, next_order_predict, o_p, o, next_v, gamma):
        try:
            ns_tf = tf.convert_to_tensor(next_state, dtype=tf.float32)
            q_next = self.value_model(self._node_feature_from_state(ns_tf)).numpy().flatten()

            node_reward = np.array(node_reward).flatten()
            o = np.array(o).flatten()
            o_p = np.array(o_p).flatten() if hasattr(o_p, 'flatten') else np.zeros(self.n_nodes)

            if len(o) > self.n_nodes:
                o_real = o[self.n_nodes:]
            else:
                o_real = o[:self.n_nodes]

            min_len = min(len(o_p), len(o_real))
            o_predict = o_p[:min_len]
            o_real = o_real[:min_len]

            kl = np.abs(o_predict - o_real)
            if len(kl) < self.n_nodes:
                kl = np.pad(kl, (0, self.n_nodes - len(kl)), 'constant')

            targets = []
            p_kl = kl

            for idx in range(self.n_nodes):
                neighbor_ids = self.neighbors_list[idx]

                if hasattr(self.env, 'valid_action_mask') and idx < len(self.env.valid_action_mask):
                    valid_mask = self.env.valid_action_mask[idx] > 0
                    grid_prob = valid_prob[idx][valid_mask]
                else:
                    grid_prob = valid_prob[idx]

                if len(grid_prob) > 0 and np.sum(grid_prob) > 0:
                    grid_prob = grid_prob / np.sum(grid_prob)

                neigh_rewards = node_reward[neighbor_ids] if len(neighbor_ids) <= len(node_reward) else node_reward[:len(neighbor_ids)]
                neigh_q = q_next[neighbor_ids] if len(neighbor_ids) <= len(q_next) else q_next[:len(neighbor_ids)]
                neigh_kl = kl[neighbor_ids] if len(neighbor_ids) <= len(kl) else kl[:len(neighbor_ids)]

                min_len = min(len(grid_prob), len(neigh_rewards), len(neigh_q), len(neigh_kl))
                if min_len > 0:
                    curr_grid_target = np.sum(grid_prob[:min_len] * (neigh_rewards[:min_len] - neigh_kl[:min_len] + gamma * neigh_q[:min_len]))
                else:
                    curr_grid_target = 0.0
                targets.append(curr_grid_target)

            return np.array(targets).reshape([-1, 1]), p_kl
        except Exception as e:
            print(f"Error in compute_targets: {e}")
            return np.zeros((self.n_nodes, 1)), np.zeros(self.n_nodes)

    def behavior_cloning_pretrain(self, states, actions_onehot, epochs=5, batch_size=64):
        num_samples = states.shape[0]
        idx = np.arange(num_samples)
        for ep in range(epochs):
            np.random.shuffle(idx)
            for i in range(0, num_samples, batch_size):
                batch_idx = idx[i:i+batch_size]
                s_batch = states[batch_idx]
                a_batch = actions_onehot[batch_idx]
                node_feats = self._node_feature_from_state(tf.convert_to_tensor(s_batch, dtype=tf.float32))

                with tf.GradientTape() as tape:
                    logits = self.policy_model(node_feats, training=True)
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=a_batch, logits=logits))
                grads = tape.gradient(loss, self.policy_model.trainable_variables)
                self.policy_optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables))
            print(f"BC epoch {ep+1}/{epochs} loss={loss.numpy():.6f}")

    def save_weights(self):
        self.manager.save()
        print("Saved checkpoint.")

    def load_weights(self, latest=True):
        if latest:
            self.ckpt.restore(self.manager.latest_checkpoint)
            print("Restored latest checkpoint:", self.manager.latest_checkpoint)
