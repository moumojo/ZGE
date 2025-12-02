
import numpy as np
import tensorflow as tf
import random, os
from algorithm.alg_utility import *
from copy import deepcopy
import scipy.stats

tf.config.run_functions_eagerly(True)

class Estimator:
    def __init__(self,
                 action_dim,
                 state_dim,
                 env,
                 predict_time,
                 scope="estimator",
                 summaries_dir=None):
        self.n_valid_grid = env.n_valid_grids
        self.action_dim = action_dim
        self.enhanced_state_dim = 4 * env.n_valid_grids + 144 + 4
        self.state_dim = state_dim
        self.M = env.M
        self.N = env.N
        self.scope = scope
        self.T = 144
        self.env = env
        self.predict_time = predict_time

        self.summary_writer = None

        value_loss = self._build_value_model()

        actor_loss, entropy = self._build_mlp_policy()

        self.loss = actor_loss + 0.5 * value_loss - 10 * entropy

        if summaries_dir:
            summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.summary_writer = tf.summary.create_file_writer(summary_dir)

        self.neighbors_list = []
        for idx, node_id in enumerate(env.target_grids):
            neighbor_indices = env.nodes[node_id].layers_neighbors_id[0]
            neighbor_ids = [env.target_grids.index(env.nodes[item].get_node_index()) for item in neighbor_indices]
            neighbor_ids.append(idx)

            self.neighbors_list.append(neighbor_ids)

    def _graph_convolution(self, node_features, neighbors_list):
        batch_size = tf.shape(node_features)[0]
        n_nodes = tf.shape(node_features)[1]
        feature_dim = tf.shape(node_features)[2]

        aggregated_features = []

        for i in range(n_nodes):

            neighbor_indices = neighbors_list[i]

            neighbor_features = tf.gather(node_features, neighbor_indices, axis=1)

            aggregated = tf.reduce_mean(neighbor_features, axis=1)
            aggregated_features.append(aggregated)

        aggregated_features = tf.stack(aggregated_features, axis=1)

        aggregated_features = tf.keras.layers.Dense(feature_dim, activation='relu')(aggregated_features)

        return aggregated_features

        self.valid_action_mask = np.ones((self.n_valid_grid, self.action_dim))
        self.valid_neighbor_node_id = np.zeros((self.n_valid_grid, self.action_dim))
        self.valid_neighbor_grid_id = np.zeros((self.n_valid_grid, self.action_dim))
        for grid_idx, grid_id in enumerate(env.target_grids):
            for neighbor_idx, neighbor in enumerate(self.env.nodes[grid_id].neighbors):
                if neighbor is None:
                    self.valid_action_mask[grid_idx, neighbor_idx] = 0
                else:
                    node_index = neighbor.get_node_index()
                    self.valid_neighbor_node_id[grid_idx, neighbor_idx] = node_index
                    self.valid_neighbor_grid_id[grid_idx, neighbor_idx] = env.target_grids.index(node_index)

            self.valid_neighbor_node_id[grid_idx, -1] = grid_id
            self.valid_neighbor_grid_id[grid_idx, -1] = grid_idx

    def _build_value_model(self):

        state_input = tf.keras.Input(shape=(self.enhanced_state_dim,), name="state")
        v_feature_input = tf.keras.Input(shape=(self.env.n_valid_grids*2,), name="v_feature")
        order_predict_input = tf.keras.Input(shape=(self.env.n_valid_grids + (144 + self.predict_time - 1) + 
                                                   self.env.n_valid_grids, self.predict_time), name="order_predict")
        y_target = tf.keras.Input(shape=(1,), name="y_target")

        predict = all_c_se_block(order_predict_input, self.predict_time)

        cold_hot_zone = self.compute_cold_hot_zone(v_feature_input, predict)

        driver_density = state_input[:, :self.env.n_valid_grids]
        order_density = state_input[:, self.env.n_valid_grids:2*self.env.n_valid_grids]
        supply_demand_ratio = state_input[:, 2*self.env.n_valid_grids:3*self.env.n_valid_grids]
        balance_state = state_input[:, 3*self.env.n_valid_grids:4*self.env.n_valid_grids]

        inputs = tf.keras.layers.concatenate(
            [driver_density, order_density, supply_demand_ratio, balance_state, predict], axis=1)

        l1 = tf.keras.layers.Dense(512, activation='relu', 
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005))(inputs)
        l1_dropout = tf.keras.layers.Dropout(0.2)(l1)

        l2 = tf.keras.layers.Dense(384, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005))(l1_dropout)
        l2_dropout = tf.keras.layers.Dropout(0.2)(l2)

        l3 = tf.keras.layers.Dense(256, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005))(l2_dropout)
        l3_dropout = tf.keras.layers.Dropout(0.15)(l3)

        l4 = tf.keras.layers.Dense(128, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005))(l3_dropout)
        l4_dropout = tf.keras.layers.Dropout(0.1)(l4)

        l5 = tf.keras.layers.Dense(64, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005))(l4_dropout)

        if l3_dropout.shape[-1] == l5.shape[-1]:
            l5 = l5 + l3_dropout
        else:

            l3_proj = tf.keras.layers.Dense(64, activation=None)(l3_dropout)
            l5 = l5 + l3_proj

        l5_dropout = tf.keras.layers.Dropout(0.1)(l5)

        attention_weights = tf.keras.layers.Dense(64, activation='sigmoid')(l5_dropout)
        l5_attended = l5_dropout * attention_weights

        batch_size = tf.shape(l5_dropout)[0]
        node_features = tf.reshape(l5_dropout, [batch_size, self.n_valid_grid, -1])

        neighbor_aggregated = self._graph_convolution(node_features, self.neighbors_list)

        global_graph_info = tf.reduce_mean(neighbor_aggregated, axis=1)
        global_graph_info = tf.tile(tf.expand_dims(global_graph_info, 1), [1, self.n_valid_grid, 1])

        enhanced_features = neighbor_aggregated + 0.2 * global_graph_info
        enhanced_features = tf.reshape(enhanced_features, [batch_size, -1])

        multi_scale_features = tf.concat([
            l1_dropout,
            l3_dropout,
            l5_dropout,
            enhanced_features
        ], axis=1)

        fusion_layer = tf.keras.layers.Dense(128, activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.0005))(multi_scale_features)
        fusion_layer = tf.keras.layers.Dropout(0.1)(fusion_layer)

        l5_final = l5_dropout + 0.3 * l5_attended + 0.2 * enhanced_features + 0.1 * fusion_layer

        grade = fc(l5_final, "value_output", 1, act=tf.nn.relu)

        query = tf.nn.sigmoid(cold_hot_zone)
        key = tf.nn.sigmoid(cold_hot_zone)
        value = tf.nn.sigmoid(cold_hot_zone)

        value_output = hyper_multihead_attention(query, key, value, grade, ax=1)

        if query.shape == value_output.shape:
            value_output = value_output + 0.1 * query

        mse_loss = tf.reduce_mean(tf.square(y_target - value_output))

        value_output_sigmoid = tf.nn.sigmoid(value_output)
        orr_loss = tf.reduce_mean(tf.square(y_target - value_output_sigmoid))

        supply_demand_ratio = inputs[:, 2*self.env.n_valid_grids:3*self.env.n_valid_grids] 
        ideal_ratio = 1.0
        balance_loss = tf.reduce_mean(tf.square(supply_demand_ratio - ideal_ratio))

        order_density = inputs[:, self.env.n_valid_grids:2*self.env.n_valid_grids] 
        orr_potential = order_density / (supply_demand_ratio + 1e-6)
        orr_potential_loss = -tf.reduce_mean(orr_potential) 

        completion_rate = tf.minimum(supply_demand_ratio, 1.0)  
        completion_loss = -tf.reduce_mean(completion_rate)  

        value_loss = mse_loss + 0.5 * orr_loss + 0.2 * balance_loss + 0.3 * orr_potential_loss + 0.2 * completion_loss

        self.value_model = tf.keras.Model(
            inputs=[state_input, v_feature_input, order_predict_input, y_target],
            outputs=[value_output, value_loss]
        )

        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.state_input = state_input
        self.v_feature_input = v_feature_input
        self.order_predict_input = order_predict_input
        self.y_target = y_target
        self.value_output = value_output
        self.value_loss = value_loss

        return value_loss

    def _build_mlp_policy(self):
        policy_state_input = tf.keras.Input(shape=(self.enhanced_state_dim,), name="policy_state")
        policy_v_feature_input = tf.keras.Input(shape=(self.env.n_valid_grids*2,), name="policy_v_feature")
        action_input = tf.keras.Input(shape=(self.action_dim,), name="action")
        advantage_input = tf.keras.Input(shape=(1,), name="advantage")
        neighbor_mask_input = tf.keras.Input(shape=(self.action_dim,), name="neighbor_mask")
        policy_order_predict_input = tf.keras.Input(shape=(self.env.n_valid_grids + (144 + self.predict_time - 1) +
                                                   self.env.n_valid_grids, self.predict_time), name="policy_order_predict")

        predict = all_c_se_block(policy_order_predict_input, self.predict_time)
        cold_hot_zone = self.compute_cold_hot_zone(policy_v_feature_input, predict)

        driver_density = policy_state_input[:, :self.env.n_valid_grids]
        order_density = policy_state_input[:, self.env.n_valid_grids:2*self.env.n_valid_grids]
        supply_demand_ratio = policy_state_input[:, 2*self.env.n_valid_grids:3*self.env.n_valid_grids]
        balance_state = policy_state_input[:, 3*self.env.n_valid_grids:4*self.env.n_valid_grids]

        inputs = tf.keras.layers.concatenate(
            [driver_density, order_density, supply_demand_ratio, balance_state, predict], axis=1)

        l1 = tf.keras.layers.Dense(512, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005))(inputs)
        l1_dropout = tf.keras.layers.Dropout(0.2)(l1)

        l2 = tf.keras.layers.Dense(384, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005))(l1_dropout)
        l2_dropout = tf.keras.layers.Dropout(0.2)(l2)

        l3 = tf.keras.layers.Dense(256, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005))(l2_dropout)
        l3_dropout = tf.keras.layers.Dropout(0.15)(l3)

        l4 = tf.keras.layers.Dense(128, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005))(l3_dropout)
        l4_dropout = tf.keras.layers.Dropout(0.1)(l4)

        l5 = tf.keras.layers.Dense(64, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005))(l4_dropout)

        if l3_dropout.shape[-1] == l5.shape[-1]:
            l5 = l5 + l3_dropout
        else:
            l3_proj = tf.keras.layers.Dense(64, activation=None)(l3_dropout)
            l5 = l5 + l3_proj

        l5_dropout = tf.keras.layers.Dropout(0.1)(l5)

        attention_weights = tf.keras.layers.Dense(64, activation='sigmoid')(l5_dropout)
        l5_attended = l5_dropout * attention_weights

        batch_size = tf.shape(l5_dropout)[0]
        node_features = tf.reshape(l5_dropout, [batch_size, self.n_valid_grid, -1])

        neighbor_aggregated = self._graph_convolution(node_features, self.neighbors_list)

        global_graph_info = tf.reduce_mean(neighbor_aggregated, axis=1)
        global_graph_info = tf.tile(tf.expand_dims(global_graph_info, 1), [1, self.n_valid_grid, 1])

        enhanced_features = neighbor_aggregated + 0.2 * global_graph_info
        enhanced_features = tf.reshape(enhanced_features, [batch_size, -1])

        multi_scale_features = tf.concat([
            l1_dropout,  
            l3_dropout,  
            l5_dropout,  
            enhanced_features
        ], axis=1)

        fusion_layer = tf.keras.layers.Dense(128, activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.0005))(multi_scale_features)
        fusion_layer = tf.keras.layers.Dropout(0.1)(fusion_layer)

        l5_final = l5_dropout + 0.3 * l5_attended + 0.2 * enhanced_features + 0.1 * fusion_layer

        grade = fc(l5_final, "logits", self.action_dim, act=tf.nn.relu)
        c = fc(cold_hot_zone, 'c_h', self.action_dim, act=tf.nn.sigmoid)

        logits = hyper_multihead_attention(c, c, c, grade, ax=2)
        valid_logits = logits * neighbor_mask_input
        softmaxprob = tf.nn.softmax(tf.math.log(valid_logits + 1e-8))
        logsoftmaxprob = tf.nn.log_softmax(softmaxprob)

        neglogprob = - logsoftmaxprob * action_input
        actor_loss = tf.reduce_mean(tf.reduce_sum(neglogprob * advantage_input, axis=1))
        entropy = - tf.reduce_mean(softmaxprob * logsoftmaxprob)
        supply_demand_ratio = policy_state_input[:, 2*self.env.n_valid_grids:3*self.env.n_valid_grids]
        order_density = policy_state_input[:, self.env.n_valid_grids:2*self.env.n_valid_grids]

        action_orr_potential = tf.reduce_sum(softmaxprob * tf.expand_dims(order_density, axis=1), axis=1)
        orr_potential_loss = -tf.reduce_mean(action_orr_potential) 

        balance_improvement = tf.reduce_sum(softmaxprob * tf.expand_dims(supply_demand_ratio, axis=1), axis=1)
        ideal_balance = 1.0
        balance_loss = tf.reduce_mean(tf.square(balance_improvement - ideal_balance))

        orr_direct_loss = -tf.reduce_mean(tf.reduce_sum(softmaxprob * tf.expand_dims(order_density / (supply_demand_ratio + 1e-6), axis=1), axis=1))

        completion_potential = tf.minimum(supply_demand_ratio, 1.0)
        completion_loss = -tf.reduce_mean(tf.reduce_sum(softmaxprob * tf.expand_dims(completion_potential, axis=1), axis=1))

        undersupplied_mask = tf.cast(supply_demand_ratio < 0.8, tf.float32)
        undersupplied_reward = tf.reduce_sum(softmaxprob * tf.expand_dims(undersupplied_mask * order_density, axis=1), axis=1)
        undersupplied_loss = -tf.reduce_mean(undersupplied_reward)

        policy_loss = actor_loss - 0.01 * entropy + 0.4 * orr_potential_loss + 0.2 * balance_loss + 0.3 * orr_direct_loss + 0.2 * completion_loss + 0.3 * undersupplied_loss

        self.policy_model = tf.keras.Model(
            inputs=[policy_state_input, policy_v_feature_input, action_input, advantage_input, 
                   neighbor_mask_input, policy_order_predict_input],
            outputs=[softmaxprob, actor_loss, entropy, policy_loss]
        )

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.policy_state_input = policy_state_input
        self.policy_v_feature_input = policy_v_feature_input
        self.action_input = action_input
        self.advantage_input = advantage_input
        self.neighbor_mask_input = neighbor_mask_input
        self.policy_order_predict_input = policy_order_predict_input
        self.softmaxprob = softmaxprob
        self.actor_loss = actor_loss
        self.entropy = entropy
        self.policy_loss = policy_loss

        return actor_loss, entropy

    def predict(self, s, v_feature, order_predict):

        value_output = self.value_model([s, v_feature, order_predict, tf.zeros((s.shape[0], 1))])
        return value_output[0]

    def compute_cold_hot_zone(self, vehicle, order_predict):
        order_predict = tf.reshape(order_predict, (-1, self.env.n_valid_grids))
        order_predict = tf.cast(order_predict, tf.float32)
        max_order_num = tf.reshape(tf.reduce_max(order_predict, axis=1), (-1, 1))
        curr_o = order_predict / max_order_num
        curr_d = vehicle[:, :self.env.n_valid_grids]
        curr_d = tf.cast(curr_d, tf.float32)
        mask = vehicle[:, self.env.n_valid_grids:]
        mask = tf.cast(mask, tf.float32)
        curr_o = tf.multiply(mask, curr_o)
        curr_d = tf.multiply(mask, curr_d)
        cold_hot_zone = curr_o - curr_d
        cold_hot_zone = tf.reshape(tf.reduce_sum(cold_hot_zone, axis=1), (-1, 1))
        return cold_hot_zone

    def action(self, s, context, order_predict, next_v, epsilon):

        predict = all_c_se_block(order_predict, self.predict_time)
        cold_hot_zone = self.compute_cold_hot_zone(next_v, predict)
        cold_hot = tf.numpy_function(lambda x: np.where(x > 0, 1, np.where(x < 0, -1, x)), [cold_hot_zone], tf.float32)

        value_output, o_p = self.value_model([s, next_v, order_predict, tf.zeros((s.shape[0], 1))])
        value_output = tf.reshape(value_output, [-1])
        value_output = value_output.numpy()
        o_p = predict
        o_p = tf.linalg.diag_part(o_p)

        action_tuple = []
        valid_prob = []

        action_choosen_mat = []
        policy_state = []
        curr_state_value = []
        next_state_ids = []
        policy_order_predict = []
        policy_next_v = []

        grid_ids = np.argmax(s[:, -self.n_valid_grid:], axis=1)

        curr_neighbor_mask = deepcopy(self.valid_action_mask)
        for idx, grid_valid_idx in enumerate(grid_ids):
            valid_qvalues = value_output[self.neighbors_list[grid_valid_idx]]
            temp_qvalue = np.zeros(self.action_dim)
            temp_qvalue[curr_neighbor_mask[grid_valid_idx] > 0] = valid_qvalues
            temp_qvalue[temp_qvalue < temp_qvalue[-1]] = 0
            curr_neighbor_mask[grid_valid_idx][np.where(temp_qvalue < temp_qvalue[-1])] = 0
            if np.sum(curr_neighbor_mask[grid_valid_idx]) == 0:
                curr_neighbor_mask[grid_valid_idx] = self.valid_action_mask[grid_valid_idx]

        action_probs = self.policy_model([s, next_v, tf.zeros((s.shape[0], self.action_dim)), 
                                        tf.zeros((s.shape[0], 1)), curr_neighbor_mask, order_predict])
        action_probs = action_probs[0]
        action_probs = action_probs.numpy()

        curr_neighbor_mask_policy = []

        for idx, grid_valid_idx in enumerate(grid_ids):
            action_prob = action_probs[idx]

            action_prob[self.valid_action_mask[grid_valid_idx] == 0] = 0

            valid_prob.append(action_prob)
            if int(context[idx]) == 0:
                continue

            prob_sum = np.sum(action_prob)
            if prob_sum > 0:
                action_prob_normalized = action_prob / prob_sum
            else:

                action_prob_normalized = np.ones(self.action_dim) / self.action_dim
                action_prob_normalized[self.valid_action_mask[grid_valid_idx] == 0] = 0
                prob_sum = np.sum(action_prob_normalized)
                if prob_sum > 0:
                    action_prob_normalized = action_prob_normalized / prob_sum
                else:

                    continue

            max_prob_idx = np.argmax(action_prob_normalized)

            prob_entropy = -np.sum(action_prob_normalized * np.log(action_prob_normalized + 1e-8))
            max_entropy = np.log(self.action_dim)
            normalized_entropy = prob_entropy / max_entropy

            max_prob = action_prob_normalized[max_prob_idx]
            prob_variance = np.var(action_prob_normalized)
            confidence = max_prob - prob_variance
            confidence = np.clip(confidence, 0.0, 1.0)

            driver_density = s[idx, :self.env.n_valid_grids] if hasattr(s, 'shape') and s.shape[1] > self.env.n_valid_grids else np.ones(self.env.n_valid_grids)
            order_density = s[idx, self.env.n_valid_grids:2*self.env.n_valid_grids] if hasattr(s, 'shape') and s.shape[1] > 2*self.env.n_valid_grids else np.ones(self.env.n_valid_grids)

            if order_density[grid_valid_idx] > 0:
                current_ratio = driver_density[grid_valid_idx] / (order_density[grid_valid_idx] + 1e-6)
            else:
                current_ratio = 1.0

            if current_ratio < 0.2:
                if confidence > 0.6:
                    temperature = 0.1
                    boost_factor = 5.0
                elif normalized_entropy < 0.3:
                    temperature = 0.2
                    boost_factor = 4.0
                else:
                    temperature = 0.3
                    boost_factor = 3.0
            elif current_ratio < 0.4:
                if confidence > 0.5:
                    temperature = 0.2
                    boost_factor = 4.0
                elif normalized_entropy < 0.4:
                    temperature = 0.3
                    boost_factor = 3.0
                else:
                    temperature = 0.4
                    boost_factor = 2.5
            elif current_ratio < 0.7:
                if confidence > 0.4:
                    temperature = 0.3
                    boost_factor = 3.0
                elif normalized_entropy < 0.5:
                    temperature = 0.4
                    boost_factor = 2.5
                else:
                    temperature = 0.5
                    boost_factor = 2.0
            elif current_ratio <= 1.0:
                if confidence > 0.3:
                    temperature = 0.5
                    boost_factor = 2.0
                elif normalized_entropy < 0.6:
                    temperature = 0.6
                    boost_factor = 1.5
                else:
                    temperature = 0.8
                    boost_factor = 1.2
            else:
                if confidence > 0.2:
                    temperature = 0.7
                    boost_factor = 1.0
                elif normalized_entropy < 0.7:
                    temperature = 0.8
                    boost_factor = 0.8
                else:
                    temperature = 1.0
                    boost_factor = 0.5

            action_prob_scaled = np.power(action_prob_normalized + 1e-8, 1.0 / temperature)
            action_prob_scaled = action_prob_scaled / np.sum(action_prob_scaled)

            confidence_boost = 1.0 + confidence * 0.5
            action_prob_scaled[max_prob_idx] *= boost_factor * confidence_boost
            action_prob_scaled = action_prob_scaled / np.sum(action_prob_scaled)

            try:

                neighbor_order_density = []
                neighbor_driver_density = []
                neighbor_ratios = []

                for neighbor_idx in self.neighbors_list[grid_valid_idx]:
                    if neighbor_idx < len(order_density):
                        neighbor_order_density.append(order_density[neighbor_idx])
                        neighbor_driver_density.append(driver_density[neighbor_idx])

                        if order_density[neighbor_idx] > 0:
                            ratio = driver_density[neighbor_idx] / (order_density[neighbor_idx] + 1e-6)
                        else:
                            ratio = 1.0
                        neighbor_ratios.append(ratio)
                    else:
                        neighbor_order_density.append(0.0)
                        neighbor_driver_density.append(0.0)
                        neighbor_ratios.append(1.0)

                if len(neighbor_order_density) > 0 and len(neighbor_order_density) <= len(action_prob_scaled):

                    neighbor_weights = np.zeros(len(neighbor_order_density))

                    order_weights = np.array(neighbor_order_density)
                    if np.sum(order_weights) > 0:
                        order_weights = order_weights / np.sum(order_weights)
                        neighbor_weights += order_weights * 0.4

                    ratio_weights = np.array(neighbor_ratios)
                    ratio_weights = np.exp(-ratio_weights * 2)
                    if np.sum(ratio_weights) > 0:
                        ratio_weights = ratio_weights / np.sum(ratio_weights)
                        neighbor_weights += ratio_weights * 0.3

                    distance_weights = np.ones(len(neighbor_order_density))
                    distance_weights[0] = 1.5
                    if len(distance_weights) > 1:
                        distance_weights[1:] = 1.0 / np.arange(1, len(distance_weights))
                    if np.sum(distance_weights) > 0:
                        distance_weights = distance_weights / np.sum(distance_weights)
                        neighbor_weights += distance_weights * 0.2

                    success_weights = np.ones(len(neighbor_order_density))

                    if np.sum(success_weights) > 0:
                        success_weights = success_weights / np.sum(success_weights)
                        neighbor_weights += success_weights * 0.1

                    if np.sum(neighbor_weights) > 0:
                        neighbor_weights = neighbor_weights / np.sum(neighbor_weights)

                        for i in range(min(len(action_prob_scaled), len(neighbor_weights))):

                            weight_factor = 1.0 + neighbor_weights[i] * 0.8

                            if i < len(neighbor_ratios) and neighbor_ratios[i] < 0.2:
                                weight_factor *= 6.0
                            elif i < len(neighbor_ratios) and neighbor_ratios[i] < 0.4:
                                weight_factor *= 4.0
                            elif i < len(neighbor_ratios) and neighbor_ratios[i] < 0.6:
                                weight_factor *= 3.0
                            elif i < len(neighbor_ratios) and neighbor_ratios[i] < 0.8:
                                weight_factor *= 2.0
                            elif i < len(neighbor_ratios) and neighbor_ratios[i] > 2.0:
                                weight_factor *= 0.3
                            elif i < len(neighbor_ratios) and neighbor_ratios[i] > 1.5:
                                weight_factor *= 0.5

                            if i < len(neighbor_order_density):
                                order_density_factor = 1.0 + neighbor_order_density[i] * 0.8
                                weight_factor *= order_density_factor

                            if i < len(neighbor_ratios):

                                success_rate = min(1.0, neighbor_ratios[i])
                                success_factor = 1.0 + success_rate * 0.5
                                weight_factor *= success_factor

                            if i < len(neighbor_order_density) and i < len(neighbor_ratios):
                                orr_potential = neighbor_order_density[i] / (neighbor_ratios[i] + 1e-6)
                                orr_factor = 1.0 + orr_potential * 0.3
                                weight_factor *= orr_factor

                            action_prob_scaled[i] *= weight_factor

                        action_prob_scaled = action_prob_scaled / np.sum(action_prob_scaled)

            except Exception as e:

                print(f"预测性调度计算失败: {e}")
                pass

            curr_action_indices_temp = np.random.choice(self.action_dim, int(context[idx]),
                                                        p=action_prob_scaled)

            curr_action_indices = [0] * self.action_dim
            for kk in curr_action_indices_temp:
                curr_action_indices[kk] += 1

            start_node_id = self.env.target_grids[grid_valid_idx]
            for curr_action_idx, num_driver in enumerate(curr_action_indices):
                if num_driver > 0:
                    end_node_id = int(self.valid_neighbor_node_id[grid_valid_idx, curr_action_idx])
                    if end_node_id != start_node_id:
                        action_tuple.append((start_node_id, end_node_id, num_driver))

                    temp_a = np.zeros(self.action_dim)
                    temp_a[curr_action_idx] = 1
                    action_choosen_mat.append(temp_a)
                    policy_state.append(s[idx])
                    curr_state_value.append(value_output[idx])
                    next_state_ids.append(self.valid_neighbor_grid_id[grid_valid_idx, curr_action_idx])
                    curr_neighbor_mask_policy.append(curr_neighbor_mask[idx])

                    policy_order_predict.append(order_predict[idx])
                    policy_next_v.append(next_v[idx])

        if hasattr(o_p, 'numpy'):
            o_p = o_p.numpy()

        if len(policy_state) == 0:

            return action_tuple, np.stack(valid_prob), \
                   np.array([]), np.array([]), [], \
                   np.array([]), [], np.array([]), o_p, \
                   np.array([])

        return action_tuple, np.stack(valid_prob), \
               np.stack(policy_state), np.stack(action_choosen_mat), curr_state_value, \
               np.stack(curr_neighbor_mask_policy), next_state_ids, np.stack(policy_order_predict), o_p, \
               np.stack(policy_next_v)

    def compute_advantage(self, curr_state_value, next_state_ids, next_state, node_reward, next_order_predict, next_v,
                          p_kl, gamma):
        advantage = []
        node_reward = node_reward.flatten()

        if next_state.ndim == 1:
            next_state = np.expand_dims(next_state, 0)

        try:
            qvalue_next = self.value_model([next_state, next_v, next_order_predict, tf.zeros((next_state.shape[0], 1))])
            qvalue_next = tf.reshape(qvalue_next[0], [-1])
            qvalue_next = qvalue_next.numpy()
        except Exception as e:
            print(f"Value model error in advantage: {e}")
            qvalue_next = np.zeros(self.n_valid_grid)

        if hasattr(p_kl, 'numpy'):
            p_kl = p_kl.numpy()
        p_kl = np.array(p_kl).flatten()

        node_reward = np.array(node_reward).flatten()

        for idx, next_state_id in enumerate(next_state_ids):
            next_state_id = int(next_state_id)

            if (next_state_id < len(qvalue_next) and 
                next_state_id < len(node_reward) and 
                next_state_id < len(p_kl) and
                idx < len(curr_state_value)):

                temp_adv = (-p_kl[next_state_id] + 
                           node_reward[next_state_id] + 
                           gamma * qvalue_next[next_state_id] - 
                           curr_state_value[idx])
            else:

                temp_adv = node_reward[min(next_state_id, len(node_reward)-1)] - curr_state_value[min(idx, len(curr_state_value)-1)]

            advantage.append(temp_adv)

        return advantage

    def compute_targets(self, valid_prob, next_state, node_reward, next_order_predict, o_p, o, next_v, gamma):
        targets = []
        node_reward = node_reward.flatten()

        if next_state.ndim == 1:
            next_state = np.expand_dims(next_state, 0)

        try:
            qvalue_next = self.value_model([next_state, next_v, next_order_predict, tf.zeros((next_state.shape[0], 1))])
            qvalue_next = tf.reshape(qvalue_next[0], [-1])
            qvalue_next = qvalue_next.numpy()
        except Exception as e:
            print(f"Value model error: {e}")
            qvalue_next = np.zeros(self.n_valid_grid)

        o_real = o[self.env.n_valid_grids:] if len(o) > self.env.n_valid_grids else o
        o_predict = o_p.flatten() if hasattr(o_p, 'flatten') else o_p

        min_len = min(len(o_real), len(o_predict), self.n_valid_grid)
        o_real = o_real[:min_len]
        o_predict = o_predict[:min_len]

        prediction_error = np.abs(o_predict - o_real)

        if len(prediction_error) < self.n_valid_grid:
            prediction_error = np.pad(prediction_error, (0, self.n_valid_grid - len(prediction_error)), 'constant')

        p_kl = prediction_error

        for idx in range(self.n_valid_grid):

            if idx < len(valid_prob) and idx < len(self.valid_action_mask):
                valid_mask = self.valid_action_mask[idx] > 0
                grid_prob = valid_prob[idx][valid_mask]
            else:
                grid_prob = np.ones(self.action_dim) / self.action_dim

            if len(grid_prob) > 0 and np.sum(grid_prob) > 0:
                grid_prob = grid_prob / np.sum(grid_prob)

            neighbor_grid_ids = self.neighbors_list[idx] if idx < len(self.neighbors_list) else [idx]

            safe_neighbor_ids = [nid for nid in neighbor_grid_ids if nid < len(node_reward) and nid < len(qvalue_next)]

            if len(safe_neighbor_ids) > 0 and len(grid_prob) >= len(safe_neighbor_ids):

                neighbor_rewards = node_reward[safe_neighbor_ids]
                neighbor_qvalues = qvalue_next[safe_neighbor_ids]
                neighbor_errors = prediction_error[safe_neighbor_ids]

                curr_grid_target = np.sum(grid_prob[:len(safe_neighbor_ids)] * 
                                       (neighbor_rewards - neighbor_errors + gamma * neighbor_qvalues))
            else:

                curr_grid_target = node_reward[idx] + gamma * qvalue_next[idx] - prediction_error[idx]

            targets.append(curr_grid_target)

        return np.array(targets).reshape([-1, 1]), p_kl

    def initialization(self, s, y, learning_rate):

        self.value_optimizer.learning_rate = learning_rate
        with tf.GradientTape() as tape:
            _, value_loss = self.value_model([s, tf.zeros((s.shape[0], self.env.n_valid_grids*2)), 
                                            tf.zeros((s.shape[0], self.env.n_valid_grids + (144 + self.predict_time - 1) + 
                                                    self.env.n_valid_grids, self.predict_time)), y])
        gradients = tape.gradient(value_loss, self.value_model.trainable_variables)
        self.value_optimizer.apply_gradients(zip(gradients, self.value_model.trainable_variables))
        return value_loss

    def update_policy(self, policy_state, advantage, action_choosen_mat, curr_neighbor_mask, policy_o, policy_v,
                      learning_rate, global_step):

        self.policy_optimizer.learning_rate = learning_rate
        with tf.GradientTape() as tape:
            _, actor_loss, entropy, policy_loss = self.policy_model([policy_state, policy_v, action_choosen_mat, 
                                                                   advantage, curr_neighbor_mask, policy_o])
        gradients = tape.gradient(policy_loss, self.policy_model.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(gradients, self.policy_model.trainable_variables))

        if self.summary_writer:
            with self.summary_writer.as_default():
                tf.summary.scalar("policy_loss", actor_loss, step=global_step)
                tf.summary.scalar("adv", tf.reduce_mean(advantage), step=global_step)
                tf.summary.scalar("entropy", entropy, step=global_step)
        return policy_loss

    def update_value(self, s, y, o, v, learning_rate, global_step):

        self.value_optimizer.learning_rate = learning_rate
        with tf.GradientTape() as tape:
            value_output, value_loss = self.value_model([s, v, o, y])
        gradients = tape.gradient(value_loss, self.value_model.trainable_variables)
        self.value_optimizer.apply_gradients(zip(gradients, self.value_model.trainable_variables))

        if self.summary_writer:
            with self.summary_writer.as_default():
                tf.summary.scalar("value_loss", value_loss, step=global_step)
                tf.summary.scalar("value_output", tf.reduce_mean(value_output), step=global_step)
        return value_loss

    def preassign_predict(self, curr_o, curr_d):
        azz = self.env.get_observation()[1]
        curr_o = curr_o.flatten()
        max_order_num = np.max(curr_o[curr_o != 0])
        curr_o = curr_o / max_order_num
        curr_o = curr_o * np.max(azz)
        curr_o = curr_o.astype(np.int32)
        curr_o = curr_o.astype(np.float64)
        remain_drivers = curr_d - curr_o
        remain_drivers[remain_drivers < 0] = 0

        remain_orders = curr_o - curr_d
        remain_orders[remain_orders < 0] = 0

        if np.sum(remain_orders) == 0 or np.sum(remain_drivers) == 0:
            context = np.concatenate((remain_drivers, remain_orders), axis=0)
            return context

        d = remain_drivers
        return d

class stateProcessor:

    def __init__(self,
                 target_id_states,
                 target_grids,
                 n_valid_grids):
        self.target_id_states = target_id_states
        self.target_grids = target_grids
        self.n_valid_grids = n_valid_grids
        self.T = 144
        self.action_dim = 7
        self.extend_state = True

    def utility_conver_states(self, curr_state):
        curr_s = np.array(curr_state).flatten()
        curr_s_new = [curr_s[idx] for idx in self.target_id_states]
        return np.array(curr_s_new)

    def utility_normalize_states(self, curr_s):
        max_driver_num = np.max(curr_s[:self.n_valid_grids])
        max_order_num = np.max(curr_s[self.n_valid_grids:])
        if max_order_num == 0:
            max_order_num = 1
        if max_driver_num == 0:
            max_driver_num = 1
        curr_s_new = np.zeros_like(curr_s)
        curr_s_new[:self.n_valid_grids] = curr_s[:self.n_valid_grids] / max_driver_num
        curr_s_new[self.n_valid_grids:] = curr_s[self.n_valid_grids:] / max_order_num
        return curr_s_new

    def enhanced_state_representation(self, curr_s, curr_city_time):

        if np.all(curr_s == 0):
            print("⚠️  Warning: Input state is all zeros! Creating dummy features.")

            curr_s = np.random.rand(len(curr_s)) * 0.1

        driver_density = curr_s[:self.n_valid_grids]
        order_density = curr_s[self.n_valid_grids:2*self.n_valid_grids]

        if np.all(driver_density == 0) and np.all(order_density == 0):
            print("⚠️  Warning: Driver and order densities are zero! Creating dummy data.")
            driver_density = np.random.rand(self.n_valid_grids) * 0.5
            order_density = np.random.rand(self.n_valid_grids) * 0.3

        supply_demand_ratio = np.zeros_like(driver_density)
        valid_mask = (order_density > 0) & (driver_density > 0)
        supply_demand_ratio[valid_mask] = driver_density[valid_mask] / (order_density[valid_mask] + 1e-6)
        supply_demand_ratio = np.nan_to_num(supply_demand_ratio, nan=1.0, posinf=10.0, neginf=0.0)

        balance_state = np.zeros_like(driver_density)
        balance_state[supply_demand_ratio < 0.5] = 0
        balance_state[(supply_demand_ratio >= 0.5) & (supply_demand_ratio < 0.8)] = 1
        balance_state[(supply_demand_ratio >= 0.8) & (supply_demand_ratio <= 1.2)] = 2
        balance_state[(supply_demand_ratio > 1.2) & (supply_demand_ratio <= 2.0)] = 3
        balance_state[supply_demand_ratio > 2.0] = 4

        urgency_level = np.zeros_like(driver_density)
        urgency_level[order_density > 0] = order_density[order_density > 0] / (supply_demand_ratio[order_density > 0] + 1e-6)
        urgency_level = np.nan_to_num(urgency_level, nan=0.0, posinf=100.0, neginf=0.0)

        global_driver_avg = np.mean(driver_density[driver_density > 0]) if np.any(driver_density > 0) else 1.0
        global_order_avg = np.mean(order_density[order_density > 0]) if np.any(order_density > 0) else 1.0

        relative_driver_density = driver_density / (global_driver_avg + 1e-6)
        relative_order_density = order_density / (global_order_avg + 1e-6)

        time_one_hot = np.zeros(144)
        time_one_hot[curr_city_time % 144] = 1
        normalized_time = curr_city_time / 144.0

        hour_of_day = (curr_city_time % 144) / 6
        time_sin = np.sin(2 * np.pi * hour_of_day / 24)
        time_cos = np.cos(2 * np.pi * hour_of_day / 24)

        day_of_week = (curr_city_time // 144) % 7
        is_weekend = 1.0 if day_of_week >= 5 else 0.0

        is_peak_hour = 1.0 if 6 <= hour_of_day <= 9 or 17 <= hour_of_day <= 20 else 0.0

        neighbor_driver_avg = np.zeros_like(driver_density)
        neighbor_order_avg = np.zeros_like(order_density)
        neighbor_ratio_avg = np.zeros_like(driver_density)

        for i in range(self.n_valid_grids):
            if i < len(self.neighbors_list):
                neighbor_indices = self.neighbors_list[i]
                if len(neighbor_indices) > 0:
                    valid_neighbors = [idx for idx in neighbor_indices if idx < len(driver_density)]
                    if valid_neighbors:
                        neighbor_driver_avg[i] = np.mean(driver_density[valid_neighbors])
                        neighbor_order_avg[i] = np.mean(order_density[valid_neighbors])
                        neighbor_ratios = []
                        for idx in valid_neighbors:
                            if order_density[idx] > 0:
                                ratio = driver_density[idx] / (order_density[idx] + 1e-6)
                                neighbor_ratios.append(ratio)
                        if neighbor_ratios:
                            neighbor_ratio_avg[i] = np.mean(neighbor_ratios)

        trend_indicator = np.zeros_like(driver_density)
        trend_indicator[urgency_level > np.percentile(urgency_level, 75)] = 1.0
        trend_indicator[urgency_level < np.percentile(urgency_level, 25)] = -1.0

        completion_potential = np.zeros_like(driver_density)
        completion_potential[valid_mask] = np.minimum(
            driver_density[valid_mask], order_density[valid_mask]
        ) / (order_density[valid_mask] + 1e-6)
        completion_potential = np.nan_to_num(completion_potential, nan=0.0, posinf=1.0, neginf=0.0)

        supply_demand_gap = np.zeros_like(driver_density)
        supply_demand_gap[order_density > 0] = (order_density[order_density > 0] - driver_density[order_density > 0]) / (order_density[order_density > 0] + 1e-6)
        supply_demand_gap = np.nan_to_num(supply_demand_gap, nan=0.0, posinf=10.0, neginf=-10.0)

        response_priority = np.zeros_like(driver_density)
        response_priority[order_density > 0] = order_density[order_density > 0] / (supply_demand_ratio[order_density > 0] + 1e-6)
        response_priority = np.nan_to_num(response_priority, nan=0.0, posinf=100.0, neginf=0.0)

        region_activity = np.zeros_like(driver_density)
        region_activity = (driver_density + order_density) / 2.0

        enhanced_features = np.concatenate([
            driver_density,
            order_density,
            supply_demand_ratio,
            balance_state,
            urgency_level,
            relative_driver_density,
            relative_order_density,
            neighbor_driver_avg,
            neighbor_order_avg,
            neighbor_ratio_avg,
            trend_indicator,
            completion_potential,
            supply_demand_gap,
            response_priority,
            region_activity,
            time_one_hot,
            [normalized_time, time_sin, time_cos, hour_of_day, is_weekend, is_peak_hour]
        ])

        return enhanced_features

    def utility_conver_reward(self, reward_node):
        reward_node_new = [reward_node[idx] for idx in self.target_grids]
        return np.array(reward_node_new)

    def reward_wrapper(self, info, curr_s):
        info_reward = info[0]
        valid_nodes_reward = self.utility_conver_reward(info_reward[0])
        devide = curr_s[:self.n_valid_grids]
        devide[devide == 0] = 1
        valid_nodes_reward = valid_nodes_reward/devide

        order_density = curr_s[self.n_valid_grids:]
        driver_density = curr_s[:self.n_valid_grids]

        supply_demand_ratio = np.zeros_like(order_density)
        valid_mask = (order_density > 0) & (driver_density > 0)
        supply_demand_ratio[valid_mask] = driver_density[valid_mask] / (order_density[valid_mask] + 1e-6)
        supply_demand_ratio = np.nan_to_num(supply_demand_ratio, nan=1.0, posinf=10.0, neginf=0.0)

        response_reward = np.zeros_like(order_density)

        response_reward[supply_demand_ratio < 0.5] = 5.0

        response_reward[(supply_demand_ratio >= 0.5) & (supply_demand_ratio < 0.8)] = 3.0

        response_reward[(supply_demand_ratio >= 0.8) & (supply_demand_ratio <= 1.2)] = 1.0

        response_reward[supply_demand_ratio > 1.2] = 0.1

        density_reward = order_density * 2.0

        balance_reward = np.zeros_like(order_density)
        ideal_ratio = 1.0
        ratio_diff = np.abs(supply_demand_ratio - ideal_ratio)
        balance_reward = np.exp(-ratio_diff * 2)

        coverage_reward = np.zeros_like(order_density)
        driver_density_norm = driver_density / (np.max(driver_density) + 1e-6)
        coverage_reward = (1.0 - driver_density_norm) * order_density

        efficiency_reward = np.zeros_like(order_density)
        efficiency_reward[valid_mask] = np.minimum(
            order_density[valid_mask], driver_density[valid_mask]
        ) * 0.5

        total_reward = (
            valid_nodes_reward * 0.1 +
            response_reward * 5.0 +
            density_reward * 3.0 +
            balance_reward * 1.5 +
            coverage_reward * 4.0 +
            efficiency_reward * 3.0
        )

        return total_reward

    def compute_context(self, info):

        context = info.flatten()
        context = [context[idx] for idx in self.target_grids]
        return context

    def to_grid_states(self, curr_s, curr_city_time):
        T = self.T

        enhanced_s = self.enhanced_state_representation(curr_s, curr_city_time)

        enhanced_dim = len(enhanced_s)

        s_grid = np.zeros((self.n_valid_grids, enhanced_dim))

        for i in range(self.n_valid_grids):
            s_grid[i, :] = enhanced_s

        return np.array(s_grid)

    def to_grid_rewards(self, node_reward):
        return np.array(node_reward).reshape([-1, 1])

    def to_action_mat(self, action_neighbor_idx):
        action_mat = np.zeros((len(action_neighbor_idx), self.action_dim))
        action_mat[np.arange(action_mat.shape[0]), action_neighbor_idx] = 1
        return action_mat

class policyReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = []

        self.neighbor_mask = []
        self.actions = []
        self.rewards = []
        self.p_o = []

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    def add(self, s, a, r, mask, p_o, p_next_v):

        if len(s) == 0 or s.size == 0:
            return

        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.neighbor_mask = mask
            self.p_o = p_o
            self.p_next_v = p_next_v
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s),axis=0)
            self.neighbor_mask = np.concatenate((self.neighbor_mask, mask), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.p_o = np.concatenate((self.p_o, p_o), axis=0)
            self.p_next_v = np.concatenate((self.p_next_v, p_next_v), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]

            index = random.randint(0, self.curr_lens - new_sample_lens)

            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.neighbor_mask[index:(index + new_sample_lens)] = mask
            self.p_o[index:(index + new_sample_lens)] = p_o
            self.p_next_v[index:(index + new_sample_lens)] = p_next_v

    def sample(self):
        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, np.array(self.rewards), self.neighbor_mask, self.p_o, self.p_next_v]

        indices = random.sample(range(0, self.curr_lens), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.neighbor_mask[indices]
        batch_p_o = self.p_o[indices]
        batch_p_next_v = self.p_next_v[indices]
        return [batch_s, batch_a, batch_r, batch_mask, batch_p_o, batch_p_next_v]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.neighbor_mask = []
        self.p_o = []
        self.p_next_v = []
        self.curr_lens = 0

class ReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.order = []
        self.priorities = []

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001

    def add(self, s, a, r, next_s, order, next_v, priority=None):

        if len(s) == 0 or s.size == 0:
            return

        if priority is None:

            base_priority = np.abs(r).mean() + 1e-6

            if hasattr(s, 'shape') and s.shape[1] > 2 * 504:
                try:

                    supply_demand_ratio = s[:, 2*504:3*504] if s.shape[1] > 3*504 else np.ones((s.shape[0], 504))
                    order_density = s[:, 504:2*504] if s.shape[1] > 2*504 else np.ones((s.shape[0], 504))

                    orr_potential = np.mean(order_density / (supply_demand_ratio + 1e-6), axis=1)
                    orr_priority = np.mean(orr_potential) + 1e-6

                    balance_priority = np.mean(np.abs(supply_demand_ratio - 1.0)) + 1e-6

                    priority = base_priority + 0.5 * orr_priority + 0.3 * balance_priority
                except:
                    priority = base_priority
            else:
                priority = base_priority

        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.next_states = next_s
            self.order = order
            self.next_v = next_v
            self.priorities = np.full(s.shape[0], priority)
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s),axis=0)
            self.next_states = np.concatenate((self.next_states, next_s), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.order = np.concatenate((self.order, order), axis=0)
            self.next_v = np.concatenate((self.next_v, next_v), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]

            index = random.randint(0, self.curr_lens - new_sample_lens)

            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.next_states[index:(index + new_sample_lens)] = next_s
            self.order[index:(index + new_sample_lens)] = order
            self.next_v[index:(index + new_sample_lens)] = next_v

    def sample(self):
        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, self.rewards, self.next_states, self.order, self.next_v]

        priorities = np.array(self.priorities[:self.curr_lens])
        probabilities = priorities ** self.alpha
        probabilities = probabilities / np.sum(probabilities)

        if len(probabilities) != self.curr_lens:
            probabilities = np.ones(self.curr_lens) / self.curr_lens

        indices = np.random.choice(self.curr_lens, self.batch_size, p=probabilities)

        weights = (self.curr_lens * probabilities[indices]) ** (-self.beta)
        weights = weights / np.max(weights)

        self.beta = min(1.0, self.beta + self.beta_increment)

        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices] * weights.reshape(-1, 1)
        batch_mask = self.next_states[indices]
        batch_order = self.order[indices]
        batch_next_v = self.next_v[indices]
        return [batch_s, batch_a, batch_r, batch_mask, batch_order, batch_next_v]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.order = []
        self.next_v = []
        self.curr_lens = 0

class ModelParametersCopier():

    def __init__(self, estimator1, estimator2):
        self.estimator1 = estimator1
        self.estimator2 = estimator2

    def make(self):

        for src_var, dst_var in zip(self.estimator1.value_model.trainable_variables, 
                                   self.estimator2.value_model.trainable_variables):
            dst_var.assign(src_var)

        for src_var, dst_var in zip(self.estimator1.policy_model.trainable_variables, 
                                   self.estimator2.policy_model.trainable_variables):
            dst_var.assign(src_var)
