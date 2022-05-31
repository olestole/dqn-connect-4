import tensorflow as tf
import numpy as np
import logging

from history import History

class DQN():
    def __init__(self, input_shape, n_outputs):
        self.n_outputs = n_outputs
        self.input_shape = input_shape
        self.model = self.create_network(input_shape, n_outputs)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = tf.losses.mean_squared_error
    
    def create_network(self, input_shape, n_outputs):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(n_outputs, activation='softmax')
        ])
        model.summary()
        return model
    
    def predict(self, state):
        return self.model.predict(state, verbose=0)
        
    def update(self, experiences, discount_factor, target_network):
        # Use the sampled batch from replay buffer to update the network
        states, actions, rewards, next_states, dones = experiences
        # Sample next_Q_values from the target_network, and not the main_network
        next_Q_values = target_network.predict(np.array(next_states))
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + (1 - dones) * discount_factor * max_next_Q_values)
        mask = tf.one_hot(actions, self.n_outputs)
        
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        # logging.info(f"Loss: {loss}")
        return loss
        
    def save_weights(self, path):
        self.model.save_weights(path)
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)