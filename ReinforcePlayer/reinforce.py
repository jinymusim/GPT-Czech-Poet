import argparse
import os
import collections
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
import sys

import gym
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym_super_mario_bros as mario

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--render_each", default=1, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--env", default="SuperMarioBros-v2", type=str, help="Environment.")
parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=1, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
parser.add_argument("--convlotuions", default=2, type=int, help="Number of Convolutions.")
parser.add_argument("--filters", default=4, type=int, help="Number of filters in convolutions.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--replay_buffer_size", default=100_000, type=int, help="Replay buffer size.")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
parser.add_argument("--model_path", default="mario_model", type=str, help="Output file for model.")




class Network:
    
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        
        actor_input = tf.keras.layers.Input(shape=env.observation_space.shape, dtype=tf.float32)
        actor_hidden = actor_input
        for i in range(args.convlotuions):
            actor_hidden = tf.keras.layers.Conv2D(kernel_size=3, filters=args.filters, strides=2, use_bias=False)(actor_hidden)
            actor_hidden = tf.keras.layers.BatchNormalization()(actor_hidden)
            actor_hidden = tf.keras.layers.ReLU()(actor_hidden)
        actor_hidden = tf.keras.layers.Flatten()(actor_hidden)
        actor_output = tf.keras.layers.Dense(env.action_space.n, activation=tf.nn.softmax)(actor_hidden)
        
        critic_input = tf.keras.layers.Input(shape=env.observation_space.shape, dtype=tf.float32)
        critic_hidden = critic_input
        for i in range(args.convlotuions):
            critic_hidden = tf.keras.layers.Conv2D(kernel_size=3, filters=args.filters, strides=2, use_bias=False)(critic_hidden)
            critic_hidden = tf.keras.layers.BatchNormalization()(critic_hidden)
            critic_hidden = tf.keras.layers.ReLU()(critic_hidden)
        critic_hidden = tf.keras.layers.Flatten()(critic_hidden)
        critic_output = tf.keras.layers.Dense(1, activation=None)(critic_hidden)
        
        self._actor = tf.keras.Model(inputs=actor_input, outputs=actor_output)
        self._actor.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        )
        
        self._critic = tf.keras.Model(inputs=critic_input, outputs=critic_output)
        self._critic.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss = tf.keras.losses.MeanSquaredError()
        )

    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        
        with tf.GradientTape() as otherTape:
            vals = self._critic(states)
            critic_loss = self._critic.loss(returns, vals)
        grad2 = otherTape.gradient(critic_loss, self._critic.trainable_variables)
        with tf.GradientTape() as tape:
            outputs = self._actor(states)
            action_dist = tfp.distributions.Categorical(outputs)
            loss = - action_dist.log_prob(actions) * (returns-vals) \
                   - args.entropy_regularization * action_dist.entropy()
        
        grad1 = tape.gradient(loss, self._actor.trainable_variables)
        self._actor.optimizer.apply_gradients(zip(grad1,self._actor.trainable_variables))
        self._critic.optimizer.apply_gradients(zip(grad2,self._critic.trainable_variables))
        
    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_actions(self, states: np.ndarray) -> np.ndarray:
        return self._actor(states)

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        return self._critic(states)
    
    def save_weights(self, filepath):
        self._actor.save_weights(f"{filepath}-0")
        self._critic.save_weights(f"{filepath}-1")

    def load_weights(self, filepath):
        self._actor.load_weights(f"{filepath}-0")
        self._critic.load_weights(f"{filepath}-1")


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        while not done:
            # TODO: Predict the action using the greedy policy.
                                # Changed to work with batch dimension
            action = np.argmax(network.predict_actions([state])[0,:])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the mario environment
    
    best_performance = 0

    replay_buffer = collections.deque(maxlen=args.replay_buffer_size)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])
    while True:
        # Training
        for _ in range(args.evaluate_each):
            mario_env =  mario.make(args.env)
            state, done = mario_env.reset(seed=args.seed), False
            i=0
            while not done and i<5000:
                # Choose actions using `network.predict_actions`.
                # TODO: this is weird, why is there supposed to be a log?
                action = np.argmax(network.predict_actions([state])[0])
                # Perform steps in the vectorized environment
                next_state, reward, done, _ = mario_env.step(action)
                replay_buffer.append(Transition(state, action,reward, done, next_state))
                state = next_state
                i+=1
                # Training
                if len(replay_buffer) >= 4 * args.batch_size:
                    # Note that until now we used `np.random.choice` with `replace=False` to generate
                    # batch indices. However, this call is extremely slow for large buffers, because
                    # it generates a whole permutation. With `np.random.randint`, indices may repeat,
                    # but once the buffer is large, it happens with little probability.
                    batch = np.random.randint(len(replay_buffer), size=args.batch_size)
                    states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))
                    returns = rewards[:,None] + args.gamma * network.predict_values(next_states) * np.logical_not(dones)[:,None]
                    network.train(states, actions, returns)
        
        if np.mean(env._episode_returns[-100:]) >= best_performance:
            network.save_weights(args.model_path)
            best_performance = np.mean(env._episode_returns[-15:])
        #network.train(states, actions, Rs)
        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(mario.make(args.env), args.seed, args.render_each)

    # TODO: args.learning_rate/=(args.evaluate_each*args.workers)?
    main(env, args)
