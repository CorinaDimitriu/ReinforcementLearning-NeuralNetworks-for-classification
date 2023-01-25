from baselines.ppo2 import ppo2
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines import bench
from baselines import logger
from MnistEnvironment import MnistEnvironment
from IrisEnvironment import IrisEnvironment
import tensorflow as tf


def apply_ppo(training_set, testing_set, environment):
    accuracies = []
    logger.configure('./logs/mnist_ppo', ["stdout", "tensorboard"])
    x_train, y_train = training_set
    training_env = DummyVecEnv([lambda: bench.Monitor(environment(
        (x_train, y_train), images_per_episode=1), logger.get_dir(), allow_early_resets=True)])
    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
        for epoch in range(50):
            ppo_trained_model = ppo2.learn(env=training_env, network='mlp', num_layers=2,
                                                    num_hidden=64, nsteps=32,
                                                    total_timesteps=x_train.shape[0],
                                                save_interval=1000)
            accuracies.append(eval_ppo(ppo_trained_model, testing_set, environment))
    return accuracies


def eval_ppo(trained_model, testing_set, environment):
    attempted, correctly_predicted = 0, 0
    x_test, y_test = testing_set
    testing_env = DummyVecEnv([lambda: environment(images_per_episode=1,
                                                        dataset=(x_test, y_test),
                                                        train=False)])
    try:
        while True:
            obs, done = testing_env.reset(), [False]
            while not done[0]:
                observation, reward, done, _ = testing_env.step(
                    trained_model.step(obs[None])[0])
                attempted += 1
                if reward[0] > 0:
                    correctly_predicted += 1
    except StopIteration:
        # report = open("Accuracy.txt", "a")
        # print('\nFinished validation with %f accuracy\n'
        #       % ((float(correctly_predicted) / attempted) * 100), file=report)
        return (float(correctly_predicted) / attempted) * 100
