import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import sys
import gym

env = gym.make('CartPole-v1').env

#Hyperparameters
nbr_neurones = 32
batch_size = 1
learning_rate = 1e-1
gamma = 0.99

dimen = 4

tf.reset_default_graph()

observations = tf.placeholder(tf.float32, [None, dimen])


poids1 = tf.Variable(tf.random_normal([4,nbr_neurones]))
layer1 = tf.matmul(observations,poids1)


poids2 = tf.Variable(tf.random_normal([nbr_neurones,1]))
output = tf.nn.sigmoid(tf.matmul(layer1,poids2))


variables = [poids1, poids2]
input_y = tf.placeholder(tf.float32, [None,1])
avantages = tf.placeholder(tf.float32)


log_vraisemenblance = tf.log(input_y * (input_y - output) + (1 - input_y) * (output))
loss = -tf.reduce_mean(log_vraisemenblance * avantages)

grad = tf.gradients(loss, variables)
p1_grad = tf.placeholder(tf.float32)
p2_grad = tf.placeholder(tf.float32)

# Learning
P = [p1_grad, p2_grad]
optimisation = tf.train.AdamOptimizer(learning_rate=learning_rate)
update_grad = optimisation.apply_gradients(zip(P, [poids1, poids2]))


def fonction_avantage (recompenses) :
    return np.array([val * (gamma ** i) for i, val in enumerate(recompenses)])


reward_sum = 0

observations_cumulees = np.empty(0).reshape(0,4)
action_y = np.empty(0).reshape(0,1)
rewards = np.empty(0).reshape(0,1)

sess = tf.Session()
rendering = False
sess.run(tf.global_variables_initializer())
observation = env.reset()

gradients = np.array([np.zeros(var.get_shape()) for var in variables])

episodes_max = 10000
num_episode = 0

while num_episode < episodes_max:
    if reward_sum/batch_size > 500 or rendering == True :
        env.render()
        rendering = True
    x = np.reshape(observation, [1, 4])

    proba = sess.run(output, feed_dict={observations: x})

    y = 0 if np.random.rand() < proba else 1

    observations_cumulees = np.vstack([observations_cumulees, x])
    action_y = np.vstack([action_y, y])

    observation, reward, done, info = env.step(y)
    reward_sum += reward
    rewards = np.vstack([rewards, reward])

    if done == True :

        A = fonction_avantage(rewards)
        A -= A.mean()
        A /= A.std()

        gradients += np.array(sess.run(grad, feed_dict={observations: observations_cumulees,
                                               input_y: action_y,
                                               avantages: A}))

        observations_cumulees = np.empty(0).reshape(0,4)
        action_y= np.empty(0).reshape(0,1)
        rewards = np.empty(0).reshape(0,1)


        if num_episode % batch_size == 0:

            sess.run(update_grad, feed_dict={p1_grad: gradients[0],
                                             p2_grad: gradients[1]})

            gradients *= 0

            print("Récompense moyenne pour le ",num_episode,"ème épisode : ",reward_sum/batch_size)

            if reward_sum / batch_size > 2000:
                print("Résolu en ",num_episode,"épisodes!")
                break
            reward_sum = 0
        num_episode += 1
        observation = env.reset()