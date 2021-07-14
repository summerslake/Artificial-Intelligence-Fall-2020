# Nick Gustafson and Lake Summers
# crawler.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this file, you should test your Q-learning implementation on the robot crawler environment 
that we saw in class. It is suggested to test your code in the grid world environments before this one.

The package `matplotlib` is needed for the program to run.


The Crawler environment has discrete state and action spaces
and provides both of model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
  

Once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
"""


# use random library if needed
import random
import numpy as np

def q_learning(env, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """

    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    gamma = 0.95   

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    # maximum number of training iterations
    max_iterations = 5000
    #########################

### Please finish the code below ##############################################
###############################################################################
  # Set all Q(s,a) to 0
    qValues = [[0 for x in range(NUM_ACTIONS)] for x in range(NUM_STATES)]
    
    #Make a function to get the next action
    def getAction(state, iteration):
        #For the last 10% of iterations, set epsilon to .01
        if iteration >= max_iterations * 0.9:
            eps = 0.01
        #Otherwise
        else:
            eps = 1 - (iteration / max_iterations)
        action_probabilities = np.ones(NUM_ACTIONS, dtype = float) * eps / NUM_ACTIONS

        best_action = qValues[state].index(max((qValues[state])))
        action_probabilities[best_action] += (1.0 - eps)
        return np.random.choice(np.arange( 
                      len(action_probabilities)), 
                       p = action_probabilities)

    for i in range(max_iterations):
        #Get the initial state
        state = env.reset()
        #Set terminal to false
        terminal = False

        #While episode is still going
        while not terminal:
            #Set target to 0
            target = 0
            #Choose which action to take in the episode
            action = getAction(state, i)
            #Observe reward r and next state s'
            (nextState, reward, terminal, _) = env.step(action)
            #If the next state is terminal
            if terminal:
                target = reward
            #If the next state is not terminal
            else:
                #Target = r + (gamma * max Q val from nextState)
                target = reward + (gamma * max(qValues[nextState]))
            #Bellman update
            qValues[state][action] = ((1 - alpha) * qValues[state][action]) + (alpha * target)
            #Move to the next state
            state = nextState

    for i, qValue in enumerate(qValues):
        pi[i] = qValue.index(max(qValue))
###############################################################################
    return pi


if __name__ == "__main__":
    from app.crawler import App
    import tkinter as tk
    
    algs = {
        "Q Learning": q_learning,
    }

    root = tk.Tk()
    App(algs, root)
    root.mainloop()