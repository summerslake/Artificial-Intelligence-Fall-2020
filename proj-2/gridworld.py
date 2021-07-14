# Nick Gustafson and Lake Summers
# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this assignment, you will implement three classic algorithm for 
solving Markov Decision Processes either offline or online. 
These algorithms include: value_iteration, policy_iteration and q_learning.
You will test your implementation on three grid world environments. 
You will also have the opportunity to use Q-learning to control a simulated robot 
in crawler.py

The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model 
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In q_learning, once a terminal state is reached the environment should be (re)initialized by
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
You can also only update the visualization of the v values by
    logger.log(i, v)
"""


# use random library if needed
import random
import numpy as np


def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.
    See lines 20-40 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.  
        In this case, you may want to exit the algorithm earlier. A way to check 
        if value iteration has already converged is to check whether 
        the max over (or sum of) L1 or L2 norms between the values before and
        after an iteration is small enough. For the Grid World environment, 1e-4
        is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the value and policy 
    logger.log(0, v, pi)
    # At each iteration, you may need to keep track of pi to perform logging
   
### Please finish the code below ##############################################
###############################################################################
    #Repeat for max number of iterations
    for step in range(1, max_iterations):
        #Keep track of old values
        oldValues = v.copy()

        #For each state S in observation space
        for state in range(NUM_STATES):
            #Set default low value
            maxActionValue = -10000
            #For each action in state
            for action in range(NUM_ACTIONS):
                #Set initial val to 0
                value = 0
                #For each result in the transition model
                for (probability, nextState, reward, terminal) in TRANSITION_MODEL[state][action]:
                    #Bellman update
                    value += probability * (reward + (gamma * oldValues[nextState]))
                #If the found value is greater than the current max action value
                if value > maxActionValue:
                    #Update the current max action value
                    maxActionValue = value
                    #Update the policy with the current action
                    pi[state] = action
                    #Update the current value with the found value
                    v[state] = value
        
        #Update the logger
        logger.log(step,v,pi)

        #Check for convergence
        if ((sum(v) - sum(oldValues)) <= 1e-4) and ((sum(v) - sum(oldValues)) >= -1e-4):
            break

###############################################################################
    return pi


def policy_iteration(env, gamma, max_iterations, logger):
    """
    Implement policy iteration to return a deterministic policy for all states.
    See lines 20-40 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations. 
        In this case, you should exit the algorithm. A simple way to check 
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges 
        very fast and policy evaluation should end upon convergence. A way to check 
        if policy evaluation has converged is to check whether the max over (or sum of) 
        L1 or L2 norm between the values before and after an iteration is small enough. 
        For the Grid World environment, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy; 
        here you can update the visualization of value by simply calling logger.log(i, v).
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model
    
    v = [0.0] * NUM_STATES
    pi = [random.randint(0, NUM_ACTIONS-1)] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

### Please finish the code below ##############################################
###############################################################################
    #Repeat for max number of policy iterations with a break if policy converges
    for policyIterationStep in range(1, max_iterations):

        #######Policy Evaluation######
        #Repeat for max number of policy iterations
        for policyEvaluationStep in range(1, max_iterations):
            # Set delta to 0
            delta = 0

            #For all states
            for state in range(NUM_STATES):
                #Set value placeholder to 0
                value = 0
                #Get transition model output
                for (probability, newState, reward, terminal) in TRANSITION_MODEL[state][pi[state]]:
                    #Perform Bellman update
                    value += probability * (reward + (gamma * v[newState]))
                #Set delta to find max
                delta = max(delta, value-v[state])
                #Update v with the new value
                v[state] = value
            
            #Update logger every other iteration for speed
            if policyEvaluationStep % 2:
                logger.log(policyEvaluationStep, v)
            #If v converged (i.e there was no change within tolerance) break out of policy eval
            if delta < 1e-4 and delta > -1e-4:
                break
        
        ######Policy Improvement######
        policyStable = True
        #For each state S in observation space
        for state in range(NUM_STATES):
            #Take down old action
            oldAction = pi[state]

            #Use arbitrary low value
            maxActionValue = -10000

            #For all actions
            for action in range(NUM_ACTIONS):
                #Set value to 0
                newVal = 0
                #Get transition model output
                for (probability, newState, reward, terminal) in TRANSITION_MODEL[state][action]:
                    #Find best action by looking ahead and using action, but refer to policy for future
                    newVal += probability * (reward + (gamma * v[newState]))
                    #If action is better than the current 
                    if newVal > maxActionValue:
                        # print("Update maxActionVal to ", newVal)
                        maxActionValue = newVal
                        # print("Update policy")
                        pi[state] = action
    
            #Check to see if policy has changed
            if oldAction != pi[state]:
                policyStable = False
        
        #Update the logger
        logger.log(policyIterationStep,v,pi)

        #If policy has converged (i.e policy at each state hasn't changed from the previous iteration)
        if policyStable:
            break
                


###############################################################################
    return pi


def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model as above. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 40-50 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (training episodes) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
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
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
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
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "Q Learning": q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            [10, "s", "s", "s", 1],
            [-10, -10, -10, -10, -10],
        ],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()