#AI for Self Driving Car
#TODO::when to use self.parameter instead of just the paramenter name
#Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action): #nb_action = number of actions
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        """Linear connects the two arguments, can change number of hidden layer 
        nuerons later if wanted """
        self.fc2 = nn.Linear(30, nb_action)
        
    def forward(self, state):
        x = F.relu(self.fc1(state)) #relu = rectifier function
        q_values = self.fc2(x)
        return q_values
    
    
#Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        #if list = ((1,2,3), (4,5,6)), then zip(*list) = ((1,4), (2,3), (5,6))
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
    
# Implementing Deep Q Learning
        
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action) #making the whole outline of the network I believe
        self.memory = ReplayMemory(100000) #can change # in batch if wanted
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) 
        """ ^^^can change "Adam" and see different results
       
        Def: To use torch.optim you have to construct an optimizer object, that 
        will hold the current state and will update the parameters based on the 
        computed gradients. 
        
        lr = learning rate, don't want to make it too large or agent won't have
        time to explore, can change if wanted"""
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #unsqueeze adds dimenason at 0 index
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*7) #T=7
        """ ^^ ex: softmax([1,2,3]) = [0.04, 0.11, 0.85] => softmax([1,2,3]*3) = [0.00, 0.02, 0.98]
        so, the higher the T value, the more pronouced the softmax function will become """
        action = probs.multinomial()
        return action.data[0,0]
        
            
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action).unsqueeze(1).squeeze(1) #TODO: understand this line
        next_outputs = self.model(batch_next_state).detach().max(1)[0] #TODO: understand this line
        target = self.gamma*next_outputs + batch_reward
        td_loss =  F.smooth_l1_loss(outputs, target) #td = temporal difference
        self.otimizer
























    
    
    
    
    
    
    