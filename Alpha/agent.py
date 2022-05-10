import keras
keras.__version__


from keras.layers import Dense, Activation, Conv2D, Input
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0 #mem counter 
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape)) # I think this is the previous step memory
        self.new_state_memory = np.zeros((self.mem_size, input_shape)) #And this is the memory of the most recent step 
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1-int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_cntr += 1
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
        # Instead  of input_dims we will set the shape as [84,84]
        # Input(fc1_dims, input_shape=(input_dims,  )),

        # Let's temporarily set the shape to None
        Input(input_shape=([200, 300, 4]  )),
        # Input( shape=None,input_shape=(input_dims  )),
        # The filter is the number of neurons in the first conv layer
        Conv2D(filter=128),
        Conv2D(filter=256),
        Conv2D(filter=512),
        # If there will be errors persisting, relu activation will be added
        # Conv2D('relu', filter=128),

        # Later we might uncomment this part, however the paper hasn't mentioned anything regarding this type of syntax
        # Dense(fc2_dims),
        # Activation('relu'),
        Dense(n_actions) 
        # n_actions will be 3 (buy, sell, hold)
    ])
    model.compile(optimizer=Adam(lr=lr), loss='mse') #lr is the learning rate
    
    return model
class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=0.996, epsilon_end=0.01, mem_size=1000, fname='Model_Alpha.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 84,84)
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state,action,reward,new_state,done)
    def choose_action(self,state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action
        # Since here I am quite sleepy so i might not make the best code choices ! TODO Review the code

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)
        
        q_eval = self.q_eval.predict(state)
        
        q_next = self.q_eval.predict(new_state)
        
        q_target = q_eval.copy()
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        q_target[batch_index, action_indices] = reward + \
                        self.gamma*np.max(q_next, axis=1 )*done
        _ = self.q_eval.fit(state, q_target, verbose=0)
    
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > \
                        self.epsilon_min else self.epsilon_min

    def save_model(self):
        self.q_eval.save(self.model_file)
        
    def load_model(self):
        self.q_eval = load_model(self.model_file)
