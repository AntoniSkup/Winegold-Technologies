from agent import Agent 
from environment import CandleStickEnv





import numpy as np
import gym

if __name__ == '__main__':
    env = CandleStickEnv()
    n_days = 100
    score = 0
    observation = env.reset()
    


    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.005, input_dims=[200,300,4] , n_actions=3, mem_size=1000, batch_size=64, epsilon_end=0.01)

    scores = []
    eps_history = []

    for i in range(n_days):
        action = agent.choose_action(observation)
        observation_,reward, info = env.step(action)
        print(f"Day {i}--{info}")
        scores.append(reward)
        agent.remember(observation, action, reward, observation_)
        observation = observation_
        agent.learn()

        eps_history.append(agent.epsilon)

        if i % 10 == 0 and i > 0 :
            agent.save_model()

    