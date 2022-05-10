import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import random
import matplotlib.image as mpimg
import glob

# Let's do it without our previous price so that we always immidiately get a return for our action 

class CandleStickEnv(Env):
    def __init__(self,):
        # Actions we can take: Buy, Sell, Hold
        self.action_space = Discrete(3)
        # Here the observation space will be created

        # This is the action that is currently held
        self.state = 0
        self.day= 1
        self.previous_price = 0
        self.current_stock = 'ASIANPAINT'

        self.profit = 0

    def step(self, action):

        # firsly we get the observation



        # 0 is hold 1 is buy and 2 is sell
        # Now we will do an if statement 
        # We will return info as well
        # if action == 0:

        # if(action == 0 and self.state == 0):
        #     return 0

        # answer = action + "BOY"
        # return(answer)


        # img = mpimg.imread('images/TrainingDS/1,3216.3,ASIANPAINT.png')
        path_str = f"images/attempt2/{self.day},*"
        img_path = glob.glob(path_str)
        img = mpimg.imread(img_path[0])

        path_str_2 = f"images/attempt2/{self.day+1},*"
        img_path_2 = glob.glob(path_str_2)
        array_splt_2 = img_path_2[0].split(",")

        array_splt = img_path[0].split(",")
        

        next_stock = (array_splt_2[2].split("."))[0]

        if(next_stock != self.current_stock):
            # return  profit and everythng as zero except of the observation
            reward = 0
            info  = f"The EQUITY is changing to: {next_stock}"
            self.current_stock = next_stock
        else:
            current_price = float(array_splt[1])
            next_price = float(array_splt_2[1])
            if(action == 0):
                reward = 0
                info  = "No action was taken"
            elif(action == 1):
                reward = next_price - current_price
                info  = f"The stock was BOUGHT, profit:{round(reward,2)}"
            elif(action == 2):
                reward = current_price - next_price
                info  = f"The stock was SOLD, profit:{round(reward,2)}"
            else:
                reward = 0
                info  = "An ERROR OCCURED"
        
        self.profit = self.profit + reward
        info = f"{info} ----- Total Revenue: {round(self.profit, 2)}"
    
        self.day = self.day +1

        return img,reward, info


    def reset(self):
        
        # img = mpimg.imread('images/TrainingDS/1,3216.3,ASIANPAINT.png')
        path_str = f"images/attempt2/{self.day},*"
        img_path = glob.glob(path_str)
        img = mpimg.imread(img_path[0])
        # now we have the image and the path

        array_splt = img_path[0].split(",")
        self.current_price = float(array_splt[1])

        self.current_stock = (array_splt[2].split("."))[0]
        
        # self.previous_price = self.current_price
        self.day = 1

        # return self.current_price
        self.profit = 0
        return img


