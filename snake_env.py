from snake import Snake
from apple import Apple
import pygame
import random 
import gymnasium as gym
import numpy as np

import gymnasium.utils as utils 

from gymnasium.spaces import Discrete, Box


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self,config,  **kwargs):
        utils.EzPickle.__init__(self, config, **kwargs)

        self.screen_width = config.get('screen_width', 300)
        self.screen_height = config.get('screen_height', 300)
        self.block_size =  config.get('block_size', 20)
        self.snake = Snake(self.screen_width, self.screen_height, self.block_size)
        self.apple = self.generate_apple()
        self.latest_distance = 1
        self.max_steps = config.get('max_steps', self.screen_width * self.screen_height // self.block_size**2)
        if(config.get('render_mode',"rgb_array") == "human"):
            pygame.init()
            self.screen =  pygame.display.set_mode((self.screen_width, self.screen_height))
        #self.observation_space = Box(low=0, high=1, shape=(self.screen_height//self.block_size*self.screen_width//self.block_size* 3,), dtype=np.uint8)
        self.observation_space = Box(low = 0, high = 1, shape = (self.screen_height//self.block_size, self.screen_width//self.block_size, 3,), dtype = np.float32)
        self.action_space = Box(low = 0, high = 1, shape = (4,), dtype = np.float32)
        self.hunger = 0
        self.steps = 0
        self.reset()

    def normalized_distance(self, a, b):    
        return np.linalg.norm(np.array(a) - np.array(b)) / np.sqrt(self.screen_width**2 + self.screen_height**2)
    

    def compute_reward(self, action):
        rewards = {
            "apple": 1 if self.snake.head == self.apple.position else 0,
            "death": -1 if self.snake.head in self.snake.body[:-1] or not self.in_grid_bounds(self.snake.head) else 0,
            # "getting closer": 1 if self.latest_distance > self.normalized_distance(self.snake.head, self.apple.position) else -2,

        }
        return( rewards["apple"] + rewards["death"] ) 
        # self.latest_distance = self.normalized_distance(self.snake.head, self.apple.position)
        # self.reward =( rewards["getting closer"] + rewards["apple"] + rewards["death"] ) / len(rewards)
        # self.reward = np.clip(self.reward,0,1)


    def reset(self, iteration=0, seed=None, options=None):
        # This function resets the game state and returns the initial observation
        # of the game state.
        np.random.seed(seed)    
        # Initialize the snake and apple
        self.snake = Snake(self.screen_width, self.screen_height, self.block_size)
        start = (np.random.randint(1,self.screen_width//self.block_size - 1 ) , np.random.randint(1,self.screen_height//self.block_size - 1 ))

        self.snake.head = (start[0] * self.block_size, start[1] * self.block_size)
        self.snake.body = [self.snake.head]
        self.latest_distance = 1 
        self.snake.direction = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        self.snake.grow = False
        self.apple = self.generate_apple()
        self.score = 0
        self.done = False
        self.reward = 0
        # Return the initial observation of the game state
        return self._get_obs(), {}


    
    def step(self, action):
        # Change snake direction
        self.snake.change_direction(action)
        # Move snake
        self.snake.move()
        self.compute_reward(action)
        
        self.hunger += 1

        if self.snake.head == self.apple.position:
            self.hunger = 0
            self.score += 1
            self.snake.grow = True
            self.apple = self.generate_apple()

        # Check if snake collides with wall
        if self.snake.head[0] < 0 or self.snake.head[0] >= self.screen_width or self.snake.head[1] < 0 or self.snake.head[1] >= self.screen_height:
            self.done = True

        # Check if snake collides with body
        if self.snake.head in self.snake.body[:-1]:
            self.done = True    

        if self.steps > self.max_steps:
            self.done = True

        return self._get_obs(),self.reward, self.done, False, {}
    
    # Make a random apple
    def generate_apple(self):
        # Make a random x and y location
        x = random.randint(0, (self.screen_width - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.screen_height - self.block_size) // self.block_size) * self.block_size
        # Make an apple with those x and y values

        # Check if the apple is in the snake's body
        # If it is, generate a new apple
        while (x,y) in self.snake.body or (x,y) == self.snake.head:
            x = random.randint(0, (self.screen_width - self.block_size) // self.block_size) * self.block_size
            y = random.randint(0, (self.screen_height - self.block_size) // self.block_size) * self.block_size
        return Apple(x, y, self.block_size)
    
    def render(self, mode = "rgb_array"):
        if(mode == "rgb_array"):
            image = np.zeros((self.screen_height, self.screen_width, 3),dtype=np.uint8)
            #make the image white 
            image[:,:,:] = [255, 255, 255]
            #red for the apple 
            image[self.apple.position[1]:self.apple.position[1]+self.block_size, self.apple.position[0]:self.apple.position[0]+self.block_size, :] = [255, 0, 0]

            #green for the snake
            for pos in self.snake.body:
                image[pos[1]:pos[1]+self.block_size, pos[0]: pos[0]+self.block_size, :] = [0, 255, 0]
            #blue for the head
            image[self.snake.head[1]: self.snake.head[1]+self.block_size, self.snake.head[0]:self.snake.head[0]+self.block_size, :] = [0, 0, 255]
            return image
        else:    
            # Fill the screen with white background
            self.screen.fill((255, 255, 255))
            # Draw the snake on the screen
            self.snake.draw(self.screen)
            # Draw the apple on the screen
            self.apple.draw(self.screen)
            # Update the screen to show the changes
            pygame.display.update()
            # Wait 100 milliseconds
            pygame.time.delay(100)

    def in_grid_bounds(self, pos):
        return 0 <= pos[0] < self.screen_width and 0 <= pos[1] < self.screen_height
    
    def _get_obs(self):
        obs = np.zeros((self.screen_height//self.block_size, self.screen_width//self.block_size, 3), dtype=np.uint8)
        obs[self.apple.position[1]//self.block_size, self.apple.position[0]//self.block_size, 0] = 1
        
        for pos in self.snake.body[:-1]:
            if(self.in_grid_bounds(pos)):
                obs[pos[1]//self.block_size, pos[0]//self.block_size, 1] = 1
        if(self.in_grid_bounds(self.snake.head)):
            obs[self.snake.head[1]//self.block_size, self.snake.head[0]//self.block_size, 2] = 1
        return obs
    
