import pygame
from pygame import font
import random
import math
# import gymnasium as gym
from gymnasium import Env
import numpy as np
import gymnasium.spaces as spaces
from pygame_helpers import *

# Particle push class
class particlePush(Env):
    def __init__(self, action_space_size=16, agent_init=None, num_balls=None, ball_inits=None, ball_goals=None, ball_sizes=None, render_mode='rgb_array'):
        super(particlePush, self).__init__()

        # Initialize render settings
        if render_mode == 'rgb_array':
            self.screen = pygame.display.set_mode((width, height), flags=pygame.HIDDEN)
        elif render_mode == 'human':
            self.screen = pygame.display.set_mode((width, height))
            pygame.font.init()
            self.font = font.SysFont("candara", 35) 
            
        pygame.display.set_caption('Particle Push')
        self.w = width
        self.h = height
        self.render_mode = render_mode

        # Chooses whether to use a dense distance-based reward or a sparse reward indicating success/failure
        self.use_dense_reward = True

        # Static hyperparameters
        self.agent_weight = 1
        self.agent_size = 10
        self.ball_weight = 1e-6
        self.v = 30
        self.reward = 0
        self.reward_threshold = 10
        self.reward_on_success = 1
        self.reward_on_failure = 0
        self.t = 0

        # Maximum number of steps
        self.T = 499
        self.goal_reached = False

        # Observation space is a standardized set of 2D coordinates for the agent, ball, and goal
        self.observation_space = spaces.Box(low=np.array([[-1.,-1.],[-1.,-1.],[-1.,-1.]]), high=np.array([[1.,1.],[1.,1.],[1.,1.]]), shape=(3,2), dtype=np.float32)

        # Action space is a set of simple 2D movements (default 17)
        self.action_space = spaces.Discrete(action_space_size + 1)
        # Uniformly sample action_space_size angles from 0 to 2pi
        self.action_map = {i: [math.cos(2*math.pi*i/action_space_size), math.sin(2*math.pi*i/action_space_size)] for i in range(action_space_size)}
        self.action_map[action_space_size] = [0., 0.]

        # self.action_space = spaces.Box(low=np.array([-1.,-1.]), high=np.array([1.,1.]), shape=(2,), dtype=np.float32)

        # Initialize environment
        self.set_env(agent_init, num_balls, ball_inits, ball_goals, ball_sizes)


    def set_env(self, agent_init=None, num_balls=None, ball_inits=None, ball_goals=None, ball_sizes=None):
        self.agent_init = agent_init if agent_init is not None else np.random.randint(low=self.agent_size, high=(width - self.agent_size), size=(2,))
        self.num_balls = num_balls if num_balls is not None else 1
        self.ball_sizes = ball_sizes if ball_sizes is not None else [30 for _ in range(self.num_balls)]
        self.ball_inits = ball_inits if ball_inits is not None else [np.random.randint(low=self.ball_sizes[i], high=(width - self.ball_sizes[i]), size=(2,)) for i in range(self.num_balls)]
        self.ball_goals = ball_goals if ball_goals is not None else [np.random.randint(low=self.ball_sizes[i], high=(width - self.ball_sizes[i]), size=(2,)) for i in range(self.num_balls)]

    # Places balls at initial locations
    def reset(self, seed=None, options=None):
        self.balls = []
        self.agent = None
        self.reward = 0
        self.t = 0
        self.goal_reached = False
        
        for i in range(self.num_balls):
            particle = Particle(self.ball_inits[i], self.ball_sizes[i], self.ball_weight)
            particle.colour = (100, 100, 255)
            particle.speed = 0
            particle.angle = 0
            self.balls.append(particle)

        # Make the agent
        self.agent = Particle(self.agent_init, self.agent_size, self.agent_weight, name="Agent")
        self.agent.colour = (255, 0, 0)
        self.agent.speed = 0
        self.agent.angle = 0
        
        # Initialize pygame
        self.clock = pygame.time.Clock()
        self.selected_particle = None
        self.running = True

        self.render()

        return self.get_state(), {}

    
    # Wrapper for draw_elements_on_canvas
    def render(self):
        if self.render_mode != 'None':
            vis = self.draw_elements_on_canvas()
            return vis
    
    # End the game
    def close(self):
        pygame.quit()
        quit()

    # Draws the current state of the game on the canvas
    def draw_elements_on_canvas(self):
        self.screen.fill(background_colour)

        for i, particle in enumerate(self.balls):
            particle.display(self.screen)
            pygame.draw.circle(self.screen, (0,255,0), (self.ball_goals[i][0], self.ball_goals[i][1]), 5, 0)
        self.agent.display(self.screen)

        # Render the pygame display window
        if self.render_mode == 'human':
            pygame.display.flip()
            return None
        # Return the rendered image as a numpy array
        elif self.render_mode == 'rgb_array':
            x3 = pygame.surfarray.array3d(self.screen)
            return np.uint8(x3)

    # Returns the current state of the game
    def get_state(self):
        agent_state = np.array([self.agent.x, self.agent.y])
        ball_states = []
        for ball in self.balls:
            ball_states.append([ball.x, ball.y])
        ball_states = np.array(ball_states)

        state = np.zeros((2 * self.num_balls + 1, 2))

        state[0] = agent_state
        state[1:self.num_balls + 1] = ball_states
        state[self.num_balls + 1:] = self.ball_goals
        # Subtract self.w/2 and self.h/2 from each element of state
        state = state - np.array([self.w/2, self.h/2])
        # Divide each element of state by self.w/2 and self.h/2
        state = state / np.array([self.w/2, self.h/2])
        
        state_flattened = state.flatten()
        return state_flattened

    # Returns the reward for the current state of the game
    def get_reward(self):
        info = self.get_info()
        if all(info):
            self.goal_reached = True
            return self.reward_on_success
        return self.reward_on_failure
    
    # Returns whether each ball is on its goal
    def get_info(self):
        on_goal = [False for _ in range(self.num_balls)]
        for i, ball in enumerate(self.balls):
            dist = np.sqrt( (ball.x - self.ball_goals[i][0])**2 + (ball.y - self.ball_goals[i][1])**2 )
            if dist < self.reward_threshold:
                on_goal[i] = True
        return on_goal
    
    # Returns the parameters of the game
    def get_params(self):
        # Params is a dictionary that looks like the following - this dictionary specifies the game environment
        # "num_balls" : Int
        # "ball_sizes" : Int array
        # "ball_inits" : float array (num_balls X 2)
        # "agent_init" : float array (2, )
        # "ball_goals" : float array (num_balls X 2)
        params = {
            "num_balls" : self.num_balls,
            "ball_sizes" : self.ball_sizes,
            "ball_inits" : self.ball_inits,
            "agent_init" : self.agent_init,
            "ball_goals" : self.ball_goals
        }
        return params
    
    def dense_reward(self):
        # Determines reward scaling factor
        max_ball_size = np.max(self.ball_sizes)
        self.min_reward = np.sqrt(2) * (2 * self.w + 2 * self.h - 3*max_ball_size - self.agent_size)

        reward = 0
        # Add a small reward if the agent is closer to a ball than it was at initialization
        ball_rewards = np.zeros(self.num_balls)

        for i, ball in enumerate(self.balls):
            ball_rewards[i] = np.sqrt( (ball.x - self.agent.x)**2 + (ball.y - self.agent.y)**2 )
        # Add the reward for the closest ball
        reward -= np.max(ball_rewards) / 100

        # Add a large reward if any ball is closer to its goal than it was at initialization
        for i, ball in enumerate(self.balls):
            reward -= np.sqrt( (ball.x - self.ball_goals[i][0])**2 + (ball.y - self.ball_goals[i][1])**2 ) / 10

        # Scale so min_reward maps to -1
        reward = reward / self.min_reward

        assert reward <= 0 and reward >= -1, "REWARDBOUNDSERROR: Reward is not in the correct range"

        return reward

            
    # Runs one step of the game
    def step(self, action):
        action_arr = self.action_map[action]
        # action_arr = action
        dx = action_arr[0] * self.v
        dy = action_arr[1] * self.v

        self.agent.angle = 0.5*math.pi + math.atan2(dy, dx)
        self.agent.speed = math.hypot(dx, dy) * 0.1

        self.agent.move()
        self.agent.bounce()

        # Added so that the agent can still propagate collisions with the balls
        self.agent.angle = 0.5*math.pi + math.atan2(dy, dx)
        self.agent.speed = math.hypot(dx, dy) * 0.1

        # Model ball collisions
        for particle in self.balls:
            collide(self.agent, particle)
        for i, particle in enumerate(self.balls):
            particle.move()
            particle.bounce()
            for particle2 in self.balls[i+1:]:
                collide(particle, particle2)

        # Get the reward
        curr_reward = self.get_reward()
        self.reward += curr_reward

        # Check if either end condition is met
        term = True if self.goal_reached else False
        # term = False
        trunc = True if self.t >= self.T else False
        done = term or trunc

        self.clock.tick(200)

        if self.use_dense_reward:
            if term:
                reward = self.dense_reward() + 1
            else:
                reward = self.dense_reward()
        else:
            reward = curr_reward

        self.t += 1

        # return self.get_state(), self.dense_reward(), term, trunc, {}
        return self.get_state(), reward, term, trunc, {}
