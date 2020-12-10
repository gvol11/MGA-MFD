import copy
import sys
import time
import os
import pygame
import numpy as np
from PIL import Image

from environment.gameGrid import GameGrid
import math
import random
from core.zombie import Zombie
import torchvision.transforms as T
from runnable_scripts.Utils import get_config, plot_progress
from itertools import count
import torch
from torch.utils.tensorboard import SummaryWriter

def update_variables():
    MAX_HIT_POINTS = int(get_config("MainInfo")['max_hit_points'])
    MAX_ANGLE = int(get_config("MainInfo")['max_angle']) * math.pi / 180
    MAX_VELOCITY = int(get_config("MainInfo")['max_velocity'])
    BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
    BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])
    return MAX_HIT_POINTS, MAX_ANGLE, MAX_VELOCITY, BOARD_WIDTH, BOARD_HEIGHT


class Game:
    MAX_HIT_POINTS = int(get_config("MainInfo")['max_hit_points'])
    MAX_ANGLE = int(get_config("MainInfo")['max_angle']) * math.pi / 180
    MAX_VELOCITY = int(get_config("MainInfo")['max_velocity'])
    BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
    BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])

    def __init__(self, device, agent_zombie, agent_light):
        Game.MAX_HIT_POINTS, Game.MAX_ANGLE, Game.MAX_VELOCITY, Game.BOARD_WIDTH, Game.BOARD_HEIGHT = update_variables()
        main_info = get_config("MainInfo")
        self.grid = GameGrid()
        self.light_size = int(main_info['light_size'])
        self.max_angle = int(main_info['max_angle'])
        self.start_positions = self.calculate_start_positions()
        if len(self.start_positions) < 2:
            print("The angle is too wide!")
            sys.exit()
        # set interactive mode
        self.interactive_mode = main_info.getboolean('interactive_mode')
        if self.interactive_mode:
            pygame.init()
            pygame.display.set_caption('pickleking')
            self.display_width = int(main_info['display_width'])
            self.display_height = int(main_info['display_height'])
            self.game_display = pygame.display.set_mode((self.display_width, self.display_height))
            self.zombie_image, self.light_image, self.grid_image = self.set_up()
            self.clock = pygame.time.Clock()
        else:
            os.environ["SDL_VIDEODRIVER"] = "dummy"  # not really necessary, here to make sure nothing will pop-up
        # set our agents
        self.agent_zombie = agent_zombie(device, 'zombie')
        self.agent_light = agent_light(device, 'light')
        # load main info
        self.steps_per_episodes = int(main_info['zombies_per_episode']) + int(main_info['board_width']) - 1
        self.zombies_per_episode = int(main_info['zombies_per_episode'])
        self.check_point = int(main_info['check_point'])
        self.total_episodes = int(main_info['num_train_episodes']) + int(main_info['num_test_episodes'])
        # other fields
        self.max_hit_points = Game.MAX_HIT_POINTS
        self.current_time = 0
        self.alive_zombies = []  # list of the currently alive zombies
        self.all_zombies = []  # list of all zombies (from all time)
        self.max_velocity = int(main_info['max_velocity'])
        self.dt = int(main_info['dt'])
        self.device = device
        self.current_screen = None
        self.done = False
        self.writer = SummaryWriter(log_dir= '../runs')

    def calculate_start_positions(self):
        zombie_home_length = int(self.grid.get_height() - 2 * self.grid.get_width() * math.tan(self.max_angle * math.pi / 180))
        zombie_home_start_pos = int(self.grid.get_height() - zombie_home_length - self.grid.get_width() * math.tan(self.max_angle * math.pi / 180))  # m-n-b
        return np.multiply(list(range(zombie_home_start_pos, zombie_home_start_pos + zombie_home_length)), self.grid.get_width())

    def reset(self):
        self.current_time = 0
        Zombie.reset_id()
        self.alive_zombies = []  # list of the currently alive zombies
        self.all_zombies = []  # list of all zombies (from all time)
        self.current_screen = None
        self.agent_light.reset()
        self.agent_zombie.reset()

    def play_zero_sum_game(self, path):
        episodes_dict = {'episode_rewards': [], 'episode_durations': []}
        steps_dict_light = {'epsilon': [], 'action': [], 'step': []}
        steps_dict_zombie = {'epsilon': [], 'action': [], 'step': []}

        for episode in range(self.total_episodes):
            self.reset()
            state_zombie, state_light = self.get_state()
            zombie_master_reward = 0
            episode_start_time = time.time()
            for time_step in count():
                action_zombie, rate, current_step = self.agent_zombie.select_action(state_zombie,self.alive_zombies,self.writer)
                action_light, rate, current_step = self.agent_light.select_action(state_light,self.alive_zombies,self.writer)

                # update dict
                steps_dict_light['epsilon'].append(rate)
                steps_dict_light['action'].append(int(action_light // self.grid.get_width()))
                steps_dict_light['step'].append(time_step)
                steps_dict_zombie['epsilon'].append(rate)
                steps_dict_zombie['action'].append(int(action_zombie))
                steps_dict_zombie['step'].append(time_step)

                reward = self.apply_actions(action_zombie, action_light)
                if reward > 0:
                    zombie_master_reward += reward
                next_state_zombie, next_state_light = self.get_state()

                self.agent_zombie.learn(state_zombie.unsqueeze(0), action_zombie, next_state_zombie.unsqueeze(0), reward,self.writer)
                self.agent_light.learn(state_light.unsqueeze(0), action_light, next_state_light.unsqueeze(0), reward * -1,self.writer)  # agent_light gets the opposite

                state_zombie, state_light = next_state_zombie, next_state_light

                if self.done:  # if the episode is done, store it's reward and plot the moving average
                    episodes_dict['episode_rewards'].append(zombie_master_reward)
                    episodes_dict['episode_durations'].append(time.time() - episode_start_time)
                    break

            # plotting the moving average
            if episode % self.check_point == 0:
                plot_progress(path, episodes_dict, self.check_point)

        plot_progress(path, episodes_dict, self.check_point)

        return episodes_dict, steps_dict_light, steps_dict_zombie

    def action_space(self):
        light_action_space = self.grid.get_height() * self.grid.get_width()
        zombie_action_space = len(self.start_positions)
        return light_action_space, zombie_action_space

    def apply_actions(self, zombie_action, light_action):
        """
        This method steps the game forward one step and
        shoots a bubble at the given angle.
        Parameters
        ----------
        zombie_action : int
            The action is an angle between 0 and 180 degrees, that
            decides the direction of the bubble.
        light_action
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing the
                state of the environment.
            reward (float) :
                amount of reward achieved by the previous action.
            episode_over (bool) :
                whether it's time to reset the environment again.
        """
        self.current_time += 1
        # update display in case of interactive mode
        if self.interactive_mode:
            self.update(light_action)

        # add new zombie
        if len(self.all_zombies) < self.zombies_per_episode:
            new_zombie = Game.create_zombie(zombie_action)
            self.alive_zombies.append(new_zombie)
            self.all_zombies.append(new_zombie)

        # move all zombies one step and calc reward
        reward, self.alive_zombies = Game.calc_reward_and_move_zombies(self.alive_zombies, light_action)

        self.done = self.current_time > self.steps_per_episodes  # TODO - maybe pick another terminal condition of the game and assign it to done (as True/False)
        return reward

    @staticmethod
    def calc_reward_and_move_zombies(alive_zombies, light_action):
        """
        moving all zombies while aggregating and outputting current reward
        :return all alive zombies (haven't step out of the grid)
        """
        # temp list for later be equal to self.alive_zombies list, it's here just for the for loop (NECESSARY!)
        new_alive_zombies = list(copy.deepcopy(alive_zombies))
        reward = 0
        indices_to_keep = list(range(len(new_alive_zombies)))
        for index, zombie in enumerate(new_alive_zombies):
            zombie.move(light_action)
            if 0 >= zombie.y or zombie.y >= Game.BOARD_HEIGHT:
                indices_to_keep.remove(index)
            elif zombie.x >= Game.BOARD_WIDTH:
                if Game.keep_alive(zombie.hit_points):  # decide whether to keep the zombie alive, if so, give the zombie master reward
                    reward += 1
                else:
                    reward -= 1
                indices_to_keep.remove(index)  # deleting a zombie that reached the border
        return reward, list(np.array(new_alive_zombies)[indices_to_keep])

    @staticmethod
    def keep_alive(h):
        if h >= Game.MAX_HIT_POINTS:  # if the zombie sustained a lot of damaged
            return False
        else:  # else decide by the sine function -> if the result is greater than 0.5 -> keep alive, else -> kill it (no reward for the zombie master)
            """
            the idea is: if the hit points is close to 3 then the result is close to 1 ->
             -> there is small chance for keeping him alive and therefor rewarding the zombie with positive reward
             For example, if zombie hit points is 3 - > the result is 1 -> always return False (the random will never be greater than 1)
            in the past sin(h * pi / 2 * self.max_hit_points) < random.random()
            """
            #return np.power(h / Game.MAX_HIT_POINTS, 1 / 3) < random.random()
            return True

    def get_state(self):
        zombie_grid = self.grid.get_values()
        zombie_grid = zombie_grid.astype(np.float32)
        zombie_grid.fill(0)
        health_grid = copy.deepcopy(zombie_grid)
        for i in self.alive_zombies:
            zombie_grid[int(i.y), int(i.x)] = 1
            health_grid[int(i.y), int(i.x)] = i.hit_points
        return torch.from_numpy(zombie_grid).flatten(), torch.from_numpy(np.concatenate((zombie_grid, health_grid))).flatten()

    def get_pygame_window(self):
        return pygame.surfarray.array3d(pygame.display.get_surface())

    @staticmethod
    def create_zombie(position):
        if Game.MAX_ANGLE == 0:
            angle = Game.MAX_ANGLE
        else:
            angle = random.uniform(-Game.MAX_ANGLE, Game.MAX_ANGLE)
        return Zombie(angle, Game.MAX_VELOCITY, position)

    def set_up(self):
        # create the gameUtils directory if doesn't exist
        path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "gameUtils")
        if not os.path.exists(path):
            os.mkdir(path)
            os.chmod(path, 777)
        # get images
        zombie_image = Image.open(os.path.join(path, 'zombie.png'))
        light_image = Image.open(os.path.join(path, 'light.png'))
        # resize (light_image is doubled for 2x2 cells)
        zombie_image = zombie_image.resize((int(self.display_width / self.grid.get_width()), int(self.display_height / self.grid.get_height())), 0)
        light_image = light_image.resize(
            (int(self.display_width / self.grid.get_width()) * self.light_size, int(self.display_height / self.grid.get_height()) * self.light_size), 0)
        # save
        zombie_image.save(os.path.join(path, 'zombie_image.png'))
        light_image.save(os.path.join(path, 'light_image.png'))
        # draw and save the grid
        self.draw_grid()
        # return the images in the pygame format
        return pygame.image.load(os.path.join(path, 'zombie_image.PNG')), pygame.image.load(os.path.join(path, 'light_image.PNG')), pygame.image.load(
            os.path.join(path, 'grid.jpeg'))

    def update(self, light_action):
        event = pygame.event.get()
        self.game_display.blit(self.grid_image, (0, 0))
        x_adjustment = int(self.display_width / self.grid.get_width())
        y_adjustment = int(self.display_height / self.grid.get_height())
        self.game_display.blit(self.light_image, (int(np.mod(light_action, self.grid.get_width()) * x_adjustment),
                                                  int(light_action / self.grid.get_width()) * y_adjustment))
        for z in self.alive_zombies:
            self.game_display.blit(self.zombie_image,
                                   (z.x * x_adjustment, z.y * y_adjustment))
        pygame.display.update()  # better than pygame.display.flip because it can update by param, and not the whole window
        self.clock.tick(30)  # the number of frames per second

    def draw_grid(self):
        x_size = self.display_width / self.grid.get_width()  # x size of the grid block
        y_size = self.display_height / self.grid.get_height()  # y size of the grid block
        for x in range(self.display_width):
            for y in range(self.display_height):
                rect = pygame.Rect(x * x_size, y * y_size,
                                   x_size, y_size)
                pygame.draw.rect(self.game_display, (255, 255, 255), rect, 1)
        # draw the start line
        y_adjustment = int(self.display_height / self.grid.get_height())
        pygame.draw.rect(self.game_display, (0, 200, 50), [0, int((min(self.start_positions))) / self.grid.get_width() * y_adjustment, 10,
                                                           int((max(self.start_positions) + np.diff(self.start_positions)[0] - min(
                                                               self.start_positions))) / self.grid.get_width() * y_adjustment])

        path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "gameUtils")
        pygame.image.save(self.game_display, os.path.join(path, 'grid.jpeg'))

    def end_game(self):
        pygame.quit()
        quit()

    def just_starting(self):
        # current screen is set to none in the beginning and in the end of an episode
        return self.current_screen is None

    def get_state_old(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1

    def get_processed_screen(self):
        screen = self.get_pygame_window().transpose((2, 0, 1))  # PyTorch expects CHW
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]

        # Strip off top and bottom
        top = int(screen_height * 0)
        bottom = int(screen_height * 1)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            , T.Resize((60, 30))
            , T.ToTensor()
        ])

        return resize(screen).unsqueeze(0).to(self.device)  # add a batch dimension (BCHW)