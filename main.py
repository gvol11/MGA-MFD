import os
import time
import torch
from numpy import linspace

from configparser import RawConfigParser

from core.node import Node
from core.zombie import Zombie


def main():
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "true"  # cancel py-game display
    from environment.game import Game
    from agents.random_agent import RandomAgent
    from agents.ddqn_agent import DdqnAgent
    from agents.constant_agent import ConstantAgent
    from agents.basic_mcts_agent import BasicMCTSAgent
    from runnable_scripts.Utils import create_dir, ridge_plot, save_ini_file

    # create directory for storing the results
    dir_path = create_dir()

    # create the game with the required agents
    env = Game(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), agent_zombie=RandomAgent, agent_light=DdqnAgent)
#    env = Game(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), agent_zombie=BasicMCTSAgent, agent_light=ConstantAgent)

    # play the game and produce the dictionaries of the results
    episodes_dict, steps_dict_light, steps_dict_zombie = env.play_zero_sum_game(dir_path)

    # save and create results graph
    results_file_name = '/results_' + time.strftime('%d_%m_%Y_%H_%M')
    save_ini_file(dir_path, results_file_name, steps_dict_light, steps_dict_zombie, episodes_dict)
    ridge_plot(dir_path, results_file_name + '.xlsx')

    print('Eliav king')


if __name__ == "__main__":
    temp = 2
    if temp == 1:
        for size in range(2, 10, 2):
            for exploration_rate in linspace(0.0, 4., 21):
                path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'configs', 'config.ini')
                parser = RawConfigParser()
                parser.read(path)
                parser.set('TreeAgentInfo', 'exploration_const', str(exploration_rate))
                parser.set('MainInfo', 'board_height', str(size))
                parser.set('MainInfo', 'board_width', str(size//2))
                parser.set('MainInfo', 'light_size', str(size//2))
                config_file = open(path, 'w')
                parser.write(config_file, space_around_delimiters=True)
                config_file.close()

                # update all variables due to changes in the configuration file
                Zombie.update_variables()
                Node.update_variables()

                main()
    elif temp == 2:
        main()
