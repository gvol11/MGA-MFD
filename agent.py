from abc import abstractmethod
from runnable_scripts.Utils import get_config


class Agent:
    BOARD_WIDTH = int(get_config("MainInfo")['board_width'])

    def __init__(self, strategy, agent_type):
        self.agent_type = agent_type
        self.strategy = strategy

    @abstractmethod
    def select_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def learn(self, state, action, next_state, reward):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
