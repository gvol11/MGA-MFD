import copy
import math
import os
import random
import multiprocessing as mp
from environment.game import Game
from agents.agent import Agent
from strategies.epsilonGreedyStrategy import EpsilonGreedyStrategy
from runnable_scripts.Utils import get_config
from core.node import Node
import numpy as np
from core.costly_simulation import CostlySimulation


def update_variables():
    MAX_HIT_POINTS = int(get_config("MainInfo")['max_hit_points'])
    MAX_ANGLE = int(get_config("MainInfo")['max_angle'])
    MAX_VELOCITY = int(get_config("MainInfo")['max_velocity'])
    BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
    BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])
    C = float(get_config("TreeAgentInfo")['exploration_const'])
    return MAX_HIT_POINTS, MAX_ANGLE, MAX_VELOCITY, BOARD_WIDTH, BOARD_HEIGHT, C


class BasicMCTSAgent(Agent):
    MAX_HIT_POINTS = int(get_config("MainInfo")['max_hit_points'])
    MAX_ANGLE = int(get_config("MainInfo")['max_angle'])
    MAX_VELOCITY = int(get_config("MainInfo")['max_velocity'])
    BOARD_WIDTH = int(get_config("MainInfo")['board_width'])
    BOARD_HEIGHT = int(get_config("MainInfo")['board_height'])
    C = float(get_config("TreeAgentInfo")['exploration_const'])

    def __init__(self, device, agent_type):
        BasicMCTSAgent.MAX_HIT_POINTS, BasicMCTSAgent.MAX_ANGLE, BasicMCTSAgent.MAX_VELOCITY, BasicMCTSAgent.BOARD_WIDTH, BasicMCTSAgent.BOARD_HEIGHT, BasicMCTSAgent.C = update_variables()
        super().__init__(EpsilonGreedyStrategy(), agent_type)
        self.possible_actions = list(range(Game.BOARD_HEIGHT)) if self.agent_type == 'zombie' else list(range(Game.BOARD_HEIGHT * Game.BOARD_WIDTH))
        self.root = Node([], self.possible_actions)
        self.temporary_root = self.root  # TODO - change its name to something like: real world state-node
        self.current_step = 0
        self.simulation_reward = 0
        self.simulation_num = int(get_config("TreeAgentInfo")['simulation_num'])  # number of simulations in the simulation phase
        self.simulation_depth = int(get_config("TreeAgentInfo")['simulation_depth'])  # number of times to expand a node in single simulation
        self.episode_reward = 0
        self.tree_depth = 0

        self.pool = mp.Pool(mp.cpu_count())

        main_info = get_config('MainInfo')
        self.steps_per_episodes = int(main_info['zombies_per_episode']) + int(main_info['board_width'])
        self.total_episodes = int(main_info['num_train_episodes']) + int(main_info['num_test_episodes'])

    def select_action(self, state,alive_zombies):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        self.current_step += 1

        # selection phase
        selected_child = self.selection()
        assert selected_child.num_children == 0 or selected_child.is_terminal

        # expansion phase, here we selecting the action from which we will simulate the selected_child play-out
        # keep in mind that in this phase we expand a node that is NOT the temporary root, the expansion action doesn't relate to the real action we are taking
        # action = self.expansion_all_children(selected_child)
        if selected_child == self.root:
            # if the selected child is root, expand all its children
            expanded_child = self.expansion_all_children(selected_child)
        elif selected_child.parent is not None and selected_child.parent.num_children != len(self.possible_actions):
            # if the selected child is missing a brother (we managed to choose him thanks to some real action), expand all its brothers and choose one
            expanded_child = self.expansion_all_children(selected_child.parent)
        elif selected_child.visits == 0:
            # if we never visited that node, start roll-out from there
            expanded_child = selected_child
        else:
            # in case the node is a leaf but we already been there
            expanded_child = self.expansion_all_children(selected_child)

        assert expanded_child.num_children == 0
        assert selected_child.num_children == 0 or selected_child.num_children == len(self.possible_actions)

        # simulation phase
        self.simulation(expanded_child)

        # select next action
        action = self.select_expansion_action(self.temporary_root, self.possible_actions)
        self.expansion_all_children(self.temporary_root)
        self.temporary_root = self.temporary_root.children[action]
        assert self.temporary_root.num_children == len(self.possible_actions) or self.temporary_root.num_children == 0
        # self.PrintTree()

        # when the game ends - close the pool to avoid memory explosion
        if self.current_step == self.total_episodes * self.steps_per_episodes:
            self.pool.close()
            self.pool.join()

        return action, rate, self.current_step

    def learn(self, _, action, __, reward):
        # back-propagation phase, start back-propagating from the current real world node
        # self.episode_reward += reward
        # self.back_propagation(self.temporary_root, reward, self.root)
        pass

    def selection(self):
        """
        The selection Phase in the MCTS algorithm.
        selects leaf by following the UCT algorithm
        :return:
        """
        selected_child = self.temporary_root

        # Check if child nodes exist.
        if selected_child.num_children > 0:
            has_child = True
        else:
            has_child = False

        while has_child:
            # selecting the best child unless there is unexpanded child in the way - select_child method is required!
            selected_child = self.select_child(selected_child)
            if selected_child.num_children == 0 or selected_child.is_terminal:
                has_child = False

        return selected_child

    def select_child(self, node: Node) -> Node:
        """
        Given a node, selects a random unvisited child node.
        Or if all children are visited, selects the node with greatest UTC value.
         @note: we must start the selection from here - imagine that a child was expanded, immediately we expanded all its brothers too.
         in the next turn we might want to start simulating from one of its brothers instead of picking always him with the 'select_best_child'
         (after we evaluated one of the brothers with 'eval_utc' method, that brother would always be selected via 'select_best_child' method)
        :param node: node from which to select child node from.
        :return: The selected child
        """
        if node.num_children == 0:
            return node

        # check if 'node' has any unexpanded nodes - which is any None value in children dictionary OR there is a child but it's simulated
        not_visited_actions = []
        assert node.num_children == 0 or node.num_children == len(self.possible_actions)
        for action, child in node.children.items():
            # search for children that never rolled out (simulation started from them)
            if child.visits == 0:
                not_visited_actions.append(action)
        # chosen child from one of the unexpanded children - if there are any
        if len(not_visited_actions) > 0:
            action = random.sample(not_visited_actions, 1)[0]
            return node.children[action]

        return BasicMCTSAgent.select_best_child(node)

    @staticmethod
    def select_best_child(node):
        """
        Selects the best child of a node
        :param node: Node to select one of its children
        :return: highest UCT valued child
        """
        selected_child = node

        if node.num_children == 0:
            return node

        max_weight = 0.0
        possible_children = []
        for child in list(filter(None, node.children.values())):
            weight = child.uct
            if len(possible_children) == 0:
                possible_children.append(child)
                max_weight = weight
            elif weight == max_weight:
                possible_children.append(child)
            elif weight > max_weight:
                possible_children = [child]
                max_weight = weight
        if len(possible_children) > 0:
            selected_child = random.sample(possible_children, 1)[0]
        return selected_child

    def expansion_all_children(self, leaf):
        self.eval_children(leaf, self.possible_actions)
        return random.sample(list(leaf.children.values()), 1)[0]

    def expansion_one_child(self, leaf):
        action = self.select_expansion_action(leaf, self.possible_actions)
        self.eval_children(leaf, [action])
        return action

    def eval_children(self, node, actions):
        """
        Evaluates all the possible children states given a node state
        :param node: node from which to evaluate children.
        :param actions: list of all possible actions to choose from
        :return: returns the possible children Nodes
        """
        assert node.num_children == len(self.possible_actions) or node.num_children == 0
        if node.num_children == 0:
            for action in actions:
                _, alive_zombies = BasicMCTSAgent.simulate_action(node.state, self.agent_type, action)
                node.add_child(alive_zombies, action)

        return node.children

    def select_expansion_action(self, node, possible_actions):
        """
        Wisely selects a child node.
        :param node: the selected node to expand child from
        :param possible_actions: list of all possible actions to choose from
        :return: the selected action
        """
        selected_child = self.select_best_child(node)
        assert selected_child is not None

        selected_action = None
        if selected_child == node:
            selected_action = random.sample(self.possible_actions, 1)[0]
        else:
            for key, value in node.children.items():
                if value == selected_child:
                    selected_action = key
        assert selected_action is not None

        return selected_action

    @staticmethod
    def select_simulation_action(alive_zombies, possible_actions):
        # Randomly selects a child node.
        i = random.sample(possible_actions, 1)[0]
        return i

    def simulation(self, selected_child):
        """
        Simulating states from previous states and actions
        This phase happens right after we've chose the expansion, and from the selected child with action
        :param selected_child: node from which to perform simulation.
        :return:
        """
        # Perform simulation.
        list_of_objects = []
        simulation_state = selected_child.state

        for _ in range(self.simulation_num):
            obj = CostlySimulation(self.simulation_depth, simulation_state, self.possible_actions, self.agent_type)
            list_of_objects.append(obj)

        list_of_results = self.pool.map(BasicMCTSAgent.worker, ((obj, BasicMCTSAgent.BOARD_HEIGHT, BasicMCTSAgent.BOARD_WIDTH) for obj in list_of_objects))
        assert np.max(list_of_results) <= self.simulation_depth

        average_total_reward = np.average(list_of_results) if self.agent_type == 'zombie' else -1 * np.average(list_of_results)

        # back-prop from the expanded child (the child of the selected node)
        BasicMCTSAgent.back_propagation(selected_child, average_total_reward, self.root)

    @staticmethod
    def worker(arg):
        return arg[0].costly_simulation(arg[1], arg[2])

    @staticmethod
    def simulate_action(alive_zombies, agent_type, action):
        """
        Simulating future states by 'actions' of an agent
        :param alive_zombies: all alive zombies at the real world
        :param agent_type: 'zombie' or 'light' agent
        :param action: array containing all the actions to simulate
        :return: total reward of the simulation
        """
        new_alive_zombies = list(copy.deepcopy(alive_zombies))  # make a copy of all zombies - we do not want to make any act in real world

        # set action and light agents actions
        if agent_type == 'zombie':
            zombie_action = action
            # random sample len(actions) times from light-agent actions-space
            light_action = 0  # np.random.randint(0, BasicMCTSAgent.BOARD_HEIGHT * BasicMCTSAgent.BOARD_WIDTH)
        else:
            light_action = action
            # sample n times from zombie-agent actions-space
            zombie_action = np.random.randint(0, BasicMCTSAgent.BOARD_HEIGHT)

        # simulate and aggregate reward
        total_reward = 0
        new_zombie = Game.create_zombie(zombie_action)
        new_alive_zombies.append(new_zombie)
        reward, final_alive_zombies = Game.calc_reward_and_move_zombies(new_alive_zombies, light_action)
        total_reward += reward

        return total_reward, final_alive_zombies

    @staticmethod
    def back_propagation(node, result, root):
        current_node = node

        # Update node's weight.
        BasicMCTSAgent.eval_utc(current_node, result)

        # keep updating until the desired root
        while current_node.level != root.level:
            # Update parent node's weight.
            current_node = current_node.parent
            BasicMCTSAgent.eval_utc(current_node, result)

    @staticmethod
    def eval_utc(node, result):
        node.wins += result
        node.visits += 1

        node.uct = node.wins / node.visits + BasicMCTSAgent.evaluate_exploration(node)

    @staticmethod
    def evaluate_exploration(node):
        n = node.visits
        if node.parent is None:
            t = node.visits
        else:
            t = node.parent.visits

        # avoid log of 0 with: 't or 1'
        return BasicMCTSAgent.C * np.sqrt(np.log(t or 1) / n)

    @staticmethod
    def has_parent(node):
        if node.parent is None:
            return False
        else:
            return True

    def reset(self):
        # BasicMCTSAgent.back_propagation(self.temporary_root, self.episode_reward)
        self.temporary_root = self.root
        # self.episode_reward = 0
        # if self.agent_type == 'zombie':
        #     self.PrintTree()

    def print_tree(self):
        """
        Prints the tree to file.
        :return:
        """
        f = open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'Tree.txt'), 'w')
        node = self.root
        self.print_node(f, node, "")
        f.close()

    def print_node(self, file, node, indent):
        """
        Prints the tree node and its details to file.
        :param file: file to write into
        :param node: node to print.
        :param indent: Indent character.
        :return:
        """
        file.write(indent)
        file.write("|-")
        indent += "| "

        string = str(node.level) + " ("
        string += "W: " + str(node.wins) + ", N: " + str(node.visits) + ", UCT: " + str(node.uct) + ") \n"
        file.write(string)

        for child in list(filter(None, list(node.children.values()))):
            self.print_node(file, child, indent)
