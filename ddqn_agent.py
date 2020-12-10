from strategies.epsilonGreedyStrategy import EpsilonGreedyStrategy
from core.replayMemory import ReplayMemory
from core.experience import Experience
from core.qValues import QValues
import torch.nn.functional as F
from agents.agent import Agent
import torch.optim as optim
from core.DQN import DQN
import random
import torch
from runnable_scripts.Utils import get_config
import math


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return t1, t2, t3, t4


def create_networks(device, agent_type, possible_actions):
    main_info = get_config('MainInfo')
    h = int(main_info['board_height'])
    w = int(main_info['board_width'])
    # create networks
    neurons_number = h * w if agent_type == 'light' else h * w / 2
    input_size = 2 * h * w if agent_type == 'light' else h * w  # the light agents get extra information
    num_actions = len(possible_actions)
    target_net = DQN(input_size, num_actions, neurons_number).to(device)
    policy_net = DQN(input_size, num_actions, neurons_number).to(device)
    # set up target network as the same weights
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    return num_actions, target_net, policy_net


class DdqnAgent(Agent):

    def __init__(self, device, agent_type):
        super().__init__(strategy=EpsilonGreedyStrategy(), agent_type=agent_type)  # use the 'EpsilonGreedyStrategy' strategy
        # load values from config
        self.LIGHT_SIZE = int(get_config("MainInfo")['light_size'])
        ddqn_info = get_config('DdqnAgentInfo')
        self.batch_size = int(ddqn_info['batch_size'])
        self.gamma = float(ddqn_info['gamma'])
        self.memory_size = int(ddqn_info['memory_size'])
        self.target_update = int(ddqn_info['target_update'])
        self.lr = float(ddqn_info['lr'])
        # init networks
        self.num_actions, self.target_net, self.policy_net = create_networks(device, agent_type, self.possible_actions)
        # other fields
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory()
        self.current_step = 0
        self.device = device

    def select_action(self, state,alive_zombies,writer):
        rate = self.strategy.get_exploration_rate(current_step=self.current_step)
        writer.add_scalar('exploration_rate/train', rate, self.current_step)

        self.current_step += 1
        random_number = random.random()
        if rate > random_number:
            if self.agent_type == 'zombie':
                action = random.randrange(self.num_actions)
            else :
                zombies_num = len(alive_zombies);
                if zombies_num  == 0 :
                    action = random.randrange(self.num_actions)
                else:
                    index = random.randrange(zombies_num)
                    action = int(max(alive_zombies[index].y-math.floor(self.LIGHT_SIZE/2),0)* self.BOARD_WIDTH + max(alive_zombies[index].x-math.floor(self.LIGHT_SIZE/2),0))

            return action, rate, self.current_step  # explore
        else:
            with torch.no_grad():
                # here we are getting the action from one pass along the network. after that we:
                # convert the tensor to data, then move to cpu using then converting to numpy and lastly, wrapping back to tensor
                if self.agent_type == 'zombie' or len(alive_zombies)  == 0 :
                    action = self.policy_net(state).argmax(dim=0).data.cpu().numpy()[0]  # max over rows! (dim=0)
                else:
                    alive_zombie_indexes = [getattr(alive_zombies[i],'id') for i in range(len(alive_zombies))]
                    net_actions = self.policy_net(state).data.cpu()
                    alive_zombie_indexes_actions = net_actions[alive_zombie_indexes]
                    action = alive_zombie_indexes_actions.argmax(dim=0)
                    index = action.numpy()[0]
                    try:
                        action = int((max(alive_zombies[index].y-math.floor(self.LIGHT_SIZE/2),0)) * self.BOARD_WIDTH + (max(alive_zombies[index].x-math.floor(self.LIGHT_SIZE/2),0)))
                    except:
                        print("An exception occurred")
                return action, rate, self.current_step

    def learn(self, state, action, next_state, reward,writer):
        self.memory.push(Experience(state, torch.tensor([action], device=self.device), next_state, torch.tensor([reward], device=self.device)))
        if self.memory.can_provide_sample(self.batch_size):
            experiences = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(self.policy_net, states, actions)
            next_q_values = QValues.get_next(self.target_net, next_states, self.policy_net)
            target_q_values = (next_q_values * self.gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values).to(self.device)
            writer.add_scalar('Loss/train', loss, self.current_step)
            writer.add_histogram('current_q_values/train', current_q_values, self.current_step)
            writer.add_histogram('target_q_values/train', target_q_values, self.current_step)
            writer.add_histogram('rewards/train', rewards, self.current_step)


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.current_step % self.target_update == 0:
            # update the target net to the same weights as the policy net
            self.target_net.load_state_dict(self.policy_net.state_dict())



    def reset(self):
        pass
