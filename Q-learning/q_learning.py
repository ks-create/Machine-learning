import sys
import random
import numpy as np
from environment import MountainCar

def q(state, a, w, b):
    return np.dot(np.transpose(state), w[:, a]) + b

def get_state_list(state, indices):
    if (indices == False):
        state_list = np.zeros(2)
        state_list[0] = state[0]
        state_list[1] = state[1]
    else:
        state_list = np.zeros(2048)
        for key in state:
            state_list[int(key)] = 1
    return state_list

class LinearModel:
    def __init__(self, state_size: int, action_size: int, 
    lr: float, indices: bool):
        # indices is True if indices are used as input for one-hot features.
        # Otherwise, use the sparse representation of state as features
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.indices = indices

        self.bias = 0
        self.weight = np.zeros((state_size, action_size))

    def predict(self, state):
        # state: state_list, output  List[float]
        # Given state, makes predictions.
        
        res = np.zeros(3)
        for i in range(len(res)):
            res[i] = q(state, i, self.weight, self.bias)
        
        return res


    def update(self, state, action: int, target):
        # state: Dict[int, int]
        # Given state, action, and target, update weights.
        d = np.zeros((self.state_size, self.action_size))
        d[: ,action] = state
        self.weight -= self.lr * target * d
        self.bias -= self.lr * target

class QLearningAgent:
    def __init__(self, env: MountainCar, mode: str = None, gamma: float = 0.9, 
    lr: float = 0.01, epsilon:float = 0.05):
        self.env = env
        self.mode = mode
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon

        if (mode == "raw"):
            self.linearModel = LinearModel(2, 3, lr, False)
        else:
            self.linearModel = LinearModel(2048, 3, lr, True)
    
    def get_action(self, state):
        # state: Dict[int, int], output int
        # epsilon-greedy strategy.
        # Given state, returns action.
        
        r = random.random()
        if (r < self.epsilon):
            action = random.randint(0,2)
        else:
            action = np.argmax(self.linearModel.predict(state))
        return action

    def train(self, episodes: int, max_iterations: int):
        # output List[float]
        # training function.
        # Train for ’episodes’ iterations, where at most ’max_iterations‘ iterations
        # should be run for each episode. Returns a list of returns.
        reward_output = list()

        for i in range(episodes):
            state = get_state_list((self.env).reset(), self.linearModel.indices)
            reward = 0
            for j in range(max_iterations):
                action = self.get_action(state)
                new_state, r, done = self.env.step(action)
                new_state = get_state_list(new_state, self.linearModel.indices)
                reward += r

                # p = (self.linearModel).predict(new_state)
                # max_a = (np.where(p == np.amax(p)))[0][0]
                target = q(state, action, self.linearModel.weight, self.linearModel.bias)
                target -= r
                target -= self.gamma * np.amax(self.linearModel.predict(new_state))
                self.linearModel.update(state, action, target)

                state = new_state

                if (done == True): 
                    break
            reward_output.append(reward)
        
        return reward_output, self.linearModel


if __name__ == "__main__":
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])

    env = MountainCar(mode=mode)
    agent = QLearningAgent(env, mode=mode, gamma=gamma, epsilon=epsilon, lr=learning_rate)
    reward_output, linearModel = agent.train(episodes, max_iterations)

    weight_out_file = open(weight_out, "w")
    returns_out_file = open(returns_out, "w")

    weight_out_file.write(str(linearModel.bias)+"\n")
    for line in linearModel.weight:
        for elem in line:
            weight_out_file.write(str(elem)+"\n")
    for i in reward_output:
        returns_out_file.write(str(i)+"\n")
    
    weight_out_file.close()
    returns_out_file.close()


