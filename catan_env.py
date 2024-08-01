import gym
from gym import spaces
import numpy as np

class CatanEnv(gym.Env):
    def __init__(self):
        super(CatanEnv, self).__init__()
        self.action_space = spaces.Discrete(10)  # This is for the amount of actions possible for the agent, I just set it to 10 as a baseline for now, feel free to modify
        self.observation_space = spaces.Box(low=0, high=1, shape=(50,), dtype=np.float32) # This is the size of the observation space which I also just set to 50 as a baseline

        # Initialize game state
        self.reset()

    # This function will reset the current game state
    def reset(self):
        self.board = self.initialize_board()
        self.players = self.initialize_players()
        self.current_player = 0
        self.done = False

        return self.get_state()

    # This applies an action and updates the current game state
    def step(self, action):
        reward = 0
        if not self.done:
            reward = self.perform_action(action)
            self.current_player = (self.current_player + 1) % len(self.players)
            self.done = self.check_done()
        
        state = self.get_state()
        info = {}
        
        return state, reward, self.done, info

    # This will generate a board with resources tiles and numbers randomly to ensure new boards each game
    def initialize_board(self):
        board = {
            'tiles': ['wood', 'brick', 'wheat', 'sheep', 'ore'] * 3,
            'numbers': [2, 3, 4, 5, 6, 8, 9, 10, 11, 12] * 2
        }
        np.random.shuffle(board['tiles'])
        np.random.shuffle(board['numbers'])
        return board

    # This will initialize all 4 players with 0 resources to start and 0 roads or settlements
    def initialize_players(self):
        players = []
        for _ in range(4):
            player = {
                'resources': {'wood': 0, 'brick': 0, 'wheat': 0, 'sheep': 0, 'ore': 0},
                'victory_points': 0,
                'settlements': [],
                'roads': []
            }
            players.append(player)
        return players

    # This will make the game environment execute actions that the AI agent makes
    # TODO: Finish the implementation of the different actions the AI can make
    def perform_action(self, action):
        player = self.players[self.current_player]
        reward = 0
        if action == 0:  # Build a road
            reward = self.build_road(player)
        elif action == 1:  # Build a settlement
            reward = self.build_settlement(player)
        # TODO: We need to add the other actions the agent can make here
        return reward

    # Function to build a road, will also handle removing resources + rewarding or penalizing agent
    def build_road(self, player):
        if player['resources']['wood'] > 0 and player['resources']['brick'] > 0:
            player['resources']['wood'] -= 1
            player['resources']['brick'] -= 1
            player['roads'].append('new_road')
            return 1  # Reward for building a road
        return -1  # If unable to build it will return -1 so the agent learns not to attempt this action without proper resources

    # This will build a settlement and handle removing resources the player will use, rewards or penalizes agent
    def build_settlement(self, player):
        if player['resources']['wood'] > 0 and player['resources']['brick'] > 0 and player['resources']['wheat'] > 0 and player['resources']['sheep'] > 0:
            player['resources']['wood'] -= 1
            player['resources']['brick'] -= 1
            player['resources']['wheat'] -= 1
            player['resources']['sheep'] -= 1
            player['settlements'].append('new_settlement')
            player['victory_points'] += 1
            return 1 # Reward for building a settlement
        return -1 # If unable to build it will return -1 so the agent learns not to attempt this action without proper resources

    def get_state(self):
        state = np.random.rand(50) # TODO:  Implement game state representation
        return state

    # Check if there is currently a winner (10 points)
    def check_done(self):
        for player in self.players:
            if player['victory_points'] >= 10: # Win condition
                return True
        return False

# This will test environment creation
env = CatanEnv()
state = env.reset()
print("Initial state:", state)

for _ in range(10):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print("Action:", action, "State:", state, "Reward:", reward, "Done:", done)
    if done:
        break
