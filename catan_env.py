import gym
from gym import spaces
import numpy as np
import random

class CatanEnv(gym.Env):
    def __init__(self):
        super(CatanEnv, self).__init__()
        self.action_space = spaces.Discrete(23) # Number of possible/valid actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(165,), dtype=np.float32) 

        # Initialize game state
        self.reset()

    def nextTurn(self):
        # Increment current player
        self.current_player = (self.current_player + 1) % len(self.players)
        print (f'\nIt is Player {self.current_player + 1}\'s turn.')

        # Roll dice and distribute resources
        rolled_number = self.roll_dice()
        print (f'Player {self.current_player + 1} rolled a {rolled_number}! Distributing Resources...')
        self.distribute_resources(rolled_number)

        return 0

    def step(self, action):
        reward = 0
        if not self.done:
            # Perform the chosen action
            reward = self.perform_action(action)

        state = self.get_state()
        info = {}
        
        return state, reward, self.done, info

    def roll_dice(self):
        return random.randint(1, 6) + random.randint(1, 6)

    def distribute_resources(self, rolled_number):
        for i, number in enumerate(self.board['numbers']):
            if number == rolled_number:
                tile_position = self.get_tile_position(i)
                for player_index, player in enumerate(self.players):
                    for settlement in player['settlements']:
                        if self.is_adjacent(settlement, i):
                            resource = self.board['tiles'][i]
                            if resource != 'barren':
                                player['resources'][resource] += 1
                                print(f"Player {player_index + 1} receives 1 {resource} from tile at {tile_position}.")

    def reset(self):
        self.board = self.initialize_board()
        self.players = self.initialize_players()
        self.done = False
        self.set_starting_positions()
        
        self.current_player = -1
        self.nextTurn()
        
        self.initDevCards()

        return self.get_state()
    
    def initDevCards(self):
        self.dev_card_map = {0 : 'knight',
                             1 : 'victory',
                             2 : 'road building',
                             3 : 'year of plenty',
                             4 : 'monopoly',}
        self.banked_dev_cards = ([0]*14) + ([1]*5) + ([2]*2) + ([3]*2) + ([4]*2)
        random.shuffle(self.banked_dev_cards)

    def set_starting_positions(self):
        starting_positions = [
            {'settlement': (0, 0), 'road': (0, 1)},  # Player 1
            {'settlement': (5, 5), 'road': (5, 6)},  # Player 2
            {'settlement': (10, 10), 'road': (10, 11)},  # Player 3
            {'settlement': (15, 15), 'road': (15, 16)}   # Player 4
        ]

        for i, player in enumerate(self.players):
            position = starting_positions[i]
            self.place_settlement(player, position['settlement'])
            self.place_road(player, position['road'])

    def place_settlement(self, player, position):
        player['settlements'].append(position)
        self.update_resources(player, position)

    def place_road(self, player, position):
        player['roads'].append(position)

    def update_resources(self, player, position):
        adjacent_tiles = self.get_adjacent_tiles(position)

        for tile in adjacent_tiles:
            resource_type = tile['resource']
            if resource_type != 'barren':
                player['resources'][resource_type] += 1

    def get_tile_position(self, tile_index):
        hex_layout = [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
            (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
            (0, 3), (1, 3), (2, 3), (3, 3), (4, 3),
            (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)
        ]
        return hex_layout[tile_index]

    def is_adjacent(self, position, tile_index):
        tile_position = self.get_tile_position(tile_index)
        adjacent_positions = [
            (tile_position[0], tile_position[1] - 1),  # Up
            (tile_position[0], tile_position[1] + 1),  # Down
            (tile_position[0] - 1, tile_position[1]),  # Left
            (tile_position[0] + 1, tile_position[1]),  # Right
            (tile_position[0] - 1, tile_position[1] - 1),  # Up-Left
            (tile_position[0] + 1, tile_position[1] + 1)   # Down-Right
        ]
        return position in adjacent_positions

    def get_adjacent_tiles(self, position):
        adjacent_tiles = []
        for i, tile in enumerate(self.board['tiles']):
            if self.is_adjacent(position, i):
                adjacent_tiles.append({'resource': tile, 'number': self.board['numbers'][i]})
        return adjacent_tiles

    def initialize_board(self):
        board = {
            'tiles': ['sheep', 'wheat', 'wheat', 'ore', 'ore', 'ore', 'sheep',
                      'wood', 'wheat', 'sheep', 'brick', 'wood',
                      'wood', 'wood', 'wheat', 'brick', 'sheep', 'barren', 'brick'],
            'numbers': [5, 8, 4, 2, 10, 3, 11, 6, 9, 11, 6, 12, 3, 4, 5, 9, 8, 0, 10]
        }
        return board

    def initialize_players(self):
        players = []
        for _ in range(4):
            player = {
                'resources': {'wood': 0, 'brick': 0, 'wheat': 0, 'sheep': 0, 'ore': 0},
                'dev_cards': {'knight': 0, 'victory': 0, 'road building': 0, 'year of plenty': 0, 'monopoly': 0},
                'victory_points': 0,
                'settlements': [],
                'roads': []
            }
            players.append(player)
        return players

    def perform_action(self, action):
        player = self.players[self.current_player]
        reward = 0
        if action == 0:  # Build a road
            reward = self.build_road(player)
        elif action == 1:  # Build a settlement
            reward = self.build_settlement(player)
        elif action <= 21:  # Trade with bank -- actions 2-21
            give, receive = self.map_action_to_trade(action)
            reward = self.trade_with_bank(player, give, receive)
        elif action == 22: # End turn
            reward = self.nextTurn()
        # ---------------- WIP ----------------
        elif action == 26: # Buy dev cards
            reward = self.buy_dev_card(player)
        elif action <= 27: # Play dev card -- soldier, year-of-plenty, monopoly, build-roads
            reward = self.use_dev_card(player)
        elif action == 28: # Build city
            pass
            
        else:
            raise Exception('Unknown Action ID')

        # Check if the player has enough resources:
        if reward == -1 and player['resources']['wood'] >= 4 and player['resources']['brick'] >= 4:
            # Encourage agent to trade if it has enough resources to trade but not to build
            reward += 5
        
        return reward

    def buy_dev_card(self, player):
        # check if bank contains any dev cards and if player can afford to buy a dev card
        if len(self.banked_dev_cards) > 0 and player['resources']['wheat'] > 0 and player['resources']['sheep'] > 0 and player['resources']['ore'] > 0:
            # get dev card from bank
            receive = self.dev_card_map[self.banked_dev_cards.pop()]
            # update player inventory
            player['resources']['wheat'] -= 1
            player['resources']['sheep'] -= 1
            player['resources']['ore'] -= 1
            player['dev_cards'][receive] += 1
            
            # update player VP if received a victory card
            if receive == 'victory':
                self.players[player]['victory_points'] += 1
            return 5 # maybe change reward
        return -1
    
    def use_dev_card(self, player, actionId):
        cards = ['soldier', 'year of plenty', 'monopoly', 'road building']
        id_offset = 24
        
        # ------- WIP -----------
        
        return -1

    def map_action_to_trade(self, action):
        if 2 <= action <= 5:  # Trade wood
            give = 'wood'
            receive = ['brick', 'wheat', 'sheep', 'ore'][action - 2]
        elif 6 <= action <= 9:  # Trade brick
            give = 'brick'
            receive = ['wood', 'wheat', 'sheep', 'ore'][action - 6]
        elif 10 <= action <= 13:  # Trade wheat
            give = 'wheat'
            receive = ['wood', 'brick', 'sheep', 'ore'][action - 10]
        elif 14 <= action <= 17:  # Trade sheep
            give = 'sheep'
            receive = ['wood', 'brick', 'wheat', 'ore'][action - 14]
        elif 18 <= action <= 21:  # Trade ore
            give = 'ore'
            receive = ['wood', 'brick', 'wheat', 'sheep'][action - 18]
        return give, receive

    def trade_with_bank(self, player, give, receive):
        if player['resources'][give] >= 4:
            player['resources'][give] -= 4
            player['resources'][receive] += 1
            return 5  # Reward for successful trade
        return -1  # Penalty for insufficient resources to trade

    def build_road(self, player):
        if player['resources']['wood'] > 0 and player['resources']['brick'] > 0:
            player['resources']['wood'] -= 1
            player['resources']['brick'] -= 1
            player['roads'].append('new_road')
            return 10  # Reward for building a road
        return -1

    def build_settlement(self, player):
        if player['resources']['wood'] > 0 and player['resources']['brick'] > 0 and player['resources']['wheat'] > 0 and player['resources']['sheep'] > 0:
            player['resources']['wood'] -= 1
            player['resources']['brick'] -= 1
            player['resources']['wheat'] -= 1
            player['resources']['sheep'] -= 1
            player['settlements'].append('new_settlement')
            player['victory_points'] += 1
            return 50  # Reward for building a settlement
        return -1

    def get_state(self):
        state = []

        # Board state (Tiles and numbers)
        for tile in self.board['tiles']:
            state.extend(self.one_hot_encode_tile(tile))
        state.extend(self.board['numbers'])

        # Player resources, roads, and settlements
        for player in self.players:
            for resource in ['wood', 'brick', 'wheat', 'sheep', 'ore']:
                state.append(player['resources'][resource])
            state.append(len(player['roads']))
            state.append(len(player['settlements']))
            state.append(player['victory_points'])

        return np.array(state, dtype=np.float32)

    def one_hot_encode_tile(self, tile):
        encoding = [0] * 6  # Assume 5 resources + barren
        if tile == 'wood':
            encoding[0] = 1
        elif tile == 'brick':
            encoding[1] = 1
        elif tile == 'wheat':
            encoding[2] = 1
        elif tile == 'sheep':
            encoding[3] = 1
        elif tile == 'ore':
            encoding[4] = 1
        elif tile == 'barren':
            encoding[5] = 1
        return encoding
    
    def check_done(self):
        for player in self.players:
            if player['victory_points'] >= 10:  # Win condition
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
