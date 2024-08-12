import gym
from gym import spaces
import numpy as np
import random

class CatanEnv(gym.Env):
    def __init__(self):
        super(CatanEnv, self).__init__()
        self.num_players = 4
        self.current_player = 0

        self.action_space = spaces.Discrete(23)  # Number of possible/valid actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(165,), dtype=np.float32)

        # Initialize game state
        self.reset()

    def step(self, action):
        reward = 0
        if not self.done:
            # Perform the chosen action
            reward = self.perform_action(action)

            # Roll dice and distribute resources
            rolled_number = self.roll_dice()
            self.distribute_resources(rolled_number)

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
        self.current_player = 0
        self.done = False
        self.set_starting_positions()

        return self.get_state()

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
        if not self.is_valid_settlement_position(player, position):
            return -1  # Invalid position
        player['settlements'].append(position)
        self.update_resources(player, position)
        return 50  # Reward for valid settlement

    def place_road(self, player, position):
        if not self.is_valid_road_position(player, position):
            return -1  # Invalid position
        player['roads'].append(position)
        return 10  # Reward for valid road

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
                'victory_points': 0,
                'settlements': [],
                'roads': []
            }
            players.append(player)
        return players

    def perform_action(self, action):
        player = self.players[self.current_player]
        reward = 0
        valid_action = True

        if action == 0:  # Build a road
            reward = self.build_road(player)
        elif action == 1:  # Build a settlement
            reward = self.build_settlement(player)
        elif action == 2:  # Pass to next player
            reward = self.pass_turn(player)
        else:  # Trade with bank
            give, receive = self.map_action_to_trade(action)
            reward = self.trade_with_bank(player, give, receive)

        if reward == -1:  # Penalty for attempting an invalid action
            reward = -0.1
            valid_action = False

        if valid_action:  # If action is valid, move on to next player
            self.current_player = (self.current_player + 1) % len(self.players)
        else:  # Check if there are no more valid actions
            if not self.has_valid_moves(player):
                self.current_player = (self.current_player + 1) % len(self.players)

        # Check if the game is done after each action
        self.done = self.check_done()

        return reward

    def pass_turn(self, player):
        self.current_player = (self.current_player + 1) % self.num_players  # Move on to next player
        return 1  # Encourage passing turn but not as much as building

    def has_valid_moves(self, player):
        if self.can_build_road(player):  # Check for valid road
            return True
        if self.can_build_settlement(player):  # Check for valid settlement
            return True
        for action in range(3, 23):  # Check for valid trades with the bank (adjusted to start from 3)
            give, receive = self.map_action_to_trade(action)
            if give is not None and self.can_trade_with_bank(player, give):
                return True
        return False


    def can_build_road(self, player):
        return player['resources']['wood'] > 0 and player['resources']['brick'] > 0

    def can_build_settlement(self, player):
        return (player['resources']['wood'] > 0 and 
                player['resources']['brick'] > 0 and 
                player['resources']['wheat'] > 0 and 
                player['resources']['sheep'] > 0)

    def can_trade_with_bank(self, player, give):
        return player['resources'][give] >= 4

    def map_action_to_trade(self, action):
        if 3 <= action <= 6:  # Trade wood
            give = 'wood'
            receive = ['brick', 'wheat', 'sheep', 'ore'][action - 3]
        elif 7 <= action <= 10:  # Trade brick
            give = 'brick'
            receive = ['wood', 'wheat', 'sheep', 'ore'][action - 7]
        elif 11 <= action <= 14:  # Trade wheat
            give = 'wheat'
            receive = ['wood', 'brick', 'sheep', 'ore'][action - 11]
        elif 15 <= action <= 18:  # Trade sheep
            give = 'sheep'
            receive = ['wood', 'brick', 'wheat', 'ore'][action - 15]
        elif 19 <= action <= 22:  # Trade ore
            give = 'ore'
            receive = ['wood', 'brick', 'wheat', 'sheep'][action - 19]
        else:
            give = None
            receive = None
        return give, receive



    def trade_with_bank(self, player, give, receive):
        if give is None or receive is None:
            return -1  # Invalid trade action

        if player['resources'][give] >= 4:
            player['resources'][give] -= 4
            player['resources'][receive] += 1
            return 5  # Reward for successful trade
        return -1  # Penalty for insufficient resources to trade


    def build_road(self, player):
        # Road must be connected to an existing road or settlement
        if player['resources']['wood'] > 0 and player['resources']['brick'] > 0:
            for road in player['roads']:
                if self.is_valid_road_position(player, road):
                    player['resources']['wood'] -= 1
                    player['resources']['brick'] -= 1
                    player['roads'].append(road)
                    return 10  # Reward for building a road
        return -1

    def build_settlement(self, player):
        # Settlement must be connected to one of the player's roads and not adjacent to another settlement
        if player['resources']['wood'] > 0 and player['resources']['brick'] > 0 and player['resources']['wheat'] > 0 and player['resources']['sheep'] > 0:
            for road in player['roads']:
                if self.is_valid_settlement_position(player, road):
                    player['resources']['wood'] -= 1
                    player['resources']['brick'] -= 1
                    player['resources']['wheat'] -= 1
                    player['resources']['sheep'] -= 1
                    player['settlements'].append(road)
                    player['victory_points'] += 1
                    return 50  # Reward for building a settlement
        return -1

    def is_valid_road_position(self, player, position):
        # Road must be connected to an existing road or settlement
        for settlement in player['settlements']:
            if self.is_adjacent(settlement, position):
                return True
        for road in player['roads']:
            if self.is_adjacent(road, position):
                return True
        return False

    def is_valid_settlement_position(self, player, position):
        # Settlement must not be adjacent to another settlement and must be connected to a road
        if any(self.is_adjacent(settlement, position) for settlement in player['settlements']):
            return False
        if any(self.is_adjacent(road, position) for road in player['roads']):
            return True
        return False

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

# Test the environment with the new action logic
env = CatanEnv()
state = env.reset()
print("Initial state:", state)

for _ in range(10):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print("Action:", action, "State:", state, "Reward:", reward, "Done:", done)
    if done:
        break
