import gym
from gym import spaces
import numpy as np
import random

class HexTile:
    def __init__(self, resource, number):
        self.resource = resource  # Type of resource
        self.number = number  # Number index associated with the tile
        self.neighbors = []  # List of neighboring hextile objects

    def add_neighbor(self, neighbor_tile):
        if neighbor_tile not in self.neighbors:
            self.neighbors.append(neighbor_tile)
            neighbor_tile.neighbors.append(self)  # Ensure bidirectional link between tiles

    def __repr__(self):
        return f"HexTile(resource={self.resource}, number={self.number})"

class Settlement:
    def __init__(self, tile1, tile2, tile3):
        self.tiles = [tile1, tile2, tile3]
    
    def get_resources(self, rolled_number):
        resources = {}
        for tile in self.tiles:
            if tile.number == rolled_number and tile.resource != 'barren':
                if tile.resource not in resources:
                    resources[tile.resource] = 0
                resources[tile.resource] += 1
        return resources

class Road:
    def __init__(self, tile1, tile2):
        self.tiles = (tile1, tile2)

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
            # print(f"Dice rolled: {rolled_number}")
            self.distribute_resources(rolled_number)

        state = self.get_state()
        info = {}
        return state, reward, self.done, info

    def roll_dice(self):
        return random.randint(1, 6) + random.randint(1, 6)

    def distribute_resources(self, rolled_number):
        for player in self.players:
            for settlement in player['settlements']:
                resources = settlement.get_resources(rolled_number)
                for resource, amount in resources.items():
                    player['resources'][resource] += amount
                    # print(f"Player receives {amount} {resource} from settlement at intersection of tiles {settlement.tiles}.")
        #self.print_player_resources()  # Debugging: print resources after distribution

    def reset(self):
        self.board = self.initialize_board()
        self.players = self.initialize_players()
        self.current_player = 0
        self.done = False
        self.set_starting_positions()

        return self.get_state()

    def set_starting_positions(self):
        starting_positions = [
            {'tiles': [self.board[0], self.board[1], self.board[4]]},  # Player 1
            {'tiles': [self.board[5], self.board[6], self.board[9]]},  # Player 2
            {'tiles': [self.board[10], self.board[11], self.board[14]]},  # Player 3
            {'tiles': [self.board[15], self.board[16], self.board[18]]}   # Player 4
        ]


        for i, player in enumerate(self.players):
            tiles = starting_positions[i]['tiles']
            settlement = Settlement(tiles[0], tiles[1], tiles[2])
            player['settlements'].append(settlement)

    def initialize_board(self):
        # Create hex tiles with resources and numbers
        tiles = [
            HexTile('ore', 10), HexTile('brick', 2), HexTile('wheat', 9),
            HexTile('wheat', 12), HexTile('wood', 6), HexTile('wheat', 4), HexTile('wood', 10),
            HexTile('ore', 9), HexTile('wood', 11), HexTile('sheep', 3), HexTile('sheep', 6), 
            HexTile('brick', 8), HexTile('wood', 5), HexTile('wheat', 8), HexTile('sheep', 4), 
            HexTile('ore', 3), HexTile('wheat', 5), HexTile('barren', 0), HexTile('brick', 11)
        ]
        
        # Manually defining neighbors based on the board layout in the image

        # Top row
        tiles[0].add_neighbor(tiles[1])
        tiles[0].add_neighbor(tiles[4])
        tiles[1].add_neighbor(tiles[0])
        tiles[1].add_neighbor(tiles[2])
        tiles[1].add_neighbor(tiles[4])
        tiles[1].add_neighbor(tiles[5])
        tiles[2].add_neighbor(tiles[1])
        tiles[2].add_neighbor(tiles[5])
        tiles[2].add_neighbor(tiles[6])

        # Second row
        tiles[3].add_neighbor(tiles[4])
        tiles[3].add_neighbor(tiles[7])
        tiles[4].add_neighbor(tiles[0])
        tiles[4].add_neighbor(tiles[1])
        tiles[4].add_neighbor(tiles[3])
        tiles[4].add_neighbor(tiles[5])
        tiles[4].add_neighbor(tiles[7])
        tiles[5].add_neighbor(tiles[1])
        tiles[5].add_neighbor(tiles[2])
        tiles[5].add_neighbor(tiles[4])
        tiles[5].add_neighbor(tiles[6])
        tiles[5].add_neighbor(tiles[8])
        tiles[6].add_neighbor(tiles[2])
        tiles[6].add_neighbor(tiles[5])
        tiles[6].add_neighbor(tiles[8])

        # Third row
        tiles[7].add_neighbor(tiles[3])
        tiles[7].add_neighbor(tiles[4])
        tiles[7].add_neighbor(tiles[8])
        tiles[7].add_neighbor(tiles[12])
        tiles[8].add_neighbor(tiles[5])
        tiles[8].add_neighbor(tiles[6])
        tiles[8].add_neighbor(tiles[7])
        tiles[8].add_neighbor(tiles[9])
        tiles[8].add_neighbor(tiles[12])
        tiles[9].add_neighbor(tiles[6])
        tiles[9].add_neighbor(tiles[8])
        tiles[9].add_neighbor(tiles[10])
        tiles[9].add_neighbor(tiles[13])
        tiles[10].add_neighbor(tiles[9])
        tiles[10].add_neighbor(tiles[11])
        tiles[10].add_neighbor(tiles[14])
        tiles[11].add_neighbor(tiles[10])
        tiles[11].add_neighbor(tiles[15])

        # Fourth row
        tiles[12].add_neighbor(tiles[7])
        tiles[12].add_neighbor(tiles[8])
        tiles[12].add_neighbor(tiles[13])
        tiles[13].add_neighbor(tiles[9])
        tiles[13].add_neighbor(tiles[12])
        tiles[13].add_neighbor(tiles[14])
        tiles[14].add_neighbor(tiles[10])
        tiles[14].add_neighbor(tiles[13])
        tiles[14].add_neighbor(tiles[15])
        tiles[15].add_neighbor(tiles[11])
        tiles[15].add_neighbor(tiles[14])
        tiles[15].add_neighbor(tiles[16])

        # Fifth row
        tiles[16].add_neighbor(tiles[15])
        tiles[16].add_neighbor(tiles[17])
        tiles[17].add_neighbor(tiles[16])
        tiles[17].add_neighbor(tiles[18])
        tiles[18].add_neighbor(tiles[17])

        return tiles

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
            tile1, tile2 = self.choose_road_tiles(player)
            reward = self.build_road(player, tile1, tile2)
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
        return 1 # Encourage passing turn but not as much as building

    def has_valid_moves(self, player):
        if self.can_build_road(player): # Check for valid road
            return True
        if self.can_build_settlement(player): # Check for valid settlement
            return True
        for action in range(3, 23): # Check for valid trades with the bank
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
            return 15  # Reward for successful trade
        return -1  # Penalty for insufficient resources to trade

    # Road must be connected to an existing road or settlement
    def build_road(self, player, tile1, tile2):
        if player['resources']['wood'] > 0 and player['resources']['brick'] > 0:
            if self.is_valid_road_position(player, tile1, tile2):
                player['resources']['wood'] -= 1
                player['resources']['brick'] -= 1
                road = Road(tile1, tile2)
                player['roads'].append(road)
                return 50  # Reward for building a road
        return -1

    # Settlement must be connected to player's roads and not adjacent to another settlement
    def build_settlement(self, player):
        player_index = self.players.index(player)
        for settlement_tiles in self.get_potential_settlement_positions(player):
            if self.is_valid_settlement_position(player, *settlement_tiles):
                player['resources']['wood'] -= 1
                player['resources']['brick'] -= 1
                player['resources']['wheat'] -= 1
                player['resources']['sheep'] -= 1
                settlement = Settlement(*settlement_tiles)
                player['settlements'].append(settlement)
                player['victory_points'] += 1
                print(f"Player {player_index + 1} has built a settlement and now has {player['victory_points']} victory points!")
                # Check if player has won
                if player['victory_points'] >= 5:  # Lowered victory points requirement
                    self.done = True
                    print(f"Player {player_index + 1} wins the game with {player['victory_points']} victory points!")
                return 100  # Reward for building a settlement
        return -1

    # Road must be connected to an existing road or settlement
    def is_valid_road_position(self, player, tile1, tile2):
        for settlement in player['settlements']:
            if tile1 in settlement.tiles or tile2 in settlement.tiles:
                return True
        for road in player['roads']:
            if tile1 in road.tiles or tile2 in road.tiles:
                return True
        return False

    # Settlement can't be adjacent to another settlement and must be connected to a road
    def is_valid_settlement_position(self, player, tile1, tile2, tile3):
        if any(any(t in settlement.tiles for t in [tile1, tile2, tile3]) for settlement in player['settlements']):
            return False
        for road in player['roads']:
            if tile1 in road.tiles or tile2 in road.tiles or tile3 in road.tiles:
                return True
        return False

    # Iterate through all possible groups of three adjacent tiles on the board
    def get_potential_settlement_positions(self, player):
        potential_positions = []
        for tile1 in self.board:
            for tile2 in tile1.neighbors:
                for tile3 in tile2.neighbors:
                    if tile3 in tile1.neighbors:
                        potential_positions.append((tile1, tile2, tile3))
        return potential_positions

    def get_state(self):
        state = []

        # Board state (tiles)
        for tile in self.board:
            state.extend(self.one_hot_encode_tile(tile.resource))
        state.extend(tile.number for tile in self.board)

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
        # Ensure it checks the self.done flag, which is set when a player wins
        return self.done

    '''
    def print_player_resources(self):
        # Print resources of each player for debugging
        for i, player in enumerate(self.players):
            print(f"Player {i + 1} resources: {player['resources']}")
    '''

    def choose_road_tiles(self, player):
        # For simplicity, just choose the first valid pair of tiles connected by an existing road or settlement
        for tile1 in self.board:
            for tile2 in tile1.neighbors:
                if self.is_valid_road_position(player, tile1, tile2):
                    return tile1, tile2
        return None, None

# Test the environment
env = CatanEnv()
state = env.reset()
#print("Initial state:", state)

for _ in range(1000):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    #print("Action:", action, "State:", state, "Reward:", reward, "Done:", done)
    if done:
        print("Game completed.")
        break
