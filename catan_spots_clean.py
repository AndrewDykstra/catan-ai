import gym
from gym import spaces
import numpy as np

# definining the starting board for training.
# There are 19 tiles which each have corresponding numbers that can be rolled
# on two 6 sided dice (excluding 7 by game rules)
board= {'tiles': ['sheep', 'wheat', 'wheat',
                'ore', 'ore', 'ore', 'sheep',
                'wood', 'wheat', 'sheep', 'brick', 'wood',
                'wood', 'wood', 'wheat', 'brick',
                'sheep', 'barren', 'brick'],
        'numbers': [5, 8, 4,
                    2, 10, 3, 11,
                    6, 9, 11, 6, 12,
                    3, 4, 5, 9,
                    8, 0, 10]}
# at each of the six sides of a tile there is a vertex.
#  This list defines which vertices are connected by an edge
vertex_neighbors = [[3, 4], [4, 5], [5, 6],
                    [0, 7], [0, 1, 8], [1, 2, 9], [2, 10],
                    [3, 11, 12], [4, 12, 13], [5, 13, 14], [6, 14, 15],
                    [7, 16], [7, 8, 17], [8, 9, 18],  [9, 10, 19], [10, 20],
                    [11, 21, 22], [12, 22, 23], [13, 23, 24], [14, 24, 25], [15, 25, 26],
                    [16, 27], [16, 17, 28], [17, 18, 29], [18, 19, 30], [19, 20, 31], [20, 32],
                    [21, 33], [22, 33, 34], [23, 34, 35], [24, 35, 36], [25, 36, 37], [26, 37],
                    [27, 28, 38], [28, 29, 39], [29, 30, 40], [30, 31, 41], [31, 32, 42],
                    [33, 43], [34, 43, 44], [35, 44, 45], [36, 45, 46], [37, 46],
                    [38, 39, 47], [39, 40, 48], [40, 41, 49], [41, 42, 50],
                    [43, 51], [44, 51, 52], [45, 52, 53], [46, 53],
                    [47, 48], [48, 49], [49, 50]]

# order of tiles = wood, sheep, wheat, ore, brick, barren
# start_weights is the initial value for res_weights
start_weights = [1, 1, 1, 1, 1, 0]

# the probability of each number being rolled on two six sided dice.
# This list starts with the probability of zero being rolled and goes up to 12
probability_dist = [(0/36), (0/36), (1/36), (2/36), (3/36), (4/36), (5/36), (6/36), (5/36), (4/36), (3/36), (2/36), (1/36)]

# choose_starts takes in the game board and a list of length 6 with numbers
# corresponding to the number of times that resource has been used in the game
def choose_starts(board, res_weights):
    tiles = board.get('tiles')
    tile_indexes = index_resources(tiles)
    numbers = board.get('numbers')
    # cart_board will represent each tile and its number
    cart_board = [0]*19
    for x in range(0, 19):
        cart_board[x] = (tile_indexes[x], numbers[x])
    # define vertices as a list of vertexes of length 54
    # each vertex is a list of (tile, numbers) that the vertex touches
    vertices = get_vertices(cart_board)
    # a list of len 54 of each of the vertices' priorities
    vpriorities = vertices_priority(vertices, res_weights)
    # get a list of pairs (each of form [high priority vertex, lower priority vertex, pair priority]),
    # and a list of the priorities of each of these pairs
    pairs, priorities = all_pairs(vertices, vpriorities, res_weights)
    # get the best picks to prune less optimal branches
    top_pairs, top_vertices = best_picks(pairs, vertices)
    # get the optimal list of choices in order of choice
    optpicking = turn_picking(3, top_pairs, top_vertices, res_weights)
    return optpicking

# Recursive function that finds the optimal choices for every player
# The player with the first turn has turn 3, the player with the last turn has 0
def turn_picking(turn, current_pairs, current_vertices, res_weights):
    # When we get to the fourth player, the best pair of vertices is just
    # the first remaining pair on the list
    if (turn == 0):
        return [current_pairs[0][0], current_pairs[0][1]]

    # for each other player, they have a list of remaining pairs and vertices
    else:
        pick_priorities = []
        picks = []
        remaining = []
        indexes = list(current_vertices.keys())
        # the first vertex in the pair will be in the first half of the list
        greedy_vertices = indexes[0:len(indexes)//2]

        for vertex in greedy_vertices:
            # choose this vertex as our pick
            thispick = [vertex]
            # remove this vertex and any neighbors of this vertex for the inner picks
            inner_pairs, inner_vertices = remove_neighbors(vertex, current_pairs, current_vertices)
            # find the picks that will happen between this players turns
            inner_picks = turn_picking(turn-1, inner_pairs, inner_vertices, res_weights)
            for each in inner_picks:
                thispick.append(each)
            # create a new list of pairs and vertices remaining, that has our first pick
            # but not neighbors of the picks that have happened since our pick
            left_pairs = current_pairs
            left_vertices = current_vertices
            for pick in inner_picks:
                left_pairs, left_vertices = remove_neighbors(pick, left_pairs, left_vertices)

            # pick the best second choice given our first pick
            for pair in left_pairs:
                if pair[0] == vertex:

                    nextpick = pair[1]
                    thispick.append(nextpick)
                    picks.append(thispick)
                    pick_priorities.append(get_priority(current_vertices.get(vertex) + current_vertices.get(nextpick), res_weights))
                    break
        # find the best pick of all the ones that we've tried
        best = pick_priorities.index(max(pick_priorities))
        bestpick = picks[best]
        return bestpick



# Remove all neighbors from the pairs and vertices list for a given choice
def remove_neighbors(choice, pairs, vertices):
    new_pairs = []
    for pair in pairs:
        if (pair[0] == choice) or (pair[1] == choice):
            continue
        elif (pair[0] in vertex_neighbors[choice]):
            continue
        elif (pair[1] in vertex_neighbors[choice]):
            continue
        else:
            new_pairs.append(pair)

    new_vertices = {}
    for vertex in list(vertices.keys()):
        if (vertex == choice) or (vertex in vertex_neighbors[choice]):
            continue
        else:
            new_vertices[vertex]=(vertices.get(vertex))

    return new_pairs, new_vertices

# best picks helps us to reduce the amount of branches and calculations made
# it finds the first 4 pairs that are all valid, adding every vertex
# that is seen up to that 4th pair.
# This list is the greedy_best choice for each player
# We know that the optimal choices will be greater than or equal to these
# choices, so we don't need to consider those pairs
def best_picks(pairs, vertices):
    greedy_pairs = []
    greedy_vertices = {}
    drop_point = 0

    for i in range(0, len(pairs)):
        if not (pairs[i][0] in greedy_vertices):
            greedy_vertices[pairs[i][0]]=(vertices[pairs[i][0]])
        if not (pairs[i][1] in greedy_vertices):
            greedy_vertices[pairs[i][1]]=(vertices[pairs[i][1]])
        if (check_valid_pair(pairs[i], greedy_pairs)):
            greedy_pairs.append(pairs[i])
        if len(greedy_pairs) == 4:
            drop_point = i
            break
    return pairs[0:(drop_point+1)], greedy_vertices


# check that a pair can be placed given the other pairs that were already placed
def check_valid_pair(pair, previouspairs):
    x = pair[0]
    y = pair[1]
    for each in previouspairs:
        a = each[0]
        b = each[1]
        if((x == a) or (x == b) or (y == a) or (y == b)):
            return False
        if ((x in vertex_neighbors[(a)]) or (x in vertex_neighbors[(b)]) or
            (y in vertex_neighbors[(a)]) or (y in vertex_neighbors[(b)])):
            return False

    return True





# vertices_priority constructs a list of the priorities of each vertex
def vertices_priority(vertices, res_weights):
    vpriority = []
    vpriority = [get_priority(vertex, res_weights) for vertex in vertices]
    return vpriority

# get_priority takes in a list of tiles and a list of resource weights of
# length 6 where each number corresponds to the number of times that resource
# has been played in our games
def get_priority(tiles_list, res_weights):
    priority = 0
    tile_count = [0, 0, 0, 0, 0, 0]
    probabilities = [(0/36)]*6

    # count the number of times each tile is in the list
    # find the probability of each tile will be picked from that list on a roll
    for tile in tiles_list:
        tile_count[tile[0]] += 1
        probabilities[tile[0]] += probability_dist[tile[1]]

    # for each tile, find the priority and add it to the overall priority
    for j in range(0, 6):
        # if the tile is in the list more than once, discount it to encourage
        # an even spread of resources
        overstock = (11 - tile_count[j])/10
        # for each tile, the priority is the discount
        # times the probability of that tile times the weight of the resource
        # times a constant 100
        jpriority = 100*overstock*probabilities[j]*res_weights[j]
        priority += jpriority

    return priority


def index_resources(tiles):
    tile_indexes = []
    for resource in tiles:
        if (resource == 'wood'):
            tile_indexes.append(0)
            continue
        elif (resource == 'sheep'):
            tile_indexes.append(1)
            continue
        elif (resource == 'wheat'):
            tile_indexes.append(2)
            continue
        elif (resource == 'ore'):
            tile_indexes.append(3)
            continue
        elif (resource == 'brick'):
            tile_indexes.append(4)
            continue
        elif (resource == 'barren'):
            tile_indexes.append(5)
            continue
    return tile_indexes

# all_pairs will return a sorted list (high to low by priority)
# of all valid pairs in a game where each pair is of the form:
# [higher priority vertex, lower priority vertex, pair priority]
def all_pairs(vertices, vpriority, res_weights):

    pairs = []
    priorities = []

    for x in range(0, 54):
        for y in range(0, 54):
            if (x <= y):
                continue
            elif (y in vertex_neighbors[x]):
                continue
            else:
                priority = get_priority(vertices[y] + (vertices[x]), res_weights)
                if (vpriority[x] >= vpriority[y]):
                    pairs.append([x, y, priority])
                    priorities.append(priority)
                else:
                    pairs.append([y, x, priority])
                    priorities.append(priority)

    return sort_pairs(pairs, priorities)

# adapting quick sort
def sort_pairs(list, values):
    if len(values) == 1:
        return list, values
    else:
        list1=[]
        values1=[]
        list2=[]
        values2=[]

        split = (max(values)+min(values))/2

        if (split == max(values)):
            return list, values
        for x in range(0, len(values)):
            if values[x]>split:
                list1.append(list[x])
                values1.append(values[x])
            else:
                list2.append(list[x])
                values2.append(values[x])

        l1, v1 = sort_pairs(list1, values1)
        l2, v2 = sort_pairs(list2, values2)

        return l1+l2, v1+v2






# define vertices as a list of vertexes of length 54
# each vertex is a list of (tile, numbers) that the vertex touches
def get_vertices(cart_board):
    v = []
    for something in range(0, 54):
        v.append([])

    for i in range(0, 3):
        # define vertices 0-2
        v[i].append(cart_board[i])


        # define vertices 3-6
        v[(i+3)].append(cart_board[i])
        v[(i+4)].append(cart_board[i])


        # define vertices 7-10 right and left side
        v[(i+7)].append(cart_board[i])
        v[(i+8)].append(cart_board[i])


        # define vertices 12-14 above tiles
        v[(i+12)].append(cart_board[i])



        # define vertices 29-41 above and below tiles
        v[(i+39)].append(cart_board[i+16])


        # define vertices 43-46 right and left sides
        v[i+43].append(cart_board[i+16])
        v[i+44].append(cart_board[i+16])

        # define vertices 47-50
        v[i+47].append(cart_board[i+16])
        v[i+48].append(cart_board[i+16])

        # define vertices 51-53
        v[i+51].append(cart_board[i+16])


    for i in range(0, 4):
        # define vertices 7-10 below tiles
        v[i+7].append(cart_board[i+3])

        # define vertices 11-15 right and left sides
        v[i+11].append(cart_board[i+3])
        v[i+12].append(cart_board[i+3])

        # define vertices 16-20 right and left sides
        v[i+16].append(cart_board[i+3])
        v[i+17].append(cart_board[i+3])

        # define vertices 22-25 above tiles
        v[i+22].append(cart_board[i+3])


        # define vertices 28-31 below tiles
        v[i+28].append(cart_board[i+12])

        # define vertices 33-37 right and left sides
        v[i+33].append(cart_board[i+12])
        v[i+34].append(cart_board[i+12])

        # define vertices 38-42 right and left sides
        v[i+38].append(cart_board[i+12])
        v[i+39].append(cart_board[i+12])

        # define vertices 43-46 above tiles
        v[i+43].append(cart_board[i+12])

    for i in range(0, 5):
        # define vertices 16-20 below tiles
        v[i+16].append(cart_board[i+7])

        #define vertices 21-26 left and right tiles
        v[i+21].append(cart_board[i+7])
        v[i+22].append(cart_board[i+7])


        #define vertices 27-32 left and right tiles
        v[i+27].append(cart_board[i+7])
        v[i+28].append(cart_board[i+7])

        # define vertices 33-37 above tiles
        v[i+33].append(cart_board[i+7])

    return v



print(choose_starts(board, start_weights))
