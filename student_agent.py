# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

with open('q_table.pkl', 'rb') as f:
    QTable = pickle.load(f)

have_picked = False
passenger_picked = False
target_look = False
destination_pos = None
destination_idx = None
target_pos = None
target_idx = None
action_prev = None
visited = []
stations = []


def get_relative_pos(pos1, pos2):
    return pos2[0] - pos1[0], pos2[1] - pos1[1]

def bus_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def at_station(pos):
    global stations
    for i in range(4):
        if pos == stations[i]:
            return i
    return -1

def nothing_station(taxi_pos, passenger_look):
    global destination_pos
    return passenger_look == 0 or (passenger_picked and taxi_pos != destination_pos)



def find_nearest_target(taxi_pos):
    global destination_idx, destination_pos, visited, stations

    distance_array = [(i, bus_distance(taxi_pos, stations[i])) for i in range(4) if visited[i] == 0]
    if len(distance_array) == 0:
        return destination_idx, destination_pos

    t_idx = min(distance_array, key=lambda x: x[1])[0]
    return t_idx, stations[t_idx]


def get_state(obs, action=None):
    global have_picked, target_look, passenger_picked, destination_pos, destination_idx, target_pos, target_idx, visited, stations
    taxi_row, taxi_col, _,_,_,_,_,_,_,_, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    taxi_pos = (taxi_row, taxi_col)
    shaped_reward = 0


    at_where = at_station(taxi_pos)
    target_relative_pos = get_relative_pos(taxi_pos, target_pos)


    if at_where != -1 and destination_look == 1:
        destination_idx = at_where
        destination_pos = taxi_pos


    if action == 4:
        if have_picked == False and at_where != -1 and passenger_look == 1 and passenger_picked == False:
            if visited[at_where] == 0:
                visited[at_where] = 1

            passenger_picked = True
            have_picked = True
            shaped_reward += 75
            if destination_pos != None:
                target_idx = destination_idx
                target_pos = destination_pos
            else:
                target_idx, target_pos = find_nearest_target(taxi_pos)
            target_relative_pos = get_relative_pos(taxi_pos, target_pos)

        elif have_picked and not passenger_picked and target_relative_pos == (0,0):
            shaped_reward += 5
            passenger_picked = True
            if destination_pos != None:
                target_idx = destination_idx
                target_pos = destination_pos
            else:
                target_idx, target_pos = find_nearest_target(taxi_pos)
            target_relative_pos = get_relative_pos(taxi_pos, target_pos)


    elif action == 5:
        if passenger_picked:
            passenger_picked = False
            target_idx = -1
            target_pos = taxi_pos
            if (at_where == -1 or (at_where != -1 and destination_look == 0)):
                shaped_reward -= 40

    else:
        if at_where != -1:

            if target_relative_pos == (0,0):
                if target_idx != -1 and visited[target_idx] == 0:
                    #shaped_reward += 20
                    visited[target_idx] = 1

                if nothing_station(taxi_pos, passenger_look):
                    target_idx, target_pos = find_nearest_target(taxi_pos)
                    target_relative_pos = get_relative_pos(taxi_pos, target_pos)



    if have_picked == False and at_where != -1 and passenger_look == 1:
        target_look = True
    elif have_picked == True and passenger_picked == False and target_relative_pos == (0,0):
        target_look = True
    elif passenger_picked == True and taxi_pos == destination_pos:
        target_look = True
    else:
        target_look = False


    return (target_relative_pos, passenger_picked, target_look, obstacle_north, obstacle_south, obstacle_east, obstacle_west), shaped_reward



EPSILON = 0.07
def get_action(obs):

    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys.
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    global have_picked, target_look, passenger_picked, destination_pos, destination_idx, target_pos, target_idx, visited, stations
    global action_prev
    if len(visited) == 0:
        visited = [0, 0, 0, 0]

    if len(stations) == 0:
        stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]

    if target_idx == None:
        target_idx, target_pos = find_nearest_target((obs[0], obs[1]))

    state,_ = get_state(obs, action_prev)

    action = 0
    if np.random.random() > EPSILON:
        action = np.argmax(QTable[state])
    else:
        action = np.random.randint(0,6)

    action_prev = action
    return action

