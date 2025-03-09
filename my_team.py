# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
from queue import PriorityQueue
from collections import deque
import heapq


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='EnhancedOffensiveAgent', second='DefensiveAgentRule', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########
class EnhancedOffensiveAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1.0}
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_carried = my_state.num_carrying

        # --- Food Collection ---
        if food_list:
            features['distance_to_food'] = min(self.get_maze_distance(my_pos, food) for food in food_list)
        else:
            features['distance_to_food'] = 0

        # --- Ghost Analysis ---
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        active_ghosts = [e for e in enemies 
                        if not e.is_pacman 
                        and e.get_position() 
                        and e.scared_timer == 0]

        # Calculate ghost distances
        ghost_distances = [self.get_maze_distance(my_pos, g.get_position()) for g in active_ghosts]
        closest_ghost = min(ghost_distances) if ghost_distances else 9999
        features['active_ghost_distance'] = closest_ghost

        # --- Chase Detection ---
        being_chased = 0
        GHOST_THREAT_RADIUS = 3
        if active_ghosts and food_carried > 0 and closest_ghost <= GHOST_THREAT_RADIUS:
            being_chased = 1
            # Analyze path safety for current action
            next_pos = successor.get_agent_position(self.index)
            features['dead_end_risk'] = self.dead_end_danger(next_pos, game_state)

        # --- Strategic Retreat ---
        features['distance_to_home'] = self.get_maze_distance(my_pos, self.start)
        features['return_urgency'] = food_carried * features['distance_to_home']

        # --- Capsule Prioritization ---
        capsules = self.get_capsules(successor)
        if capsules:
            features['capsule_distance'] = min(
                self.get_maze_distance(my_pos, cap) 
                for cap in capsules
            )
        else:
            features['capsule_distance'] = 0

        # --- Action Penalties ---
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        # Calculate power mode
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        power_mode = 1 if any(e.scared_timer > 0 for e in enemies) else 0
        
        # Dynamic weights
        food_carried = game_state.get_agent_state(self.index).num_carrying
        retreat_weight = -5 * max(0, food_carried - 2)
        
        return {
            'successor_score': 200,
            'distance_to_food': -2,
            'active_ghost_distance': 10 * (1 - power_mode),
            'capsule_distance': -8,
            'return_urgency': retreat_weight,
            'distance_to_home': -0.3,
            'dead_end_risk': -1000,  # Heavy penalty for dangerous paths
            'stop': -150,
            'reverse': -2
        }

    def dead_end_danger(self, position, game_state):
        """Calculate cul-de-sac risk using neighborhood analysis"""
        x, y = position
        walls = game_state.get_walls()
        
        # Count available exits
        exits = 0
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            if not walls[x+dx][y+dy]:
                exits += 1
                
        danger_level = 0
        if exits == 1:
            danger_level = 3 
        elif exits == 2:
            neighbors = [(x+dx, y+dy) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)] 
                       if not walls[x+dx][y+dy]]
            if (abs(neighbors[0][0] - neighbors[1][0]) + 
                abs(neighbors[0][1] - neighbors[1][1])) == 2:
                danger_level = 1 
        return danger_level

# XAVI AGENT
class DefensiveAgent(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.current_patrol_index = 0

    def registerInitialState(self, game_state):
        CaptureAgent.registerInitialState(self, game_state)
        self.current_patrol_index = 0

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        invaders = []

        for enemy in self.get_opponents(game_state):
            enemy_state = game_state.get_agent_state(enemy)
            if enemy_state.is_pacman:
                enemy_pos = enemy_state.get_position()
                if enemy_pos:
                    print(f"Visible invader at: {enemy_pos}")
                    invaders.append(enemy_pos)
                else:
                    noisy_distance = game_state.get_agent_distances()[enemy]
                    print(f"Noisy distance to invader: {noisy_distance}")
                    estimated_pos = self.estimate_enemy_position(game_state, noisy_distance=noisy_distance)
                    if estimated_pos:
                        print(f"Estimated invader position: {estimated_pos}")
                        invaders.append(estimated_pos)

        if self.is_vulnerable(game_state):
            print("Mode: Escape (Vulnerable)")
            return self.escape_from_invaders(game_state, invaders)

        if invaders:
            closest_invader = min(invaders, key=lambda inv: self.get_maze_distance(my_pos, inv))
            distance_to_invader = self.get_maze_distance(my_pos, closest_invader)
            print(f"Closest invader at {closest_invader}, distance: {distance_to_invader}")

            if distance_to_invader <= 5:
                print("Mode: Greedy")
                return self.greedy_move_towards(game_state, closest_invader)
            else:
                print("Mode: A* Search")
                path = self.a_star_search(game_state, my_pos, closest_invader)
                return path[0] if path else self.patrol_own_territory(game_state)

        print("Mode: Patrol")
        return self.patrol_own_territory(game_state)

    def greedy_move_towards(self, game_state, target):
        actions = game_state.get_legal_actions(self.index)
        best_action = min(actions, key=lambda a: self.get_maze_distance(
            self.get_successor(game_state, a).get_agent_state(self.index).get_position(), target))

        return best_action

    def estimate_enemy_position(self, game_state, noisy_distance):
        my_pos = game_state.get_agent_state(self.index).get_position()
        walls = game_state.get_walls()
        positions = [(x, y) for x in range(walls.width) for y in range(walls.height)
                     if not walls[x][y] and abs(self.get_maze_distance(my_pos, (x, y)) - noisy_distance) <= 2]

        return random.choice(positions) if positions else None

    def is_vulnerable(self, game_state):
        return game_state.get_agent_state(self.index).scared_timer > 0

    def escape_from_invaders(self, game_state, invaders):
        actions = game_state.get_legal_actions(self.index)
        if invaders:
            best_action = max(actions, key=lambda a: min(
                self.get_maze_distance(self.get_successor(game_state, a).get_agent_state(self.index).get_position(), inv)
                for inv in invaders
            ))
            return best_action

        return random.choice(actions)

    def a_star_search(self, game_state, start_pos, target_pos):
        frontier = []
        heapq.heappush(frontier, (0, start_pos, []))
        visited = set()

        while frontier:
            _, current_pos, path = heapq.heappop(frontier)

            if current_pos == target_pos:
                return path

            if current_pos in visited:
                continue

            visited.add(current_pos)

            for action in game_state.get_legal_actions(self.index):
                successor_state = self.get_successor(game_state, action)
                successor_pos = successor_state.get_agent_state(self.index).get_position()
                if successor_pos not in visited:
                    new_path = path + [action]
                    cost = len(new_path) + self.get_maze_distance(successor_pos, target_pos)
                    heapq.heappush(frontier, (cost, successor_pos, new_path))

        return []

    def get_successor(self, game_state, action):
        return game_state.generate_successor(self.index, action)

    def patrol_own_territory(self, game_state):
        patrol_points = self.get_territory_transitions(game_state)
        patrol_target = patrol_points[self.current_patrol_index % len(patrol_points)]

        actions = game_state.get_legal_actions(self.index)
        best_action = min(actions, key=lambda a: self.get_maze_distance(
            self.get_successor(game_state, a).get_agent_state(self.index).get_position(), patrol_target))

        if self.get_maze_distance(game_state.get_agent_state(self.index).get_position(), patrol_target) < 2:
            self.current_patrol_index += 1

        print(f"Patrolling to: {patrol_target}")
        return best_action

    def get_territory_transitions(self, game_state):
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        mid_x = width // 2 - (2 if self.red else -2)

        transitions = [(mid_x, y) for y in range(1, height - 1) if not walls[mid_x][y]]
        center_y = height // 2
        transitions.sort(key=lambda pos: abs(pos[1] - center_y))

        return transitions if transitions else [(mid_x, center_y)]