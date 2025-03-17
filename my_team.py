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
import time

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
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

class ReflexCaptureAgent(CaptureAgent):

    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        #start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        #print(f'eval time for agent {self.index}: {time.time() - start:.4f}')
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
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

import time
import random

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Offensive agent that collects food and returns home strategically.
    """

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return None 

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        carried_food = my_state.num_carrying  # Number of food items the agent is carrying
        total_food = len(self.get_food(game_state).as_list())  # Food available on the map
        return_food = max(1, total_food // 5)  # 1/5 of the total food, at least 1

        # Determine the team's border (midline of the map)
        boundary_x = (game_state.data.layout.width // 2) - (1 if self.red else 0)
        border_positions = [(boundary_x, y) for y in range(game_state.data.layout.height) 
                            if not game_state.has_wall(boundary_x, y)]
        min_dist_to_border = min([self.get_maze_distance(my_pos, p) for p in border_positions]) # Distance from agent to border

        # If the agent carries enough food, or is close to the border with some food, return home
        if carried_food >= return_food or (carried_food >= 2 and min_dist_to_border <= 2):
            # Select the action that moves the agent closest to the border
            best_action = min(actions, key=lambda a: self.get_maze_distance(
                self.get_successor(game_state, a).get_agent_state(self.index).get_position(),
                min(border_positions, key=lambda p: self.get_maze_distance(my_pos, p))
            ))
            return best_action
        
        # If the agent is not ready to return home, go to the nearest food
        food_list = self.get_food(game_state).as_list()
        if food_list:
            # Select the action that moves the agent closest to the nearest food
            best_action = min(actions, key=lambda a: self.get_maze_distance(
                self.get_successor(game_state, a).get_agent_state(self.index).get_position(),
                min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
            ))
            return best_action
        
        # If no other strategy applies, choose a random action
        return random.choice(actions)




class DefensiveReflexAgent(ReflexCaptureAgent):
    def register_initial_state(self, game_state):
        """
        Initializes the agent's starting position and calculates the boundary of its territory.
        """
        super().register_initial_state(game_state)

        # Determine the maximum boundary (without crossing into the enemy's side)
        self.boundary_x = (game_state.data.layout.width // 2) - (2 if self.red else -2)

        # Set Y-axis limits (avoiding edges to prevent getting stuck)
        y_min = 4  # Leave a 4-cell margin from the top
        y_max = game_state.data.layout.height - 5  # Leave a 4-cell margin from the bottom

        # Find valid positions along the boundary where the agent can patrol
        patrol_positions = [
            (self.boundary_x, y) for y in range(y_min, y_max + 1)
            if not game_state.has_wall(self.boundary_x, y)
        ]

        # Select two patrol points: one in the upper part and another in the lower part of the valid area
        if len(patrol_positions) >= 2:
            self.patrol_points = [patrol_positions[0], patrol_positions[-1]]
        else:
            self.patrol_points = patrol_positions  # If there are fewer than two, use what is available

        self.patrol_index = 0  # Start at the first patrol point

    def choose_action(self, game_state):
        """
        Patrols the boundary without crossing it. If it detects invaders, it chases them.
        """
        actions = game_state.get_legal_actions(self.index)

        # Get the agent's current position
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Identify enemies that have entered our territory
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]

        # If there are invaders, move towards the closest one
        if invaders:
            best_action = None
            min_distance = float('inf')
            for action in actions:
                successor = self.get_successor(game_state, action)
                new_pos = successor.get_agent_state(self.index).get_position()
                dist = min(self.get_maze_distance(new_pos, inv.get_position()) for inv in invaders)

                if dist < min_distance:
                    min_distance = dist
                    best_action = action

            return best_action
        
        # If there are no invaders, continue patrolling the boundary
        return self.patrol(game_state, my_pos, actions)

    def patrol(self, game_state, my_pos, actions):
        """
        If no enemies are detected, the agent moves up and down along the boundary.
        """
        target_pos = self.patrol_points[self.patrol_index]

        # If the agent has reached the patrol point, switch to the other patrol point
        if my_pos == target_pos:
            self.patrol_index = 1 - self.patrol_index  # Alternate between 0 and 1
            target_pos = self.patrol_points[self.patrol_index]

        # Select the best action to move towards the next patrol point
        best_action = None
        min_dist = float('inf')
        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            dist = self.get_maze_distance(new_pos, target_pos)

            if dist < min_dist:
                min_dist = dist
                best_action = action

        # If no specific patrol movement is determined, choose a random action
        return best_action if best_action else random.choice(actions)

