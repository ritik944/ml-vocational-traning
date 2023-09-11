
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:06:39 2023

@author: shubhamjuneja
"""
import random

# Number of simulations
num_simulations = 1000  # You can adjust this number for more or fewer simulations

# Initialize variables to keep track of wins and losses
total_wins = 0
total_losses = 0

# Simulate the dice game
for _ in range(num_simulations):
    # Roll two dice
    die1 = random.randint(1, 6)
    die2 = random.randint(1, 6)
    
    # Calculate the sum of the two dice
    total = die1 + die2
    
    # Check if the sum is less than or equal to 7
    if total <= 7:
        total_wins += 1
    else:
        total_losses += 1

# Calculate the expected value
expected_value = (total_wins / num_simulations) * 1 + (total_losses / num_simulations) * (-1)

print(f"Simulated Expected Value after {num_simulations} simulations: {expected_value:.4f}")
