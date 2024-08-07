# ----------------------------------------------------------------------------------------------------------------------
# hacker_statistics.py
#
# Python script for simulating a game about climbing the Empire State Building. The player climbs the building by
# rolling a die. On a 1 or 2, the player descends one step (unless they are at the bottom). On a 3, 4, or 5, the player
# ascends one step. On a 6, the player rolls a die and ascends the number of steps on the die. For each roll (excluding
# the additional rolls for rolling a 6), there is a 0.1% chance that the player will fall to the bottom.
#
# The player rolls 100 times per game, and the random walk of their step numbers is recorded. These random walks are
# performed 10,000 times, and the results are used to plot a histogram of the endpoints of each walk. Then, the
# probability of reaching step 60 on any given walk is calculated and printed out.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(123)

# Initialize list of final step values
all_walks = []

# Perform 10,000 random walks
for i in range(10000):
    # Initialize random_walk
    random_walk = [0]

    # Perform random walk
    for x in range(100):
        # Set step: last element in random_walk
        step = random_walk[-1]

        # Roll the dice
        dice = np.random.randint(1, 7)

        # Determine next step
        if dice <= 2:
            # Do not allow negative steps
            step = max(step - 1, 0)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1, 7)

        # Implement clumsiness
        if np.random.rand() <= 0.001:
            step = 0

        # append next_step to random_walk
        random_walk.append(step)

    # Add the walk to all_walks
    all_walks.append(random_walk)

# Convert final_steps to numpy array
np_all_walks = np.array(all_walks)

# Transpose np_final_steps
np_all_walks_transposed = np.transpose(np_all_walks)

# Select the last row of the transposed walks (endpoints)
ends = np_all_walks_transposed[-1, :]

# Plot histogram of endpoints
plt.hist(ends)
plt.show()

# Calculate chance of reaching step 60
count = 0

for endpoint in ends:
    if endpoint >= 60:
        count += 1

probability = count / len(ends)

print("Chance of reaching step 60: " + str(probability * 100) + "%")
