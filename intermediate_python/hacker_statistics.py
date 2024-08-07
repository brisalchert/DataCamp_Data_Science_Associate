import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(123)

# Initialize random_walk
random_walk = [0]

# Complete the ___
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

    # append next_step to random_walk
    random_walk.append(step)

# Plot the random walk
plt.plot(random_walk)
plt.show()
