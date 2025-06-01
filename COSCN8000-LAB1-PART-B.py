import numpy as np

# Step 2: Solve System 1
# Equations:
# 2x1 + 3x2 - 4x3 = 6
# x1 - 4x2 + 0x3 = 8

A1 = np.array([[2, 3, -4],
               [1, -4, 0]])
b1 = np.array([6, 8])

# Use least squares to solve
x1_solution, residuals1, rank1, s1 = np.linalg.lstsq(A1, b1, rcond=None)
print("System 1 Solution (x1, x2, x3):", x1_solution)