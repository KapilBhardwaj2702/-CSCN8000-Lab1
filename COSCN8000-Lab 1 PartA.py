import numpy as np

# Step 2: Create an array that starts from 1, ends at 20, incremented by 3
array_step2 = np.arange(1, 21, 3)
print("Step 2 - Array with step 3:\n", array_step2)

# Step 3: Create a new array of shape (3,) with random numbers between 0 and 1
array_step3 = np.random.rand(3)
print("\nStep 3 - Random array of shape 3:\n", array_step3)

# Step 4: Create a 2D array and slice it
array_2d = np.array([[10, 20, 45], [30, 12, 16], [42, 17, 56]])

# First two rows
first_two_rows = array_2d[:2, :]

# Last two rows
last_two_rows = array_2d[1:, :]

print("\nStep 4 - Original 2D Array:\n", array_2d)
print("First two rows:\n", first_two_rows)
print("Last two rows:\n", last_two_rows)


# Step 5: Create two 2x2 arrays and demonstrate stacking and splitting
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Vertical stack
vertical_stack = np.vstack((a, b))

# Horizontal stack
horizontal_stack = np.hstack((a, b))

# Split arrays vertically and horizontally
split_vertical = np.vsplit(vertical_stack, 2)
split_horizontal = np.hsplit(horizontal_stack, 2)

print("\nStep 5 - Vertical Stack:\n", vertical_stack)
print("Horizontal Stack:\n", horizontal_stack)
print("Vertical Split:\n", split_vertical)
print("Horizontal Split:\n", split_horizontal)


# Step 6: Matrix multiplication check
X = np.array([[5, 7, 2], [4, 5, 6], [7, 4 ,2]])
Y = np.array([[4, 2], [6, 2], [4, 2]])

# Check if multiplication is possible
if X.shape[1] == Y.shape[0]:
    result = np.dot(X, Y)
    print("\nStep 6 - Matrix Multiplication Result:\n", result)
else:
    print("\nStep 6 - Cannot multiply X and Y due to shape mismatch")


# Step 7: Shape, dimensions, and reshape
x = np.array([2, -1, -8])
y = np.array([3, 1, -2])

print("\nStep 7 - Original x shape:", x.shape)
print("Original x dimensions:", x.ndim)

# Reshape x and y to (3,1)
x_reshaped = x.reshape((3, 1))
y_reshaped = y.reshape((3, 1))

print("Reshaped x:\n", x_reshaped)
print("Reshaped y:\n", y_reshaped)
print("x dimensions after reshaping:", x_reshaped.ndim)
print("y dimensions after reshaping:", y_reshaped.ndim)


# Step 8: Broadcasting example with subtraction and multiplication
matrix_3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
vector_1x3 = np.array([1, 0, 1])

# Broadcasting subtraction and multiplication
broadcast_sub = matrix_3x3 - vector_1x3
broadcast_mul = matrix_3x3 * vector_1x3

print("\nStep 8 - Original Matrix:\n", matrix_3x3)
print("Vector:\n", vector_1x3)
print("Broadcast Subtraction:\n", broadcast_sub)
print("Broadcast Multiplication:\n", broadcast_mul)
