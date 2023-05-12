import numpy as np
import matplotlib.pyplot as plt

def rotate(vector, angle, center):
    translation_matrix = np.array([[1, 0, -center[0]],
                                   [0, 1, -center[1]],
                                   [0, 0, 1]])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    inverse_translation_matrix = np.array([[1, 0, center[0]],
                                           [0, 1, center[1]],
                                           [0, 0, 1]])
    transformed_vector = inverse_translation_matrix @ rotation_matrix @ translation_matrix @ np.append(vector, 1)
    return transformed_vector[:2]

def translate(vector, tx, ty):
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transformed_vector = translation_matrix @ np.append(vector, 1)
    return transformed_vector[:2]

def shear(vector, kx, ky):
    shear_matrix = np.array([[1, kx, 0],
                             [ky, 1, 0],
                             [0, 0, 1]])
    transformed_vector = shear_matrix @ np.append(vector, 1)
    return transformed_vector[:2]

def scale(vector, sx, sy):
    scale_matrix = np.array([[sx, 0, 0],
                             [0, sy, 0],
                             [0, 0, 1]])
    transformed_vector = scale_matrix @ np.append(vector, 1)
    return transformed_vector[:2]

def reflect_x_axis(vector):
    reflection_matrix = np.array([[1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, 1]])
    transformed_vector = reflection_matrix @ np.append(vector, 1)
    return transformed_vector[:2]

def reflect_y_axis(vector):
    reflection_matrix = np.array([[-1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
    transformed_vector = reflection_matrix @ np.append(vector, 1)
    return transformed_vector[:2]

# Test with the given square
square = np.array([[0, 0], [3, 0], [3, 3], [0, 3]])

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 15))

# 1. Rotation about (0, 0)
center = (0, 0)
angle = np.pi / 4  # 45 degrees
rotated_square = [rotate(point, angle, center) for point in square]
axs[0, 0].plot(square[:, 0], square[:, 1], 'b-', label='Original')
axs[0, 0].plot([p[0] for p in rotated_square], [p[1] for p in rotated_square], 'g-', label='Rotation')
axs[0, 0].set_title('Rotation about (0, 0)')
axs[0, 0].legend()

# 2. Translation
tx = -4
ty = 7
translated_square = [translate(point, tx, ty) for point in square]
axs[0, 1].plot(square[:, 0], square[:, 1], 'b-', label='Original')
axs[0, 1].plot([p[0] for p in translated_square], [p[1] for p in translated_square], 'r-', label='Translation')
axs[0, 1].set_title('Translation')
axs[0, 1].legend()

# 3. Shear along x-axis
kx = 3/2
ky = 0
sheared_square = [shear(point, kx, ky) for point in square]
axs[1, 0].plot(square[:, 0], square[:, 1], 'b-', label='Original')
axs[1, 0].plot([p[0] for p in sheared_square], [p[1] for p in sheared_square], 'm-', label='Shear along x-axis')
axs[1, 0].set_title('Shear along x-axis')
axs[1, 0].legend()

# 4. Non-uniform scaling
sx = 2
sy = 4
scaled_square = [scale(point, sx, sy) for point in square]
axs[1, 1].plot(square[:, 0], square[:, 1], 'b-', label='Original')
axs[1, 1].plot([p[0] for p in scaled_square], [p[1] for p in scaled_square], 'c-', label='Non-uniform scaling')
axs[1, 1].set_title('Non-uniform scaling')
axs[1, 1].legend()

# 5. Shearing and scaling combined
shear_kx = 3/2
shear_ky = 2/3
sheared_scaled_square = [shear(scale(point, sx, sy), shear_kx, shear_ky) for point in square]
axs[2, 0].plot(square[:, 0], square[:, 1], 'b-', label='Original')
axs[2, 0].plot([p[0] for p in sheared_scaled_square], [p[1] for p in sheared_scaled_square], 'y-', label='Shear and Scale')
axs[2, 0].set_title('Shear and Scale')
axs[2, 0].legend()

# 6. Rotation about (1, 2)
center = (1, 2)
angle = np.pi / 3  # 60 degrees
rotated_square_2 = [rotate(point, angle, center) for point in square]
axs[2, 1].plot(square[:, 0], square[:, 1], 'b-', label='Original')
axs[2, 1].plot([p[0] for p in rotated_square_2], [p[1] for p in rotated_square_2], 'k-', label='Rotation about (1, 2)')
axs[2, 1].set_title('Rotation about (1, 2)')
axs[2, 1].legend()

# Set common x and y labels for all subplots
fig.text(0.5, 0.04, 'X-axis', ha='center')
fig.text(0.04, 0.5, 'Y-axis', va='center', rotation='vertical')

# Adjust spacing between subplots
fig.tight_layout()

# Show the plots
plt.show()