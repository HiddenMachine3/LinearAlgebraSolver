import math as m
import numpy as np

print("Pauli-X")
px = np.array([[0, 1], [1, 0]], dtype=int)

up = np.array([[1], [0]])

down = np.array([[0], [1]])

Out_up = px @ up
Out_down = px @ down  # does the matrix multiplcation

print("Out_up :\n", Out_up)
print("Out_down : \n", Out_down)


def matrix_multiply(matrix1, matrix2):
    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape
    assert cols1 == rows2, "Invalid matrix dimensions for multiplication"
    result = np.zeros((rows1, cols2), dtype=int)
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i, j] += matrix1[i, k] * matrix2[k, j]
    return result


px = np.array([[0, 1], [1, 0]], dtype=int)
up = np.array([[1], [0]])
down = np.array([[0], [1]])

Out_up = matrix_multiply(px, up)
Out_down = matrix_multiply(px, down)

print("Out_up :\n", Out_up)
print("Out_down :\n", Out_down)

print("\n\nPauli-Z")

px = np.array([[1, 0], [0, -1]], dtype=int)

up = np.array([[1], [0]])

down = np.array([[0], [1]])

Out_up = px @ up
Out_down = px @ down

print("Out_up :\n", Out_up)
print("Out_down : \n", Out_down)

print("\n\nC-NOT")

px = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=int)

Qubit_00 = np.array([[1], [0], [0], [0]])
Qubit_01 = np.array([[0], [1], [0], [0]])
Qubit_10 = np.array([[0], [0], [1], [0]])
Qubit_11 = np.array([[0], [0], [0], [1]])

Out_00 = px @ Qubit_00
Out_01 = px @ Qubit_01
Out_10 = px @ Qubit_10
Out_11 = px @ Qubit_11

print("Out_00 :\n", Out_00)
print("Out_01 : \n", Out_01)
print("Out_10 : \n", Out_10)
print("Out_11 : \n", Out_11)

print("\n\nHadamard")

px = 1 / m.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=float)

up = np.array([[1], [0]])

down = np.array([[0], [1]])

Out_up = px @ up
Out_down = px @ down

print("Out_up :\n", Out_up)
print("Out_down : \n", Out_down)
