from __future__ import annotations

import numpy as np


class MatrixError(Exception):
    def __init__(self, message):
        super().__init__(message)


# +addition, +scalar multiplication, +matrix multiplica-tion, +transpose, +inverse.


class Matrix:
    def __init__(self, elements: np.ndarray, r=None, c=None):
        if elements.dtype != float:
            self.elements = elements.astype(float)
        else:
            self.elements = elements

        if r is None or c is None:
            self.r, self.c = elements.shape
        else:
            self.r = r
            self.c = c

    def display(self):
        # print(np.array_str(self.elements))  # , precision=3, suppress_small=True))
        print(np.array_str(self.elements, max_line_width=np.inf, precision=3, suppress_small=True))

    def transpose(self):
        r = self.c
        c = self.r
        resElements = np.ndarray((r, c))

        for i in range(r):
            for j in range(c):
                resElements[i][j] = self.elements[j][i]

        return Matrix(resElements, r, c)

    @staticmethod
    def scalar_mult(k: float, a: Matrix):
        elements = np.empty(a.elements.shape)
        for i in range(a.r):
            for j in range(a.c):
                elements[i][j] = a.elements[i][j] * k
        return Matrix(elements)

    @staticmethod
    def minor(a: Matrix, i: int, j: int):
        elements = np.delete(a.elements, i, axis=0)
        elements = np.delete(elements, j, axis=1)

        return Matrix(elements)

    @staticmethod
    def determinant(a: Matrix):
        # print(a)
        if a.elements.size == 1:
            return a.elements[0][0]

        rows, cols = a.elements.shape
        if rows == cols:
            if rows == 1:
                return a.elements[0][0]
            elif rows == 2:
                return a.elements[0][0] * a.elements[1][1] - a.elements[1][0] * a.elements[0][1]
            else:
                det = 0
                for j in range(cols):
                    det += (-1 * ((j % 2) * 2 - 1)) * a.elements[0][j] * Matrix.determinant(Matrix.minor(a, 0, j))
                return det
        else:
            raise MatrixError("Cannot compute determinant for non square matrix")

    @staticmethod
    def add(a: Matrix, b: Matrix):
        if a.r == b.r and a.c == b.c:
            r = a.r
            c = a.c
            resElements = np.ndarray((r, c))

            for i in range(r):
                for j in range(c):
                    resElements[i][j] = a.elements[i][j] + b.elements[i][j]

            return Matrix(resElements, r, c)
        else:
            raise MatrixError("Number of columns and rows of 1st matrix not the same as in 2nd matrix")

    @staticmethod
    def sub(a: Matrix, b: Matrix):
        if a.r == b.r and a.c == b.c:
            r = a.r
            c = a.c
            resElements = np.ndarray((r, c))

            for i in range(r):
                for j in range(c):
                    resElements[i][j] = a.elements[i][j] - b.elements[i][j]

            return Matrix(resElements, r, c)
        else:
            raise MatrixError("Number of columns and rows of 1st matrix not the same as in 2nd matrix")

    @staticmethod
    def mult(a: Matrix, b: Matrix):
        if a.c == b.r:
            r = a.r
            c = b.c
            resElements = np.ndarray((r, c))

            for i in range(r):
                for j in range(c):
                    sum = 0
                    for k in range(a.c):
                        sum += a.elements[i][k] * b.elements[k][j]
                    resElements[i][j] = sum

            return Matrix(resElements, r, c)
        else:
            raise MatrixError("Number of columns of 1st matrix not equal to number of rows in 2nd matrix")

    @staticmethod
    def cofactor_matrix(a: Matrix):
        r, c = a.r, a.c
        elements = np.ndarray(a.elements.shape)
        for i in range(r):
            for j in range(c):
                elements[i][j] = (-1 * (((i + j) % 2) * 2 - 1)) * Matrix.determinant(Matrix.minor(a, i, j))

        return Matrix(elements)

    @staticmethod
    def adjoint(a: Matrix):
        return Matrix.transpose(Matrix.cofactor_matrix(a))

    @staticmethod
    def inverse(a: Matrix):
        det = Matrix.determinant(a)

        if (det == 0):
            raise MatrixError("Matrix is not invertible")

        return Matrix.scalar_mult(1.0 / det, Matrix.adjoint(a))


"""
3 0 0 3 0
-3 0 -2 0 0
0 -1 0 0 -3
0 0 0 3 3
0 -1 2 0 1


1 2 0 0 3
7 4 0 -1 8
3 0 0 2 2
8 0 -1 1 -3
1 -1 2 0 1
"""

A = Matrix(np.array([
    [3, 0, 0, 3, 0],
    [-3, 0, -2, 0, 0],
    [0, -1, 0, 0, -3],
    [0, 0, 0, 3, 3],
    [0, -1, 2, 0, 1]
]))
B = Matrix(np.array([
    [1, 2, 0, 0, 3],
    [7, 4, 0, -1, 8],
    [3, 0, 0, 2, 2],
    [8, 0, -1, 1, -3],
    [1, -1, 2, 0, 1]
]))


if __name__ == "__main__":
    print("A:")
    A.display()
    print("B:")
    B.display()

    print("1. AB")
    Matrix.mult(A, B).display()

    print("2. A + B")
    Matrix.add(A, B).display()

    print("3. 2A + 5B")
    Matrix.add(Matrix.scalar_mult(2, A), Matrix.scalar_mult(5, B)).display()

    print("4. A transpose")
    A.transpose().display()

    print("5. \n")
    print("A^−1:")
    try:
        Matrix.inverse(A).display()
    except MatrixError as e:
        print(e.args[0])
    print("B^−1:")
    try:
        Matrix.inverse(B).display()
    except MatrixError as e:
        print(e.args[0])
    print("(AB)^−1:")
    try:
        Matrix.inverse(Matrix.mult(A, B)).display()
    except MatrixError as e:
        print(e.args[0])
