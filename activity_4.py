import numpy as np
from activity_3 import Matrix
from activity_3 import MatrixError


class MatrixSolver:
    def __init__(self, matrix: np.ndarray, r: int = None, c: int = None):
        if matrix.dtype != float:
            self.matrix = matrix.astype(float)
        else:
            self.matrix = matrix
        if r is None or c is None:
            self.r, self.c = matrix.shape
        else:
            self.r = r
            self.c = c

    def swap_rows(self, r1: int, r2: int):
        # assigning the sub-matrix of the 2 rows to the reverse order or it
        self.matrix[[r1, r2], :] = self.matrix[[r2, r1], :]
        print(f"R{r1} <-> R{r2}")
        self.display()

    def sort(self):
        """
        sorting the matrix so that the rows with more leading 0s go down
        :return: None
        """
        for h in range(self.r - 1, -1, -1):
            most_leading_zeroes_index = 0
            most_leading_zeroes = 0

            for i in range(0, h + 1):
                num_leading_zeroes = 0
                # let's count the number of leading zeroes
                for j in range(0, self.c):
                    # if we find an element that isn't a zero, we stop incrementing the number of leading zeroes
                    if self.matrix[i][j] != 0:
                        break
                    else:
                        num_leading_zeroes += 1

                if num_leading_zeroes > most_leading_zeroes:
                    most_leading_zeroes = num_leading_zeroes
                    most_leading_zeroes_index = i

            if most_leading_zeroes > 0:
                if h != most_leading_zeroes_index:  # swapping with same row is redundant
                    self.swap_rows(h, most_leading_zeroes_index)
            else:  # if the most leading zeroes in any row is not more than 0, then we don't need to sort
                break

    def check_sig(self):
        """
         edge case: not enough info i.e., number of significant coefficients in each row > number of rows
         which means you can't find the values of all the variables
         "Infinitely many solutions"
        :return: True if (more vars > number of rows with all coeff=0) else False
        """
        non_sig = 0  # variable to keep track of the number of rows with all coeff=0
        for i in range(0, self.r):
            j = 0
            while j < self.c:
                if self.matrix[i][j] != 0:
                    break
                j += 1
            # do we have all coeff=0 in this row, then j would end on (c-1)
            if j == self.c:
                non_sig += 1

        if (self.r - non_sig) < (self.c):
            print("sig<vars Infinite solutions")
            return True

    def display(self):
        print(np.array_str(self.matrix, max_line_width=np.inf, precision=3,
                           suppress_small=True))  # , precision=3, suppress_small=True))

    def add(self, place: int, r1: int, k: float, r2: int):
        cols = self.matrix.shape[1]
        for j in range(0, cols):
            self.matrix[place][j] = self.matrix[r1][j] + k * self.matrix[r2][j]

    def mult(self, r1: int, k: float):
        cols = self.matrix.shape[1]
        for j in range(0, cols):
            self.matrix[r1][j] = self.matrix[r1][j] * k

    @staticmethod
    def minor(matrix: np.ndarray, i: int, j: int):
        elements = np.delete(matrix, i, axis=0)
        elements = np.delete(elements, j, axis=1)

        return elements

    @staticmethod
    def determinant(matrix: np.ndarray):
        # print(a)
        if matrix.size == 1:
            return matrix[0][0]

        rows, cols = matrix.shape
        if rows == cols:
            if rows == 1:
                return matrix[0][0]
            elif rows == 2:
                return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
            else:
                det = 0
                for j in range(cols):
                    det += (-1 * ((j % 2) * 2 - 1)) * matrix[0][j] * MatrixSolver.determinant(
                        MatrixSolver.minor(matrix, 0, j))
                return det
        else:
            raise MatrixError("Cannot compute determinant for non square matrix")

    def gauss_jordan_inverse(self):
        n = self.matrix.shape[0]  # Number of rows in the matrix
        self.matrix = np.hstack((self.matrix, np.eye(n)))
        self.display()

        # 1
        for i in range(0, self.r - 1):
            self.sort()
            for j in range(0, self.c):
                # is variable != 0? then make everything below this leading coefficient equal to zero
                if self.matrix[i][j] != 0:
                    for I in range(i + 1, self.r):
                        if self.matrix[I][j] != 0:  # it has found a number !=0 below it
                            # rowI = rowI + [rowi * (1 / element below it) * -leading coefficient(of i))]
                            print(f"R{I} = R{I} + (-{self.matrix[I][j]} /{self.matrix[i][j]})*R{i}")
                            self.add(I, I, -1 * self.matrix[I][j] / self.matrix[i][j], i)
                            self.display()
                    break  # break from the inner loop

        self.sort()

        if self.check_sig():
            return

        print("1. converted to row echelon form\n")

        # 2.a
        for i in range(0, self.r):
            for j in range(0, self.c):
                if self.matrix[i][j] != 0:  # checking for non-zero leading element
                    if self.matrix[i][j] != 1:  # if leading element equal to 1, we don't do anything to that row
                        print(f"R{i}=(1/{self.matrix[i][j]})*R{i}")
                        self.mult(i, 1.0 / self.matrix[i][j])  # we want the leading element to equal 1
                        self.display()
                    break

        print("\n 2.a all non-zero leading numbers = 1")
        self.display()

        # 2.b
        for i in range(1, self.r):
            for j in range(0, self.c):
                if self.matrix[i][j] == 1:  # if the current element is a leading 1
                    for I in range(i - 1, -1, -1):  # we go on making the elements above equal to zero
                        if (self.matrix[I][j] != 0):
                            print(f"R{I} = R{I} + (-{self.matrix[I][j]} /{self.matrix[i][j]})*R{i}")
                            self.add(I, I, self.matrix[I][j] * -1, i)
                            self.display()

                    break

        inverse_matrix = self.matrix[:, n:]
        return inverse_matrix

    def LUdecompose(self):
        try:
            det = MatrixSolver.determinant(self.matrix)
            if (det == 0):
                raise MatrixError("Non invertible")
        except MatrixError as error:
            raise MatrixError("Cannot apply LU decomposition. reason : " + error.args[0])

        L = np.identity(self.matrix.shape[0])
        print(np.array_str(L))
        for i in range(0, self.r):
            for j in range(0, self.c):
                # is variable != 0? then make everything below this leading coefficient equal to zero
                if self.matrix[i][j] != 0:
                    for I in range(i + 1, self.r):
                        if self.matrix[I][j] != 0:  # it has found a number !=0 below it
                            # rowI = rowI + [rowi * (1 / element below it) * -leading coefficient(of i))]
                            print(f"R{I} = R{I} + (-{self.matrix[I][j]} /{self.matrix[i][j]})*R{i}")
                            coeff = -1 * self.matrix[I][j] / self.matrix[i][j]
                            self.add(I, I, coeff, i)
                            L[I, i] = -coeff

                            print("U: ", end="")
                            self.display()

                            print("L: ", end="")
                            print(np.array_str(L))
                    break  # break from the inner loop
        return L, self.matrix

    @staticmethod
    def solveLy(L: np.ndarray, b: np.ndarray):  # Ly = b
        y = np.empty(b.shape)
        y[0, 0] = b[0, 0] / L[0, 0]
        r, c = b.shape
        for i in range(1, r):
            # y[i, 0] = (b[i] - y[i - 1, 0] * L[i, 0]) / L[i, i]
            # yi = (bi - (L[i,0]*y0 + L[i,1]*y1 + ...))/L[i,i]
            y[i, 0] = b[i]

            for j in range(0, i):
                y[i, 0] -= L[i, j] * y[j, 0]

            y[i, 0] /= L[i, i]

        return y

    @staticmethod
    def solveUx(U: np.ndarray, y: np.ndarray):
        r, c = U.shape
        x = np.empty(y.shape)
        x[r - 1, 0] = y[r - 1, 0] / U[r - 1, c - 1]

        for i in range(r - 2, -1, -1):
            # xi = (yi - (U[i,3]*x3 +U[i,2]*x2 +U[i,1]*x1))/U[i,i]
            x[i, 0] = y[i, 0]

            for j in range(r - 1, i, -1):
                x[i, 0] -= U[i, j] * x[j, 0]

            x[i, 0] /= U[i, i]

        return x


"""

15
-9
-23
33
11

1 0 0   y0  b0
2 1 0   y1  b1
3 2 1   y2  b2

y0 = b0/L[0,0]
y1 = (b1 - L[1,0]*y0)/L[1,1]
y2 = (b2 - L[2,0]*y0 - L[2,1]*y1)/L[2,2]
yi = (bi - (L[i,0]*y0 + L[i,1]*y1 + ...))/L[i,i]

1 2 3   x0  y0
0 1 2   x1  y1
0 0 1   x2  y2

x2 = y2/U[2,2]
x1 = (y1 - U[1,2]*x2)/U[1,1]            2
x0 = (y0 - U[0,2]*x2 - U[0,1]*x1)/U[0,0]        2   1
xi = (yi - (U[i,3]*x3 +U[i,2]*x2 +U[i,1]*x1))/U[i,i]        3   2   1
        j=r-2


[[ 6.8 -1.  -1.6]
 [-0.2 -1.   0.4]
 [-2.8  1.   0.6]]
 
1 1 2
1 4 -1
3 -2 1
"""

if __name__ == "__main__":
    A = Matrix(np.array([
        [1, 1, 2, 1, 2],
        [1, 4, -1, 1, 2],
        [3, -2, 1, 1, 2],
        [6, -2, 1, 1, 2],
        [7, -2, 1, 1, 3],
    ]))
    b = Matrix(np.array([
        [15],
        [-9],
        [-23],
        [33],
        [11]
    ]))

    print("A : ")
    A.display()
    print("b : ")
    b.display()

    print("A. Using gauss jordan elimination method to solve for x in Ax = b:")
    print("We will use gauss jordan elimination to deduce the inverse of A:")
    print(
        "first : Constructing an augmented matrix with A and I(I is an identity matrix with rows and cols same as A):")

    inv = Matrix(MatrixSolver(A.elements.copy()).gauss_jordan_inverse())

    print("\n\n A inverse is : ")
    inv.display()

    x = Matrix.mult(inv, b)
    print("x = A^-1 * b")
    print("x : ")
    x.display()

    print("\n\n\nB. Using LU decomposition to solve for Ax = b")
    try:
        L, U = MatrixSolver(A.elements.copy()).LUdecompose()

        L = Matrix(L)
        U = Matrix(U)

        print("\n\nSuccessfully decomposed A in to L and U:")
        print("L:", end="")
        L.display()
        print("U:", end="")
        U.display()
        print("b : ")
        b.display()

        print("\n\nAx = b ==> LUx =b ==> Ux=y ==> Ly = b")
        y = Matrix(MatrixSolver.solveLy(L.elements, b.elements))
        print("Solving Ly=b\ny:", end="")
        y.display()
        # cross checking: y=(L^-1)b
        # y = Matrix.mult(Matrix.inverse(L),b)
        # y.display()

        x = Matrix(MatrixSolver.solveUx(U.elements, y.elements))
        print("Solving Ux=y\nx:", end="")
        x.display()

        # cross checking:         x=(U^-1)y
        # x = Matrix.mult(Matrix.inverse(U), y)
        # x.display()


    except MatrixError as error:
        print(error.args[0])
