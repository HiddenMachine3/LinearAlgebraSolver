import numpy as np
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


    @staticmethod
    def proj(v1:np.ndarray, v2:np.ndarray):
        return v1*(np.dot(v1,v2)/np.dot(v1,v1))

    @staticmethod
    def orthonormal_basis_set(x:np.ndarray):
        v = x.copy()
        print(v)
        for i in range(1,3):
            for j in range(i-1):
                v[i] = np.subtract(v[i],MatrixSolver.proj(v[j],x[i]))


        print(v)
        return v

    @staticmethod
    def mult(a:np.ndarray, b:np.ndarray):
        resElements = np.ndarray((3, 3))

        for i in range(3):
            for j in range(3):
                sum = 0
                for k in range(3):
                    sum += a[i,k] * b[k,j]
                resElements[i,j] = sum
        return resElements

    def orthogonalDiagonalization(self):
        eig_values,eig_vecs = np.linalg.eig(self.matrix)
        D = np.diag(eig_values)
        Q, R = np.linalg.qr(A)
        print(MatrixSolver.orthonormal_basis_set(eig_vecs))

        print("Q:")
        MatrixSolver(Q).display()
        print("R:")
        MatrixSolver(R).display()
        print("D:")
        MatrixSolver(D).display()
        print("RHS : ")

        res = np.dot(np.dot(Q, A), np.transpose(Q))
        MatrixSolver(res).display()


        print("R:")
        R= MatrixSolver.mult(np.transpose(Q),A)
        print(R)




if __name__ == "__main__":
    A = np.array([
        [2,-2,4],
        [-2, -1, -2],
        [4, -2, 5]
    ])

    print ("A:")
    solver = MatrixSolver(A)
    solver.display()
    solver.orthogonalDiagonalization()


