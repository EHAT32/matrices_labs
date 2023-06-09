import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt

class TextBlock:
    def __init__(self, rows):
        assert isinstance(rows, list)
        self.rows = rows
        self.height = len(self.rows)
        self.width = max(map(len,self.rows))
        
    @classmethod
    def from_str(_cls, data):
        assert isinstance(data, str)
        return TextBlock( data.split('\n') )
        
    def format(self, width=None, height=None):
        if width is None: width = self.width
        if height is None: height = self.height
        return [f"{row:{width}}" for row in self.rows]+[' '*width]*(height-self.height)
    
    @staticmethod
    def merge(blocks):
        return [" ".join(row) for row in zip(*blocks)]
    
class Matrix:
    """Общий предок для всех матриц."""
    @property
    def shape(self):
        # raise NotImplementedError
        return (self.height, self.width)
    
    @property
    def dtype(self):
        raise NotImplementedError
    
    @property 
    def width(self):
        return self.shape[1]
    
    @property 
    def height(self):
        return self.shape[0]    
        
    def __repr__(self):
        """Возвращает текстовое представление для матрицы."""
        text = [[TextBlock.from_str(f"{self[r,c]}") for c in range(self.width)] for r in range(self.height)]
        width_el = np.array(list(map(lambda row: list(map(lambda el: el.width, row)), text)))
        height_el = np.array(list(map(lambda row: list(map(lambda el: el.height, row)), text)))
        width_column = np.max(width_el, axis=0)
        width_total = np.sum(width_column)
        height_row = np.max(height_el, axis=1)
        result = []
        for r in range(self.height):
            lines = TextBlock.merge(text[r][c].format(width=width_column[c], height=height_row[r]) for c in range(self.width))
            for l in lines:
                result.append(f"| {l} |")
            if len(lines)>0 and len(lines[0])>0 and lines[0][0]=='|' and r<self.height-1:
                result.append(f'| {" "*(width_total+self.width)}|')
        return "\n".join(result)
    
    def empty_like(self, width=None, height=None):
        raise NotImplementedError
    
    def __getitem__(self, key):
        raise NotImplementedError
    
    def __setitem__(self, key, value):
        raise NotImplementedError
        
    def __add__(self, other):
        if isinstance(other, Matrix):
            assert self.width==other.width and self.height==other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r,c] = self[r,c] + other[r,c]
            return matrix
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, Matrix):
            assert self.width==other.width and self.height==other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r,c] = self[r,c] - other[r,c]
            return matrix
        return NotImplemented

    def __mul__(self, other):
        return self.__matmul__(other)
    def __rmul__(self, other):
        return self.__matmul__(other)
    
    def __matmul__(self, other):
        # add matmul for Matrix and number
        if isinstance(other, float or int or Fraction):
            matrix = self.empty_like()
            for c in range(self.height):
                for r in range(self.width):
                    matrix[r,c] = other*self[r,c]
            return matrix 
        if isinstance(other, Matrix):
            assert self.width==other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(other.width):
                    acc = None
                    for k in range(self.width):
                        add = self[r,k]*other[k,c]
                        acc = add if acc is None else acc+add
                    matrix[r,c] = acc
            return matrix
        return NotImplemented
    
    def inverse(self):
        raise NotImplementedError
        
        
    def invert_element(self, element):
        if isinstance(element, float):
            return 1/element
        if isinstance(element, Fraction):
            return 1/element
        if isinstance(element, Matrix):
            return element.inverse()
        raise TypeError
    
#-------ex 1, 3-------------------------

    def __truediv__(self ,other):
        E = FullMatrix.identity(self.height)
        return self*other.solve(E)
    def __rtruediv__(self, other):
        E = FullMatrix.identity(self.height)
        return other*self.solve(E)

    def clone(self):
        copy = np.copy(self.data)
        return FullMatrix(copy)

    def lu(self):
        N = self.height

        if isinstance(self, SymmetricMatrix):
            return self.ldl()
        else:    
            lu_mat = self.clone()
            for i in range(N):
                for j in range(i + 1, N):
                            lu_mat[j, i] /= lu_mat[i, i]
                            for k in range(i + 1, N):
                                lu_mat[j, k] -= lu_mat[j, i] * lu_mat[i, k]
            return lu_mat
    
#-----ex 2------------------------------

    def dissamble_lu(self):
        '''use only for the result of lu'''
        N = self.height

        # copy = self.clone()

        lower = FullMatrix.zero(N, N, Fraction(0))
        upper = FullMatrix.zero(N, N, Fraction(0))

        for i in range(N):
            lower[i, i] = 1
            for j in range(i + 1, N):
                lower[j, i] = self[j, i]

        for i in range(N):
            upper[i, i] = self[i, i]
            for j in range(i + 1, N):
                upper[i, j] = self[i, j]
        return lower, upper
        
    def lu_disassemb(self):
        N = self.height

        l = self.lu()
        
        for i in range(N):
                if type(l[i,i]) == FullMatrix:
                    l[i,i] = FullMatrix.zero(*l[i,i].shape, Fraction(0))
                    for li in range(min(l[i,i].shape)):
                        (l[i,i])[li,li] = Fraction(1)
                    for j in range(i+1, N):
                        l[i,j] = FullMatrix.zero(*l[i,j].shape, Fraction(0))
                else:
                    l[i, i] = 1
                    for j in range(i+1, N):
                        l[i,j] = 0
        u = self.lu()
        
        for i in range(N):
                for j in range(i):
                    if type(u[i,j]) == FullMatrix:
                        u[i,j] = FullMatrix.zero(*u[i,j].shape, Fraction(0))
                    else:
                        u[i,j] = 0

        return l, u
#compare with Grisha
    def get_l(self):
        l = self.lu()
        
        for i in range(l.shape[0]):
                if type(l[i,i]) == FullMatrix:
                    l[i,i] = FullMatrix.zero(*l[i,i].shape, Fraction(0))
                    for li in range(min(l[i,i].shape)):
                        (l[i,i])[li,li] = Fraction(1)
                    for j in range(i+1, l.shape[1]):
                        l[i,j] = FullMatrix.zero(*l[i,j].shape, Fraction(0))
                else:
                    l[i, i] = 1
                    for j in range(i+1, l.shape[1]):
                        l[i,j] = 0
        return l
    
    # student
    def get_u(self):
        u = self.lu()
        for i in range(u.shape[0]):
                for j in range(i):
                    if type(u[i,j]) == FullMatrix:
                        u[i,j] = FullMatrix.zero(*u[i,j].shape, Fraction(0))
                    else:
                        u[i,j] = 0

        return u

    def det(self):
        _, upper = self.lu()

        det = 1

        for i in range(int(self.height)):
            det *= upper[i, i]

        return det

#---- ex 4-----------
        
    def swap_rows(self, row_1, row_2):
        temp = self.clone()
        temp[row_1, :] = self[row_2, :].data
        temp[row_2, :] = self[row_1, :].data
        return temp

    def lup(self):
        N = self.height
        lup = self.clone()
        p = FullMatrix.identity(N)
        for i in range(N):
            compare_val = 0
            compare_row = 0
            for j in range(i, N):
                if abs(lup[j, i]) > compare_val:
                    compare_val = abs(lup[j, i])
                    compare_row = j
            if compare_val != 0:
                #swap
                p = p.swap_rows(compare_row, i)
                lup = lup.swap_rows(compare_row, i)
                for j in range(i + 1, N):
                    lup[j, i] /= lup[i, i]
                    for k in range(i + 1, N):
                        lup[j, k] -= lup[j, i] * lup[i, k] 
        return lup, p

    def transpose(self):
        t = self.data
        return FullMatrix(t.T)


#----ex 5----------------------------

    def solve(self, b):
        N = self.height
        M = b.width
        x = FullMatrix.zero(N, M, Fraction(0, 1))
        y = FullMatrix.zero(N, M, Fraction(0, 1))
        #Ax = b <-> LUx = b <-> Ly = b
        lu, p = self.lup()
        l, u = lu.dissamble_lu()
        pb = p * b

        for i_ in range(M):
            for i in range(N):
                y[i, i_] = pb[i, i_] - sum([l[i, j] * y[j, 0] for j in range(i)])

            #Ux = y

            for i in range(N - 1, -1, -1):
                x[i, i_] = (y[i, i_] - sum([u[i, j] * x[j, i_] for j in range(N - 1, i, -1)])) / u[i, i]

        return x

    def qr(self):
        height = self.height
        width = self.width
        Q = FullMatrix.empty_like(self)
        u = FullMatrix.empty_like(self)

        u[:, 0] = self[:, 0].data
        Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

        for i in range(1, height):

            u[:, i] = self[:, i].data
            for j in range(i):
                scal = np.sum([self.data[k, i] * Q.data[k, j] for k in range(height)])
                # scal = (self.data[:, i] @ Q.data[:, j])
                u[:, i] -= (scal * Q[:, j]).data # get each u vector

            Q[:, i] = u[:, i] / np.linalg.norm(u[:, i]) # compute each e vetor

        R = FullMatrix.zero(self.width, self.width, 0.)
        for i in range(height):
            for j in range(i, width):
                R[i, j] = self.data[:, j] @ Q.data[:, i]

        return Q, R

    def lls_svd(self, y):
        rank = np.linalg.matrix_rank(self.data)
        u, s, vh = np.linalg.svd(self.data)
        x = np.empty((self.height,))
        for i in range(rank):
            to_add = (u[:, i].T.dot(y.data) / s[i] * vh.T[:, i]).T
            x = x + to_add
        return x
    
    def lls_qr(self, y):
        q, r = np.linalg.qr(self.data)
        r_trunc = r[:r.shape[1], :]
        q_trunc = q[:, :r.shape[1]]
        x = FullMatrix.zero(r.shape[1], 1, 0.)
        qy = q_trunc.T.dot(y.data)
        for i in range(1, r.shape[1] + 1):
            x[-i, 0] = (qy[-i] - (r_trunc[-i,:-i:-1].dot(x.data[:-i:-1, 0]))) / r_trunc[-i, -i]
        return x


class FullMatrix(Matrix):
    """
    Заполненная матрица с элементами произвольного типа.
    """
    def __init__(self, data):
        """
        Создает объект, хранящий матрицу в виде np.ndarray `data`.
        """
        assert isinstance(data, np.ndarray)
        self.data = data

    def empty_like(self, width=None, height=None):
        dtype = self.data.dtype
        if width is None:
            width = self.data.shape[1]
        if height is None:
            height = self.data.shape[0]       
        data = np.empty((height,width), dtype=dtype)
        return FullMatrix(data)
        
    @classmethod
    def zero(_cls, height, width, default=0):
        """
        Создает матрицу размера `width` x `height` со значениями по умолчанию `default`.
        """
        data = np.empty((height, width), dtype=type(default))
        data[:] = default
        return FullMatrix(data)
    
    def identity(N):
        E = FullMatrix.zero(N, N, Fraction(0, 1))
        for i in range(N):
            E[i, i] = Fraction(1, 1)
        return E
                    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
        
    def __getitem__(self, key):
        row, column = key
        return self.data[row, column]
    
    def __setitem__(self, key, value):
        row, column = key
        self.data[row, column] = value

#------------ex 6-------------

class SymmetricMatrix(Matrix):
    """
    Симметричная матрица с элементами произвольного типа.
    """
    def __init__(self, data):
        """
        Создает объект, хранящий матрицу в виде np.ndarray `data`.
        """
        assert isinstance(data, np.ndarray)
        self.data = data

    def empty_like(self, width=None, height=None):
        dtype = self.data.dtype
        if width is None:
            width = self.data.shape[1]
        if height is None:
            height = self.data.shape[0]       
        data = np.empty((height,width), dtype=dtype)
        return SymmetricMatrix(data)
        
    @classmethod
    def zero(_cls, height, width, default=0):
        """
        Создает матрицу размера `width` x `height` со значениями по умолчанию `default`.
        """
        data = np.empty((height, width), dtype=type(default))
        data[:] = default
        return SymmetricMatrix(data)
                    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
        
    def __getitem__(self, key):
        row, column = key
        if row > column:
            row, column = column, row
        return self.data[row, column]
    
    def __setitem__(self, key, value):
        row, column = key
        if row > column:
            row, column = column, row
        self.data[row, column] = value

    def ldl(self):
        N = self.height
        u = self.clone()
        for i in range(N - 1):
            u[i + 1:, i] = u[i, i + 1:]
        for i in range(N - 1):
            for j in range(i + 1, N):
                subst = u[i, i:].data
                subst = np.divide(subst, u[i, i])
                subst = np.multiply(subst, u[j, i])
                u[j, i:] = np.subtract(u[j, i:], subst)
        #now l 
        l = u.clone()
        for i in range(N):
            l[i, i:] = np.divide(l[i, i:], l[i, i])
        l = l.transpose()
        return l, u

class ToeplitzMatrix(Matrix):
    def __init__(self, data):
        """
        Создает объект, хранящий матрицу в виде np.ndarray `data`.
        """
        assert isinstance(data, np.ndarray)
        self.data = data
        self._height = len(data)
        self._width = self.height

    @property 
    def height(self):
        return self.data.shape[0]
    
    @property
    def width(self):
        return self.height

    def empty_like(self, width=None, height=None):
        dtype = self.data.dtype
        if width is None:
            width = self.data.shape[1]
        if height is None:
            height = self.data.shape[0]       
        data = np.empty((height,width), dtype=dtype)
        return ToeplitzMatrix(data)
        
    def __getitem__(self, index):
        i, j = index
        return self.data[abs(i-j)]
        
    def __setitem__(self, index, data):
        i, j = index
        self.data[abs(i-j)] = data
    
    def __str__(self):
        return '\n'.join([' '.join([str(self[i,j]) for j in range(len(self.data))]) for i in range(len(self.data))])

    def data_mat(self):
        N = self.height
        m = FullMatrix.zero(N, N, 0)
        for i in range(N):
            for j in range(N):
                m[i, j] = self[i, j]
        return m.data

    def levinson(self, y):
        N = self.height
        forward = [None] * N
        forward[0] = [1 / self[0, 0]]

        backward = [None] * N
        backward[0] = forward[0]

        for i in range(1, N):

            det = 1 - eps_backward(self, backward[i - 1], i) * eps_forward(self, forward[i - 1], i)

            forward[i] = 1 / det * np.concatenate((forward[i - 1], [0])) - eps_forward(self, forward[i - 1], i) / det * np.concatenate(([0], backward[i - 1]))
            backward[i] = forward[i][::-1]

        x = [None] * N

        x[0] = [y[0] / self[0, 0]]

        for i in range(1, N):
            x[i] = np.concatenate([x[i - 1], [0]]) + (y[i] - eps_forward(self, x[i - 1], i)) * backward[i]
        
        return x[N - 1]

def eps_forward(M, forward, n):
        eps_f = np.sum([M[n, i] * forward[i] for i in range(n)])
        # print(f'eps_f^{n + 1} = {eps_f}')
        return eps_f
    
def eps_backward(M, backward, n):
    eps_b = np.sum([M[0, i] * backward[i - 1] for i in range(1, n + 1)])
    # print(f'eps_b^{n + 1} = {eps_b}')
    return eps_b

class Laplace2D:
    def __init__(self, N):
        mat = np.zeros((N, N, N, N), dtype=int)
        for i1 in range(N):
            for i2 in range(N):
                mat[i1, i2, i1, i2] = -4
                mat[i1, i2, (i1- 1 )% N, i2] = 1
                mat[i1, i2, (i1 + 1) % N, i2] = 1
                mat[i1, i2, i1, (i2 - 1) % N] = 1
                mat[i1, i2, i1, (i2 + 1) % N] = 1
      
        
        self.m = mat.reshape((N**2, N**2))
        self.n = N

def f_solve(M, y):
    x = [0] * len(y)
    for i in range(1, len(y)):
        x[i] = y[i] / M[i, i]
    return x

def D(N):
    mat = Laplace2D(N).m

    return ToeplitzMatrix(mat[0, :])


N = 20
A = FullMatrix.zero(N, 2, 1.)
y = FullMatrix.zero(N, 1, 0.)
# eps = 1e-13
# A[0, 1] += eps
# A[1, 1] -= eps

for i in range(N):
    numb = np.random.randint(1, 50)
    A[i, 0] = numb
    y[i, 0] = numb + np.random.randint(-10, 10)
print(f'A:')
print(A)
print('----')
# print('------QR decomposition:-----')
# Q, R = A.qr()
# print('Q:')
# print(Q)
# print('---------')
# print('Q * Q.T')
# print(Q.data.dot(Q.data.T))
# print('------')
# print('R:')
# print(R)
# print('----')
# print('Q * R:')
# print(Q.data.dot(R.data))

# print('-----LLS SVD-------')

print('y:')
print(y)
# print('----')
# x = A.lls_svd(y)
# print('Ax')
# print(A.data.dot(x.data))
# print('singular:')
# _, s, _ = np.linalg.svd(A.data)
# print(s)

print('-----LLS QR-------')
x = A.lls_qr(y)
print('Ax')
Ax = A.data.dot(x.data)
print(Ax)

x_ax = A[:, 0].data
#---check with polyfit:
z = np.polyfit(x_ax, y.data, 1)
z = z.flatten()
poly = np.poly1d(z)


plt.scatter(x_ax, y.data, s = 7, label = 'exp data')
plt.plot(x_ax, Ax, '-r', label = 'linear fit', linewidth = 2)
plt.plot(x_ax, poly(x_ax), '--b', label = 'np.polyfit', linewidth = 0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
