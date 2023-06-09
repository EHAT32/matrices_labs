import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
import numba as nb
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
        raise NotImplementedError
    
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
    
    def zero(self, width=None, height=None):
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
        if isinstance(other, float|int|Fraction):
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
    
    # question 1
    def lu(self):
        c = self.copy()
        for i in range(self.shape[0]):
            for j in range(i+1, self.shape[0]):
                        c[j, i] /= c[i,i]
                        for k in range(i+1, self.shape[0]):
                            c[j,k]-=c[j,i]*c[i,k]
        return c


    # student
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
    
    # student
    def det(self):
        u = self.get_u()

        det = u[0,0]
        for i in range(1, self.shape[0]):
                det*=u[i,i]
        return det
    
    def solve_lup(self, b):
        x = FullMatrix.empty_like(b) # l*u*x = p*b
        y = FullMatrix.empty_like(b) # l*y = p*b
        lu, p = self.lup()
        pb = p*b
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                y[i,j] = pb[i,j]
                for k in range(i):
                    if i<=k:
                        print(i,k, 'catch')
                    y[i,j]-=lu[i,k]*y[k,j]         
        for i in range(b.shape[0]-1, -1, -1):
            for j in range(b.shape[1]-1, -1, -1):
                x[i,j] = y[i,j]/lu[i,i]
                for k in range(y.shape[0]-1, i, -1):
                    x[i,j] -= lu[i,k]*x[k,j]/lu[i,i]
        return x
    
    def copy(self):
        c = self.empty_like(self.shape[0], self.shape[1])
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                c[i,j] = self[i,j]
        return c

    def swaprows(self, i_r, j_r):
        c = self.copy()
        for i in range(self.shape[1]):
            acc = c[i_r, i]
            c[i_r, i] = c[j_r, i]
            c[j_r, i] = acc
        return c

    # Вроде работает 
    def lup(self):
        c = self.copy()
        p = FullMatrix.zero(self.shape[0], self.shape[1], Fraction(0))
        for i in range(self.shape[0]):
            p[i,i] = Fraction(1)
        for i in range(self.shape[0]):
            pVal = Fraction(0)
            pivot = -1
            for row in range(i, self.shape[0]):
                if abs(c[row, i]) > pVal:
                    pVal = abs(c[row, i])
                    pivot = row
            if pVal !=Fraction(0):
                p = p.swaprows(pivot, i)
                c = c.swaprows(pivot, i)
                for j in range(i+1, self.shape[0]):
                    c[j, i] /= c[i,i]
                    for k in range(i+1, self.shape[0]):
                        c[j,k]-=c[j,i]*c[i,k]
        return c, p
    def get_l_p(self):
        l = self.lup()[0]
        for i in range(self.shape[0]):
            l[i,i] = Fraction(1)
            for j in range(i+1, self.shape[1]):
                l[i,j] = Fraction(0)
        return l
    def get_u_p(self):
        u = self.lup()[0]
        for i in range(self.shape[0]):
            for j in range(i):
                u[i,j] = Fraction(0)
        return u
    
    def T(self):
        dat = self.data
        return FullMatrix(dat.T)
        
    
    # поменять под LUP 
    def __truediv__(self ,other):
        e = other.zero(*other.shape, Fraction(0))
        for i in range(other.shape[0]):
            e[i,i] = Fraction(1)
        return self*other.solve_lup(e)
    def __rtruediv__(self, other):
        e = self.zero(*self.shape, Fraction(0))
        for i in range(self.shape[0]):
            e[i,i] = Fraction(1)
        return other*self.solve_lup(e)
    
    def clone(self):
        copy = np.copy(self.data)
        # if isinstance(self, SymmetricMatrix):
        #     return SymmetricMatrix(copy)
        # else:
        return FullMatrix(copy)
    def transpose(self):
        t = self.data
        return FullMatrix(t.T)
        

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

def eps_b(M, b, n):
    #print(f"arr={[M[-i]*b[i] for i in range(1, n)]}")
    return sum([M[-i]*b[i-1] for i in range(1, n+1)])

def eps_f(M, f, n):
    return sum([M[n-i+1]*f[i-1] for i in range(1, n+1)])


class Toeplitz():
    def __init__(self, c, r):
        if len(r) != len(c)+1:
            raise KeyError("len(r) must be len(c)+1")
        self.c = c[::-1]
        self.r = r
        self.n = len(r)
    
    def __getitem__(self, key):
        d = key
        if d >=0:
            return self.r[d]
        else:
            return self.c[-d-1]
    
    def __setitem__(self, key, value):
        d = key
        if d >=0:
            self.r[d] = value
        else:
            self.c[-d-1] = value
    def print(self):
        for i in range(self.n):
            print([self[i-j] for j in range(self.n)])
    
    def data(self):
        return np.asarray([[self[i-j] for j in range(self.n)] for i in range(self.n)])
    
    def levinson(self, y):
        f = [None]*self.n
        b = [None]*self.n

        b[0], f[0] = [1/self[0]], [1/self[0]]

        for i in nb.prange(1, self.n):
            f[i] = 1/(1-eps_b(self, b[i-1], i)*eps_f(self, f[i-1], i))*np.concatenate([f[i-1], [0]]) -\
            eps_f(self, f[i-1], i)/(1-eps_b(self, b[i-1], i)*eps_f(self, f[i-1], i))*np.concatenate([[0],b[i-1]])

            b[i] = 1/(1-eps_b(self, b[i-1], i)*eps_f(self, f[i-1], i))*np.concatenate([[0],b[i-1]]) -\
            eps_f(self, f[i-1], i)/(1-eps_b(self, b[i-1], i)*eps_f(self, f[i-1], i))*np.concatenate([f[i-1], [0]])
            #print(f"e_f = {eps_f(self, f[i-1], i)}, i={i}")
            #print(f"e_b = {eps_b(self, b[i-1], i)}, i={i}")

        #print(f, b)
        xHat = [None]*self.n

        xHat[0] = [y[0] / self[0]]

        for i in range(1, self.n):
            xHat[i] = np.concatenate([xHat[i-1], [0]]) + (y[i] - eps_f(self, xHat[i-1], i))*b[i]
        
        return xHat[self.n-1]

        


def e_delta(n=4, pos=None):
    if pos == 'down':
        key = (-1, 0)
    if pos == 'up':
        key = (0, -1)
    e = FullMatrix(np.eye(n, n, dtype=int))
    e.data[key] = -1
    return e

def angle(n=4, pos=None):
    if pos == 'down':
        key = (0, -1)
    if pos == 'up':
        key = (-1, 0)

    a = FullMatrix.zero(n,n)
    a[key] = -1
    return a

def delta(n=4):
    d = FullMatrix.zero(n,n,0)

    d[0, 0], d[0, 1] = 4,  -1
    d[n-1, n-1], d[n-1, n-2] = 4, -1

    for i in range(1, d.shape[0]-1):
        d[i, i] = 4
        d[i, i+1 % n] = -1
        d[i, i-1 % n] = -1
    return d



class L:
    def __init__(self, n=4):
        mat = np.zeros((n, n, n, n), dtype=int)
        for i1 in range(n):
            for i2 in range(n):
                mat[i1, i2, i1, i2] = -4
                mat[i1, i2, (i1-1)%n, i2] = 1
                mat[i1, i2, (i1+1)%n, i2] = 1
                mat[i1, i2, i1, (i2-1)%n] = 1
                mat[i1, i2, i1, (i2+1)%n] = 1
        # for j in range(n):
        #     mat[j, n-1, j, 0] = 0
        #     mat[j, 0, j, n-1] = 0
        #     mat[j, 0, max(j-1, 0), n-1] = 1
        #     mat[j, n-1, min(j+1, n-1), 0] = 1    

        # mat[0, 0, 0, -1] = 0
        # mat[-1, -1, -1, 0] = 0        
        
        self.m = mat.reshape((n**2, n**2))
        self.n = n

    

# n = 20
# D = L(n)

# print(D.m)
# M = D.m.reshape((n,n,n,n))
# print((np.roll(np.roll(M, shift=1, axis=1), shift=1, axis=3) - M).reshape((n**2,n**2)))
#print("____________\n", np.roll(np.roll(M, shift=1, axis=0), shift=1, axis=2).reshape((n**2,n**2)))

#print(M.reshape((n**2,n**2)))


# FMF = np.fft.ifft2(np.fft.fft2(M, axes=[0, 1]), axes=[2, 3])
#print(np.round(FMF.reshape((n**2, n**2)), 3))
#print(M.reshape(n**2, n**2))
#print(FMF.reshape((n**2, n**2)))


# def f(x, y):
#     return np.cos(np.pi * x)*np.cos(np.pi * y)


# points = np.linspace(-1, 1, n)

# y_true = np.array([[f(x, y) for x in points] for y in points])

# y_true -= np.sum(y_true)

# y_right = D.m.dot(np.ravel(y_true).T)


# print(np.sum(y_right), 'heres')
#y_right -= np.sum(y_right)



# Fy_right = np.fft.fft2(y_right.reshape((n,n)))

# diag = np.array([FMF.reshape((n**2, n**2))[i,i] for i in range(n**2)])

# Fy_left = np.ravel(Fy_right)/diag
print("_____")
#print(Fy_left)

# Fy_left[0] = 0

# print(np.fft.ifft2((Fy_left.reshape((n,n)))).reshape((n**2,)).T)
# print(f"max Delta={np.max(np.abs(D.m.dot(np.fft.ifft2((Fy_left.reshape((n,n)))).reshape((n**2,)).T) - y_right))}")



            
            










# D = Toeplitz([angle(pos='down'), e_delta(pos='down')], [delta(), e_delta(pos='up'), angle(pos='up')])

# D.print()
# print(D[2])

# T = Toeplitz([0., 1., 1.23], [5., 1.23, 1., 0.])

# y = np.array([4, 4, 4, 4])

#print(f"T*x = {T.data().dot(T.levinson(y).T)}")
#T.print()
#print(f"x={T.levinson(y)}")
#print(f"y={y}")
#print(np.linalg.solve(T.data(), y))

# B = Toeplitz([0, 1, 2], [3, 4, 5, 6])

# B.print()
# print(B[-1])
#print(list(range(1, 1)))
