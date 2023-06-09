import numpy as np
import numba as nb
from numpy import linalg as LA
from scipy.sparse.linalg import gmres
from scipy.linalg import solve


#___________________________________ex. 1___________________________________
def dirichlet(x, i, j, k):
    N = len(x)
    if i == -1 or i == N:
        return 0
    elif j == -1 or j == N:
        return 0
    elif k == -1 or k == N:
        return 0
    else:
        return x[i,j,k] 

class Laplace_first:
    def __init__(self, n):
        self.n = n
        pass

    def mul3d(self, x, h=None):
        N = len(x)
        if h==None:
            h = 1/N
        result = np.empty_like(x)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    result[i, j, k] = -6*dirichlet(x, i,j,k)+\
                            dirichlet(x, i+1,j,k)+dirichlet(x, i-1,j,k)+\
                            dirichlet(x, i,j+1,k)+dirichlet(x, i,j-1,k)+\
                            dirichlet(x, i,j,k+1)+dirichlet(x, i,j,k-1)
        return result / h**2
    
    def mul3d_T(self, x, h=None):
        N = len(x)
        if h==None:
            h = 1/N
        result = np.empty_like(x)
        for j in range(N):
            for i in range(N):
                for k in range(N):
                    result[i, j, k] = -6*dirichlet(x, i,j,k)+\
                            dirichlet(x, i+1,j,k)+dirichlet(x, i-1,j,k)+\
                            dirichlet(x, i,j+1,k)+dirichlet(x, i,j-1,k)+\
                            dirichlet(x, i,j,k+1)+dirichlet(x, i,j,k-1)
        return result / h**2
    
    def __call___(self, x):
        n = self.n
        x_ijk = x.reshape((n,n,n))
        res = self.mul3d(x_ijk)
        return res.reshape(x.shape)
    def __matmul__(self, x):
        return self.__call___(x)

class Laplace:
    def __init__(self, n):
        self.n = n
    @staticmethod
    @nb.jit("float64[:,:,:](float64[:,:,:], int32)", nopython=True)
    def mul(x, n):
        res = -6 * x
        mat = np.zeros((n+2, n+2, n+2), dtype=np.float32)
        mat[1:n+1, 1:n+1, 1:n+1] = x
        mat[n:,n:,n:] = x[:1,:1,:1]
        res += mat[:n, 1:(n+1), 1:(n+1)]+mat[2:, 1:(n+1), 1:(n+1)]+mat[1:(n+1), :n, 1:(n+1)]+mat[1:(n+1), 2:, 1:(n+1)]\
              +mat[1:(n+1), 1:(n+1), :n]+mat[1:(n+1), 1:(n+1), 2:]
        return res*n**2
    def __matmul__(self, x):
        return self.mul(x.reshape((self.n,self.n,self.n)), self.n).flatten()


class Operator:
    def __init__(self, A):
        self.A = A
        self.shape = A.shape
    def __call__(self, x):
        return np.dot(self.A, x)
    def __matmul__(self, x):
        return np.dot(self.A, x)
    def __getitem___(self, key):
        return self.A[key]

        

def B(N=10):
    result = np.empty((N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                result[i,j,k] = (i/N-0.5)**2 + (j/N-0.5)**2 + (k/N-0.5)**2
    return result


#___________________________________ex. 2___________________________________

#___________________________________ex. 3, 4, 5___________________________________
def compute_krylov_basis(A, b, m):
    """Generates a numerical basis for the m-dimensional Krylov subspace."""
    n = A.shape[0]
    
    result = np.empty((n, m), dtype=np.float64)
    
    result[:, 0] = b
    
    for index in range(1, m):
        result[:, index] = A @ result[:, index - 1]
        
    return result

# print(compute_krylov_basis(L, b.flatten(), 10))
#___________________________________ex. 6___________________________________

#-------------------------------------------------------
# Lanczos algorithm
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
  return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def lanczos(A, v1):
  np.set_printoptions(precision=3, suppress=True)
  x, y = [], []
  n = A.shape[1]
  v2, beta = 0.0, 0.0

  for i in range(n):
    w_prime = np.dot(A, v1)
    conj = np.matrix.conjugate(w_prime)
    alpha = np.dot(conj, v1)
    w = w_prime - alpha * v1 - beta * v2
    beta = np.linalg.norm(w)
    x.append(np.linalg.norm(alpha))

    if i < (n-1):
        y.append(beta)
    v2 = v1
    v1 = w/beta
    
  return tridiag(y, x, y)

# A = np.diag([0., 1., 2., 3., 4., 5.])
# n = A.shape[1]
# v_0 = np.zeros(n)
# v_0.fill(1.)
# v = v_0 / np.linalg.norm(v_0)


#-------------------------------------------------------
# Arnoldi iteration
def compute_krylov_basis_orthogonal(A, b, m):
    """Generates an orthogonal basis for the m-dimensional Krylov subspace."""
    n = A.shape[0]
    
    result = np.empty((n, m), dtype=np.float64)
    
    result[:, 0] = b / np.linalg.norm(b)
    
    for index in range(1, m):
        tmp = A @ result[:, index - 1]
        h = result[:, :index].T @ tmp
        w = tmp - result[:, :index] @ h
        htilde = np.linalg.norm(w)
        result[:, index] = w[:] / htilde
                
    return result



# T = lanczos(A, v)
# print(f'Tridiagonalization of A: \n{T}')

# w, v = LA.eig(T)
# print(f'\nAssociated eigenvalues: \n{w}')
# print(f'\nAssociated eigenvectors: \n{v}')

#___________________________________ex. 7___________________________________

def fom(A, b, m_max = 100):
    """Explore the full orthogonalization method."""

    
    r = b
    beta = np.linalg.norm(r)

    V = np.empty((A.shape[0], m_max), dtype='float64')
    V[:, 0] = r / beta
    

    H = np.zeros((m_max, m_max), dtype='float64')
    
    rel_residuals = [1]
        
    for m in range(1, m_max):
        tmp = A @ V[:, m - 1]
        H[:m, m - 1] = V[:, :m].T @ tmp 
        w = tmp - V[:, :m] @ H[:m, m - 1] 
        htilde = np.linalg.norm(w)
        V[:, m] = w[:] / htilde
        H[m, m - 1] = htilde 

        # Now solve the projected system
        
        y = solve(H[:m, :m], V[:, :m].T @ b)
        
        x = V[:, :m] @ y
        r = b - A @ x
        rnorm = np.linalg.norm(r)
        rel_residuals.append(rnorm / beta)            
    return x, V, rel_residuals


# Arnoldi’s Method for Linear Systems (FOM)
# 6.4
# def FOM(A, b, x0, m, tol=1e-8):
#     r0 = b - np.dot(A, x0)
#     betta = np.linalg.norm(r0)
#     V = np.empty((len(b), m+1))
#     V[:, 0] = r0/betta
#     W = np.empty_like(V)
#     H = np.zeros((m,m))
#     for j in range(m):
#         W[:, j] = np.dot(A, V[:, j])
#         for i in range(j):
#             H[i,j] = np.dot(W[:, j], V[:, i])
#             W[:, j] -= H[i,j]*V[:,i]
#         H[j+1, j] = np.linalg.norm(W[:,j])
#         if H[j+1, j] < tol:
#             break
#         V[:, j+1] = W[:, j] / H[j+1, j]
#     e1 = np.zeros((H.shape[0],))
#     e1[0] = 1
#     y = np.dot(np.linalg.inv(H), betta*e1)
#     x = x0 + np.dot(V, y)
#     res = np.linalg.norm(b - np.dot(A, x))
#     return x, res



#___________________________________ex. 8___________________________________

# 6.5.1
# def GMRES(A, x0, b, m, eps = 1e-14):
#     r = b - np.dot(A, x0)
#     betta = np.linalg.norm(r)
#     v = [None]*m
#     w = [None]*m
#     H = np.empty((m+1,m))
#     v[0] = r/betta
#     for j in range(m):
#         w[j] = np.dot(A, v[j])
#         for i in range(j):
#             H[i,j] = np.dot(w[j], v[i])
#             w[j] = w[j] - np.dot(H[i,j], v[i])
        
#         H[j+1, j] = np.linalg.norm(w[j])
#         if H[j+1, j] <= eps:
#             m = j
#             break
#         v[j+1] = w[j]/H[j+1,j]
    # Compute y_m the minimizer of ||β e1 − H_m y || and xm = x0 + V_m y_m.
    # V^T_m r_0 = V^T_m(betta v_1) = betta e_1
    # V_m = (n, m) matrix = {v_1, v_2, ..., v_m}


def GMRES(A, b, x0, k, tol=1e-8):
    Q = np.empty((len(b), k+1))
    H = np.zeros((k+1, k))
    r0 = b - A @ x0
    betta = np.linalg.norm(r0)
    Q[:,0] = r0/np.linalg.norm(r0)
    for j in range(k):
        Q[:, j+1] = A @ Q[:, j]
        for i in range(j+1):
            # print(Q[:,i], Q[:, j+1])
            H[i,j] = np.dot(Q[:,i], Q[:, j+1])
            Q[:, j+1] -= H[i,j]*Q[:, i]
        H[j+1, j] = np.linalg.norm(Q[:, j+1])
        if np.abs(H[j+1, j]) > tol:
            Q[:, j+1] = Q[:, j+1] / H[j+1, j]
        e1 = np.zeros((H[:j+2, :j+1].shape[0],))
        e1[0] = 1
        # print(H[:j+2, :j+1].shape)
        y, residuals, rank, s = np.linalg.lstsq(H[:j+2, :j+1], betta * e1)
        res = np.linalg.norm(np.dot(H[:j+2, :j+1], y) - betta*e1)
        if res < tol:
            return np.dot(Q[:,:j+1], y) + x0, res


# a = np.array([[1, 0, 0],
#               [0, 2, 0],
#               [0, 0, 3]])

# A = Operator(a)

# b = np.array([1, 4, 6])

# x0 = np.zeros(b.shape)

# n = 2

# L = Laplace(n)
# L.shape = (n**2, n**2)
# b = B(n)

# print(L @ b.flatten())

# print("My GMRES: ", GMRES(L, b.flatten(), np.zeros(b.flatten().shape), 100))
#print(A @ x0)
# print("My GMRES: ", GMRES(A, b, x0, 100))

# print("Scipy's GMRES: ",gmres(a, b, x0, 1e-8))


#___________________________________ex. 9___________________________________

#___________________________________ex. 10__________________________________
