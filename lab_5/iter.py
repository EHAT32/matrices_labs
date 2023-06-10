import numpy as np
from scipy.linalg import solve
import numba as nb


#######################ex1#########################
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

class Laplace:
    def __init__(self, N):
        self.N = N

    def mul3d(self, x, h=None):
        N = len(x)
        if h==None:
            h = 1 / N
        result = np.empty_like(x)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    result[i, j, k] = -6*dirichlet(x, i, j, k)+\
                            dirichlet(x, i+1,j,k)+dirichlet(x, i - 1, j, k)+\
                            dirichlet(x, i,j+1,k)+dirichlet(x, i, j - 1, k)+\
                            dirichlet(x, i,j,k+1)+dirichlet(x, i, j, k - 1)
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
    
    def __matmul__(self, x):
        N = self.N
        return self.mul3d(x.reshape((N, N, N))).flatten()


class Numbaplace:
    def __init__(self, n):
        self.n = n
    @staticmethod
    @nb.jit("float64[:,:,:](float64[:,:,:], int32)", nopython=True)
    def mul(x, n):
        res = -6 * x
        mat = np.zeros((n+2, n+2, n+2), dtype=np.float32)
        mat[1:n+1, 1:n+1, 1:n+1] = x
        mat[n:, n:, n:] = x[:1, :1, :1]
        res += mat[:n, 1:(n + 1), 1:(n + 1)]+\
            mat[2:, 1:(n + 1), 1:(n + 1)]+\
            mat[1:(n+1), :n, 1:(n+1)]+\
            mat[1:(n+1), 2:, 1:(n+1)]+\
            mat[1:(n+1), 1:(n+1), :n]+\
            mat[1:(n+1), 1:(n+1), 2:]
        return res*n**2
    def __matmul__(self, x):
        return self.mul(x.reshape((self.n,self.n,self.n)), self.n).flatten()

def B(N=10):
    result = np.empty((N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                result[i, j, k] = (i / N-0.5) ** 2 + (j / N-0.5) ** 2 + (k / N-0.5) ** 2
    return result


#######################ex3--5#########################
def gen_krylov_basis(A, b, m):
    n = A.shape[0]
    
    result = np.empty((n, m), dtype=np.float64)
    
    result[:, 0] = b
    
    for index in range(1, m):
        result[:, index] = A @ result[:, index - 1]
        
    return result


#######################ex6#########################
# Arnoldi

def gen_krylov_basis_orthogonal(A, b, m):
    n = A.shape[0]
    
    result = np.empty((n, m), dtype=np.float64)
    
    result[:, 0] = b / np.linalg.norm(b)
    
    for i in range(1, m):
        tmp = A @ result[:, i - 1]
        # orthogonalise
        h = result[:, :i].T @ tmp
        w = tmp - result[:, :i] @ h
        # Normalise
        w_norm = np.linalg.norm(w)
        result[:, i] = w[:] / w_norm
                
    return result

# Lanczos
def tridiag(a, b, c, k1 = -1, k2 = 0, k3 = 1):
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

    if i < (n - 1):
        y.append(beta)
    v2 = v1
    v1 = w / beta
    
  return tridiag(y, x, y)


#######################ex7#########################

def FOM(A, b, m_max = 100):
    
    r = b # x_0 = 0
    beta = np.linalg.norm(r)
    
    # V = basis
    V = np.empty((A.shape[0], m_max), dtype='float64')
    V[:, 0] = r / beta
    
    # H = matrix projection
    H = np.zeros((m_max, m_max), dtype='float64')
    
    # residual
    rel_residuals = [1]
        
    for m in range(1, m_max):
        tmp = A @ V[:, m - 1]
        # orthogonalise
        H[:m, m - 1] = V[:, :m].T @ tmp 
        w = tmp - V[:, :m] @ H[:m, m - 1]
        # normalise
        w_norm = np.linalg.norm(w)
        V[:, m] = w[:] / w_norm
        H[m, m - 1] = w_norm 

        # solve the projected system
        
        y = solve(H[:m, :m], V[:, :m].T @ b)
        
        x = V[:, :m] @ y
        r = b - A @ x
        rnorm = np.linalg.norm(r)
        rel_residuals.append(rnorm / beta)            
    return x, V, rel_residuals

#######################ex8#########################

def GMRES_LAB(A, b, x0, k, tol = 1e-8):
    Q = np.empty((len(b), k + 1))
    H = np.zeros((k + 1, k))
    r0 = b - np.dot(A, x0)
    betta = np.linalg.norm(r0)
    Q[:,0] = r0/np.linalg.norm(r0)
    for j in range(k):
        Q[:, j+1] = np.dot(A, Q[:, j])
        for i in range(j + 1):
            H[i,j] = np.dot(Q[:, i], Q[:, j + 1])
            Q[:, j + 1] -= H[i, j]*Q[:, i]
        H[j + 1, j] = np.linalg.norm(Q[:, j+1])
        if np.abs(H[j + 1, j]) > tol:
            Q[:, j + 1] = Q[:, j + 1] / H[j+1, j]
        e1 = np.zeros((H[:j + 2, :j + 1].shape[0],))
        e1[0] = 1
        y, _, _, _ = np.linalg.lstsq(H[:j + 2, :j + 1], betta * e1, rcond=-1)
        res = np.linalg.norm(np.dot(H[:j + 2, :j + 1], y) - betta*e1)
        if res < tol:
            return np.dot(Q[:,:j+1], y) + x0, res
