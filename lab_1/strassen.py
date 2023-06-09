import numpy as np

def strassen(A, B):
    n = len(A)
    if n == 1:
        return A * B
    else:
        # Divide
        A11, A12, A21, A22 = A[: n // 2, : n // 2], A[: n // 2,  n // 2 :], A[ n // 2 :, : n // 2], A[ n // 2 :,  n // 2 :]
        B11, B12, B21, B22 = B[: n // 2, : n // 2], B[: n // 2,  n // 2 :], B[ n // 2 :, : n // 2], B[ n // 2 :,  n // 2 :]

        # Compute
        P1 = strassen(A11 + A22, B11 + B22)
        P2 = strassen(A21 + A22, B11)
        P3 = strassen(A11, B12 - B22)
        P4 = strassen(A22, B21 - B11)
        P5 = strassen(A11 + A12, B22)
        P6 = strassen(A21 - A11, B11 + B12)
        P7 = strassen(A12 - A22, B21 + B22)

        # Compute the result
        C11 = P1 + P4 - P5 + P7
        C12 = P3 + P5
        C21 = P2 + P4
        C22 = P1 - P2 + P3 + P6

        # Combine the result
        C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
        return C

A = np.random.random((2 ** 5, 2 ** 5))
B = np.random.random((2 ** 5, 2 ** 5))

C = strassen(A, B)

print(C)