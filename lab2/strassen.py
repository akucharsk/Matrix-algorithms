import numpy as np
from util import Number

def Strassen(A: np.ndarray, B: np.ndarray, iter:int = 0) -> np.ndarray :
    
    shape_start = A.shape[0]
    if A.shape[0] == 1:
        return A * B

     
    if iter == 0:
        
        m = 1
        n = A.shape[0]
        while m < n:
            m *= 2
        A = np.pad(A, ((0, m - A.shape[0]), (0, m - A.shape[1])), mode='constant', constant_values=(Number(0), Number(0)))
        B = np.pad(B, ((0, m - B.shape[0]), (0, m - B.shape[1])), mode='constant', constant_values=(Number(0), Number(0)))


    
    
    n = A.shape[0]//2

    
    A_11, A_12, A_21, A_22 = A[:n,:n], A[:n,n:], A[n:,:n],A[n:,n:]
    B_11, B_12, B_21, B_22 = B[:n,:n], B[:n,n:], B[n:,:n],B[n:,n:]
    
    iter_temp = iter + 1

    M1 = Strassen(A_11+A_22, B_11+B_22, iter_temp)
    M2 = Strassen(A_21+A_22, B_11, iter_temp)
    M3 = Strassen(A_11, B_12-B_22, iter_temp)
    M4 = Strassen(A_22, B_21-B_11, iter_temp)
    M5 = Strassen(A_11 + A_12, B_22, iter_temp)
    M6 = Strassen(A_21-A_11, B_11+B_12, iter_temp)
    M7 = Strassen(A_12-A_22, B_21+B_22, iter_temp)
    
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return C[:shape_start, :shape_start]


def Strassen_no_pad(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.shape[0] == 1:
        return A * B

    
    
    n = A.shape[0]//2

    
    A_11, A_12, A_21, A_22 = A[:n,:n], A[:n,n:], A[n:,:n],A[n:,n:]
    B_11, B_12, B_21, B_22 = B[:n,:n], B[:n,n:], B[n:,:n],B[n:,n:]

    M1 = Strassen_no_pad(A_11+A_22, B_11+B_22)
    M2 = Strassen_no_pad(A_21+A_22, B_11)
    M3 = Strassen_no_pad(A_11, B_12-B_22)
    M4 = Strassen_no_pad(A_22, B_21-B_11)
    M5 = Strassen_no_pad(A_11 + A_12, B_22)
    M6 = Strassen_no_pad(A_21-A_11, B_11+B_12)
    M7 = Strassen_no_pad(A_12-A_22, B_21+B_22)
    
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return C



def SMU(A, B): 
    if len(A) == 1:
        return A @ B
    
    # Dodanie wypełnienia, jeśli rozmiar macierzy nie jest potęgą dwójki
    m = 1
    n = max(len(A), len(B))
    while m < n:
        m *= 2
    A = np.pad(A, ((0, m - len(A)), (0, m - len(A[0]))), mode='constant', constant_values=(0,))
    B = np.pad(B, ((0, m - len(B)), (0, m - len(B[0]))), mode='constant', constant_values=(0,))

    n = len(A) // 2
    A_11, A_12, A_21, A_22 = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]
    B_11, B_12, B_21, B_22 = B[:n, :n], B[:n, n:], B[n:, :n], B[n:, n:]
    M1 = SMU(A_11 + A_22, B_11 + B_22)
    M2 = SMU(A_21 + A_22, B_11)
    M3 = SMU(A_11, B_12 - B_22)
    M4 = SMU(A_22, B_21 - B_11)
    M5 = SMU(A_11 + A_12, B_22)
    M6 = SMU(A_21 - A_11, B_11 + B_12)
    M7 = SMU(A_12 - A_22, B_21 + B_22)
    C = np.zeros((2 * n, 2 * n), dtype=A.dtype)
    C[:n, :n] = M1 + M4 - M5 + M7
    C[:n, n:] = M3 + M5
    C[n:, :n] = M2 + M4
    C[n:, n:] = M1 - M2 + M3 + M6
    return C[:len(A), :len(B)]