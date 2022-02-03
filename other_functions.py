import numpy as np

def lower_diagonal_matrix(n):
    """Returns a lower diagonal matrix of size n x n.
        

       For a 3x3 matrix this would look like:
                        0 0 0
                        1 0 0
                        0 1 0
    """
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if(i-1==j):
                M[i][j] = 1
    return M
