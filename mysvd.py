
import numpy as np
import math


def SVD(A):
    toFindU = A @ A.T # multiplication of A and transpose of A (to find eigenvalues and corresponding eigenvectors)
    toFindVt = A.T @ A # multiplication of A transpose and A (to find eigenvalues and corresponding eigenvectors)
    eigenvalues, matrixToFindU = np.linalg.eig(toFindU)
    eigenvalues2, matrixToFindV = np.linalg.eig(toFindVt)
    sortEigens(eigenvalues, matrixToFindU)
    sortEigens(eigenvalues2, matrixToFindV)
    sizeofU = matrixToFindU.shape
    sizeofV = matrixToFindV.shape
    U = gramSchmidt(sizeofU[0], sizeofU[1], matrixToFindU) 
    V = gramSchmidt(sizeofV[0], sizeofV[1], matrixToFindV)
    Vt = V.T
    E = singularValues(eigenvalues)
    return U, E, Vt


def singularValues(eigenvalues): # to create vector form of singularvalues (E)
    arr = np.zeros((len(eigenvalues), 1))
    for i in range(len(eigenvalues)):
        arr[i][0] = math.sqrt(max(eigenvalues[i], 0))  # in case of negative root
    return arr.flatten()

def sortEigens(eigenvalues, eigenvectors):# to sort eigenvalues and eigenvectors as ascending order 
    n = len(eigenvalues)
    for i in range(n):
        for j in range(0, n - i - 1):
            if eigenvalues[j] < eigenvalues[j + 1]:  
                eigenvalues[j], eigenvalues[j + 1] = eigenvalues[j + 1], eigenvalues[j]
                swapColumns(eigenvectors, j, j+1)

def swapColumns(matrix, index1, index2): # to change eigenvectors matrix
    temp = matrix[:, index1].copy()
    matrix[:, index1] = matrix[:, index2]
    matrix[:, index2] = temp

def gramSchmidt(rowSize , colSize, arr): 
    originalMatrix = np.zeros((rowSize, colSize)) # to create a matrix to hold vectors resulted from orthagonalization 
    originalMatrix[:,0] = (arr[:,0] / np.linalg.norm(arr[:,0])) # first column of the matrix was chosen as a first column of orthanormal matrix
    for i in range(colSize - 1): # the iteration of orthagonalization starts
        temp = np.zeros((rowSize, 1))
        for j in range(i+1):
            temp += np.dot(originalMatrix[:,j], arr[:,i+1]) * originalMatrix[:,j].reshape(-1, 1)
        unnormalizedU = arr[:,i+1].reshape(-1, 1) - temp
        norm = np.linalg.norm(unnormalizedU)
        if norm != 0: # to check norm. if norm is equal to zero vector will be a zero vector. Thus, orthagonalization will fail.
            originalMatrix[:,i+1] = (unnormalizedU / norm).flatten()
    return originalMatrix


