import numpy as np
import scipy.linalg as la
import copy as cp
import time

import ctypes
from numpy.ctypeslib import ndpointer
_ = ctypes.CDLL('libblas.so', mode=ctypes.RTLD_GLOBAL)
approxInv = ctypes.cdll.LoadLibrary("./approxInv.so")
# approxInv = ctypes.CDLL("./approxInv.so", mode=ctypes.RTLD_GLOBAL)
approxInv.approxInv.restype = ctypes.c_int
approxInv.approxInv.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int, ctypes.c_double,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

def fSubs(P, L, b):
    if len(b.shape) == 1:
        c = P.dot(b.reshape(-1,1))
    else:
        c = P.dot(b)
    n = c.shape[0]
    y = np.zeros(c.shape)
    y[0,:] = c[0,:]/L[0,0]
    for i in range(1,n):
        y[i,:] = (c[i,:]-L[i,:i].reshape(1,-1).dot(y[:i,:]))/L[i,i]
    if len(b.shape) == 1:
        y = y.ravel()
    return y

def bSubs(U, y):
    if len(y.shape) == 1:
        w = y.reshape(-1,1)
    else:
        w = y
    n = w.shape[0]
    x = np.zeros(w.shape)
    x[n-1,:] = w[n-1,:]/U[n-1,n-1]
    for i in reversed(range(0,n-1)):
        x[i,:] = (w[i,:]-U[i,i+1:].reshape(1,-1).dot(x[i+1:,:]))/U[i,i]
    if len(y.shape) == 1:
        x = x.ravel()
    return x

def solve(P, L, U, b):
    y = fSubs(P, L, b)
    x = bSubs(U, y)
    return x

def initInv(H):
    # G = H.T/(np.linalg.norm(H)**2)
    G = H
    # P, L, U = la.lu(G)
    # V = solve(P, L, U, np.eye(G.shape[0]))
    V = np.linalg.solve(G, np.eye(G.shape[0]))
    return V

def iterInv1(H1, X2, order=3, tol=1e-6):
    H2 = X2.dot(X2.T)
    I = np.eye(H2.shape[0])
    H = H1 + H2
    G = initInv(H)
    # alpha = np.max(np.sum(np.abs(H.dot(H.T)), axis=1))
    # G = (2.0/alpha)*H.T
    Y = I-H.dot(G)
    fNorm = np.sqrt(np.sum(Y**2))
    # print('|I - F.F_i| = {:f}'.format(fNorm))
    while fNorm > tol:
        Z = I + Y
        for _ in range(order-1): # 2n^2
            Z = I +Y.dot(Z)
        G = G.dot(Z) # n^2
        Y = I-H.dot(G)
        fNorm = np.sqrt(np.sum(Y**2))
        # print('|I - F.F_i| = {:f}'.format(fNorm))
    return G

# Schulz iteration
def iterInv2(H1, X2, order=3, tol=1e-6):
    H2 = X2.dot(X2.T)
    I = np.eye(H2.shape[0])
    H = H1 + H2
    alpha = np.max(np.sum(np.abs(H.dot(H.T)), axis=1))
    G = (2.0/alpha)*H.T
    G0 = np.zeros(G.shape)
    fNorm = np.sqrt(np.sum((G-G0)**2))
    # print('|G - G_prv| = {:f}'.format(fNorm))
    while fNorm > tol:
        G0 = cp.copy(G)
        Y = I-H.dot(G) # n^2
        Z = I + Y
        for _ in range(order-1): # 3n^2
            Z = I +Y.dot(Z)
        G = G.dot(Z) # n^2
        fNorm = np.sqrt(np.sum((G-G0)**2))
        # print('|G - G_prv| = {:f}'.format(fNorm))
        # G = 2*G -G.dot(H).dot(G) # 2*k*n^2
    return G

# Schulz iteration
def iterInv3(H1, X2, tol=1e-6):
    H2 = X2.dot(X2.T)
    I = np.eye(H2.shape[0])
    H = H1 + H2
    alpha = np.max(np.sum(np.abs(H.dot(H.T)), axis=1))
    G = (2.0/alpha)*H.T
    G0 = np.zeros(G.shape)
    fNorm = np.sqrt(np.sum((G-G0)**2))
    # print('|G - G_prv| = {:f}'.format(fNorm))
    while fNorm > tol:
        G0 = cp.copy(G)
        G = 2*G.dot(I -H.dot(G)) # 2*k*n^2
        fNorm = np.sqrt(np.sum((G-G0)**2))
        # print('|G - G_prv| = {:f}'.format(fNorm))
    return G

def iterInv4(H1, X2, tol=1e-6):
    H2 = X2.dot(X2.T)
    I = np.eye(H2.shape[0])
    H = H1 + H2
    alpha = np.max(np.sum(np.abs(H.dot(H.T)), axis=1))
    G = (2.0/alpha)*H.T
    P = H.dot(G)
    Y = P-I
    I2 = 2*I
    I3 = 3*I
    I7 = 7*I
    I13 = 13*I
    I15 = 15*I 
    fNorm = np.sqrt(np.sum(Y**2))
    # print('|I - F.F_i| = {:f}'.format(fNorm))
    while fNorm > tol:
        Z = I3 +P.dot(-I2 +Y) # n^2
        V = P.dot(Z) # n^2
        G = -0.25*G.dot(Z).dot(-I13 +V.dot(I15 +V.dot(-I7 +V))) # 4n^2
        P = H.dot(G) # n^2
        Y = P-I # n^2
        fNorm = np.sqrt(np.sum(Y**2))
        # print('|I - F.F_i| = {:f}'.format(fNorm))
    return G

def softAbs(M, tol=1e-6):
    L, V = np.linalg.eig(M)
    L_t = np.diag(L/np.tanh(L/tol))
    M_t = V.dot(L_t.dot(V.T))
    return M_t

def nearestSPD(A, f=1e-6, tol=1e-12):
    B = (A + A.T)/2
    L1b, Z1c = np.linalg.eig(B)
    # bound factor
    M = max(0, -np.min(L1b))
    C = (A - A.T)/2
    L1c, Z1c = np.linalg.eig(C)
    # Spectral radius
    rho_C = np.max(np.abs(L1c))
    # C^2 . u = L^2 . u
    L = L1c**2
    Z = Z1c
    B = Z.T.dot(B).dot(Z)
    b_ii = np.diag(B)
    ind = b_ii < 0
    if np.any(ind):
        S = np.sqrt(np.max(b_ii[ind]**2-L[ind]))
    else:
        S = 0.0
    # Bounds
    alpha = max([rho_C, S, M])
    beta = rho_C + M
    try:
        X_a = B + np.sqrt(alpha**2 +np.diag(L))
        _ = np.linalg.cholesky(X_a)
        beta = alpha
        return X_a
    except:
        while (beta-alpha)/2 > max(f*alpha, tol):
            r = (alpha +beta)/2
            X_r = B + np.sqrt(alpha**2 +np.diag(L))
            try:
                _ = np.linalg.cholesky(X_r)
                beta = r
            except:
                alpha = r
        return B + np.sqrt(beta**2 +np.diag(L))

d_x = 1000
d_y = 1000
h = np.random.randn(d_x,1)
H = h.dot(h.T) + 10*np.random.rand(1)*np.eye(d_x)

# gamma = np.random.randn(d_x, d_y)
# delta = np.random.randn(d_x, d_y)

# # Method 1 - sequential
# H1 = cp.copy(H)
# for i in range(d_y):
#     g = gamma[:,i].reshape(-1,1)
#     d = delta[:,i].reshape(-1,1)
#     h = H1.dot(g)
#     H1 = H1 +d.dot(d.T)/(d.T.dot(g)) -h.dot(h.T)/(g.T.dot(h))

# # Method 2 - mean
# H2 = np.zeros((d_x,d_x))
# for i in range(d_y):
#     g = gamma[:,i].reshape(-1,1)
#     d = delta[:,i].reshape(-1,1)
#     h = H.dot(g)
#     H2_i = H +d.dot(d.T)/(d.T.dot(g)) -h.dot(h.T)/(g.T.dot(h))
#     H2 += H2_i/d_y

# # Method 3 - mean of arguments
# g = np.mean(gamma, axis=1).reshape(-1,1)
# d = np.mean(delta, axis=1).reshape(-1,1)
# h = H.dot(g)
# H3 = H +d.dot(d.T)/(d.T.dot(g)) -h.dot(h.T)/(g.T.dot(h))

# # Method 4 - sum of arguments
# g = np.sum(gamma, axis=1).reshape(-1,1)
# d = np.sum(delta, axis=1).reshape(-1,1)
# h = H.dot(g)
# H4 = H +d.dot(d.T)/(d.T.dot(g)) -h.dot(h.T)/(g.T.dot(h))

# # Method 5 - all at once
# H5 = np.copy(H)
# for i in range(d_y):
#     g = gamma[:,i].reshape(-1,1)
#     d = delta[:,i].reshape(-1,1)
#     h = H.dot(g)
#     H5 += d.dot(d.T)/(d.T.dot(g)) -h.dot(h.T)/(g.T.dot(h))

# # Method 6 - denominator based on Frobenius norm
# g = gamma
# d = delta
# h = H.dot(g)
# H6 = H +d.dot(d.T)/np.linalg.norm(d.T.dot(g)) -h.dot(h.T)/np.linalg.norm(g.T.dot(h))

# # Bad Broyden's method
# g = gamma
# d = delta
# h = H.dot(g)
# H7 = H + (d-h).dot(g.T)/(np.linalg.norm(g)**2)

# # Almost exact
# # gamma_inv = np.diag(1.0/np.diag(gamma.T.dot(gamma))).dot(gamma.T)
# # H8 = delta.dot(gamma_inv)
# g = gamma
# d = delta
# h = H.dot(g)
# A1 = d.T.dot(g)
# A2 = g.T.dot(h)
# A1d = np.diag(1.0/np.diag(A1))
# A2d = np.diag(1.0/np.diag(A2))
# M = np.fliplr(np.eye(d_y))
# A1o = np.diag(1.0/np.diag(A1))
# A2o = np.diag(1.0/np.diag(A2))
# H8 = H +d.dot(A1).dot(d.T) -h.dot(A2).dot(h.T)

# # Exact
# gamma_inv = np.linalg.solve(gamma.T.dot(gamma), gamma.T)
# H9 = delta.dot(gamma_inv)
# print('d =\n{:}\nH*g =\n{:}'.format(delta, H9.dot(gamma)))

# print('H1 = \n{:}\nH2 = \n{:}\nH3 = \n{:}\nH4 = \n{:}\nH5 = \n{:}\nH6 = \n{:}\nH7 = \n{:}\nH8 = \n{:}\nH9 = \n{:}\n'.format(H1, H2, H3, H4, H5, H6, H7, H8, H9))

# u = delta-H.dot(gamma)
# g_T_u = gamma.T.dot(u)
# g_T_u_inv = np.linalg.solve(g_T_u.dot(g_T_u.T), g_T_u)
# A = g_T_u_inv
# print(A.dot(g_T_u.T))

# Other tests
X = 10*np.random.randn(d_x,1500)
X1 = 10*np.random.randn(d_x,1500)
X2 = 10*np.random.randn(d_x,1500)
X = np.concatenate((X1,X2),axis=1)

delta = np.random.randn(d_x, d_y)
gamma = np.random.randn(d_x, d_y)

G = np.linalg.inv(X.dot(X.T))
H2 = X2.dot(X2.T)
G1 = np.linalg.inv(X1.dot(X1.T))

# # print('G =\n{:}'.format(G))
# # seq_G = []
# t = time.time()
# H2 = X2.dot(X2.T)
# L, U = np.linalg.eigh(H2)
# G_i = G1
# for i in range(U.shape[1]):
#     v = np.sqrt(L[i])*U[:,i].reshape(-1,1)
#     g = G_i.dot(v)
#     G_i = G_i - g.dot(g.T)/(1.0+v.T.dot(g))
#     # seq_G.append(G_i)
#     # print('Rank-{:}: G_app =\n{:}'.format(i+1, G_i))
# print('Based on eigen decomposition: {:} s'.format(time.time()-t))
# print('|G - G_app| = {:}'.format(np.linalg.norm(G-G_i)))
# # print('G_app =\n{:}'.format(G_i))

# t = time.time()
# G_i = G1
# # print('G =\n{:}'.format(G))
# for i in range(X2.shape[1]):
#     v = X2[:,i].reshape(-1,1)
#     g = G_i.dot(v)
#     G_i = G_i - g.dot(g.T)/(1.0+v.T.dot(g))
#     # seq_G.append(G_i)
#     # print('Rank-{:}: G_app =\n{:}'.format(i+1, G_i))
# print('Based on columns: {:} s'.format(time.time()-t))
# print('|G - G_app| = {:}'.format(np.linalg.norm(G-G_i)))
# # print('G_app =\n{:}'.format(G_i))

H1 = X1.dot(X1.T)
t = time.time()
H2 = X2.dot(X2.T)
L, V = np.linalg.eigh(H1+H2)
U = V
# U, L, V = np.linalg.svd(H1+H2, hermitian=True)
G_i = V.T.dot(np.diag(1.0/L)).dot(U)
print('Based on eigendecomposition: {:} s'.format(time.time()-t))
print('|G - G_app| = {:}'.format(np.linalg.norm(G-G_i)))
# print('G_app =\n{:}'.format(G_i))

t = time.time()
H2 = X2.dot(X2.T)
I = np.eye(d_x)
P = H2.dot(G1)
# G_i = G1 -G1.dot(np.linalg.solve(np.linalg.inv(H2) + G1, G1))
G_i = G1.dot(I -np.linalg.solve(I + P, P))
print('Based on Woodbury identity: {:} s'.format(time.time()-t))
print('|G - G_app| = {:}'.format(np.linalg.norm(G-G_i)))
# print('G_app =\n{:}'.format(G_i))

t = time.time()
H2 = X2.dot(X2.T)
H_i = H1 + H2
G_i = np.linalg.inv(H_i)
# G_i = np.linalg.solve(H_i, np.eye(d_x))
print('Based on direct inversion: {:} s'.format(time.time()-t))
print('|G - G_app| = {:}'.format(np.linalg.norm(G-G_i)))
# print('G =\n{:}'.format(G_i))

# Schulz iteration O(6*n^2)
t = time.time()
# G_i = iterInv1(H1, X2, order=3, tol=1e-2)
G_i = initInv(H1+X2.dot(X2.T))
# G_i = iterInv3(H1, X2, tol=1e-2)
# G_i = iterInv4(H1, X2, tol=1e-2)
print('Based on Schulz iteration 1: {:} s'.format(time.time()-t))
print('|G - G_app| = {:}'.format(np.linalg.norm(G-G_i)))
# print('G =\n{:}\n'.format(G[:3,:3]))
# print('G_app =\n{:}'.format(G_i[:3,:3]))

t = time.time()
H2 = X2.dot(X2.T)
F_i = H1 + H2
F = np.zeros(F_i.T.shape)
ret = approxInv.approxInv(F_i, d_x, 2, 1e-2, F)
G_i = F
print('Based on Schulz iteration 2: {:} s'.format(time.time()-t))
print('|G - G_app| = {:}'.format(np.linalg.norm(G-G_i)))
print('G =\n{:}\n'.format(G[:3,:3]))
print('G_app =\n{:}'.format(G_i[:3,:3]))
