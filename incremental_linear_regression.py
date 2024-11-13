import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import sys
import random
import time

def prompt(msg):
    v = float(sys.version.split('.')[0])
    if v >= 3:
        return input(msg)
    else:
        return raw_input(msg)

def linear_regression(X, Y):
    N = X.shape[1]
    assert N  == Y.shape[1]
    x_m = np.sum(X, axis=1).reshape(-1,1)/N
    y_m = np.sum(Y, axis=1).reshape(-1,1)/N
    del_X = X-x_m
    del_Y = Y-y_m
    P_x = del_X.dot(del_X.T)/(N-1)
    P_yx = del_Y.dot(del_X.T)/(N-1)
    A = P_yx.dot(np.linalg.inv(P_x))
    B = y_m -A.dot(x_m)
    params = [A, B, x_m, y_m, P_x, P_yx, N]
    return params

def incremental_linear_regression(X, Y, params=None):
    L = X.shape[1]
    assert L  == Y.shape[1]
    if params is None:
        return linear_regression(X, Y)
    
    assert isinstance(params, list) and len(params) == 7
    _, _, x_m_N, y_m_N, P_x_N, P_yx_N, N = params
    # Updates
    x_m_L = np.sum(X, axis=1).reshape(-1,1)/L
    y_m_L = np.sum(Y, axis=1).reshape(-1,1)/L
    x_m = (N*x_m_N + L*x_m_L)/(N+L)
    y_m = (N*y_m_N + L*y_m_L)/(N+L)
    P_x = ((N-1)*P_x_N + (X-x_m_N).dot((X-x_m_N).T) -(L*x_m_N-L*x_m_L).dot((L*x_m_N-L*x_m_L).T)/(N+L))/(N+L-1)
    P_yx = ((N-1)*P_yx_N + (Y-y_m_N).dot((X-x_m_N).T) -(L*y_m_N-L*y_m_L).dot((L*x_m_N-L*x_m_L).T)/(N+L))/(N+L-1)
    A = P_yx.dot(np.linalg.inv(P_x))
    B = y_m -A.dot(x_m)
    params = [A, B, x_m, y_m, P_x, P_yx, N+L]
    return params

def lr_least_squares(X, Y):
    # Augment X
    X = np.vstack((X, np.ones((1,X.shape[1]))))
    # Compute Hessian
    H = X.dot(X.T)
    # Compute inverse Hessian
    H_inv = np.linalg.inv(H)
    # Compute weights
    W = H_inv.dot(X.dot(Y.T))
    params = [W, H_inv]
    return params

def L_c(X, Y, W):
    del_Y = Y-W.T.dot(X)
    return 0.5*np.sum(del_Y * del_Y)

def dL_da(X, Y, W, p, a):
    del_Y = Y-(W+a*p).T.dot(X)
    return np.sum(del_Y * (-p.T.dot(X)))

def optimal(X, Y, W, p):
    return -dL_da(X, Y, W, p, 0)/(np.linalg.norm(p.T.dot(X))**2)

def line_search(X, Y, W, a0, p):
    a = a0
    C_prv = L_c(X, Y, W)
    D_prv = dL_da(X, Y, W, p, a)
    for i in range(200):
        C = L_c(X, Y, W+a*p)
        D = dL_da(X, Y, W, p, a)
        if C < C_prv/2 and abs(D) < abs(D_prv)/20:
            break
        a -= C/D
        print('a = {:}, L_c = {:}, dL_da = {:}'.format(a, C, D))
    print('Step {:}: L_c = {:}, dL_da = {:}'.format(i, C, D))
    return a

# @profile
def lr_incremental_least_squares(X, Y, params=None, X_acc=None, Y_acc=None):
    L = X.shape[1]
    assert L  == Y.shape[1]
    if params is None:
        return lr_least_squares(X, Y)
    assert isinstance(params, list) and len(params) == 2
    W, H_inv = params
    d_h = H_inv.shape[0]
    # Updates
    O = np.ones((1, L))
    X_aug = np.vstack((X, O))
    # Update inverse of the Hessian
    # XXt = X_aug.dot(X_aug.T)
    # H_inv += XXt
    # XXt = np.zeros((d_h, d_h))
    # Lambda, U = np.linalg.eig(XXt)
    # for i in range(U.shape[1]):
    # for i in range(L):
    #     # v = np.sqrt(Lambda[i])*U[:,i].reshape(-1,1)
    #     v = X_aug[:,i].reshape(-1,1)
    #     g = H_inv.dot(v)
    #     H_inv -= g.dot(g.T)/(1.0+v.T.dot(g))
    v = np.mean(X_aug, axis=1).reshape(-1,1)
    g = H_inv.dot(v)
    H_inv -= g.dot(g.T)/(1.0+v.T.dot(g))


    del_Y = Y-W.T.dot(X_aug)
    p = H_inv.dot(X_aug).dot(del_Y.T)
    a = 0.1
    # print('Before: L_c = {:}, dL_da = {:}'.format(L_c(X_aug, Y, W), dL_da(X_aug, Y, W, p, 0)))
    if X_acc is not None and Y_acc is not None:
        O_acc = np.ones((1, X_acc.shape[1]))
        X_acc_aug = np.vstack((X_acc, O_acc))
        del_Y = Y_acc-W.T.dot(X_acc_aug)
        pt_X = p.T.dot(X_acc_aug)
        a = np.sum(del_Y * (pt_X))/np.sum(pt_X**2)
        # a = optimal(X_acc_aug, Y_acc, W, p)
        # a = line_search(X_acc_aug, Y_acc, W, a, p)
    else:
        pt_X = p.T.dot(X_aug)
        a = np.sum(del_Y * (pt_X))/np.sum(pt_X**2)
        # a = optimal(X_aug, Y, W, p)
        # a = line_search(X_aug, Y, W, a, p)
    # print('After: L_c = {:}, dL_da = {:}'.format(L_c(X_aug, Y, W+a*p), dL_da(X_aug, Y, W, p, a)))
    del_W = a*p
    W += del_W

    '''
    XXt = X_aug.dot(X_aug.T)
    z = XXt.dot(del_W)
    dW_zt = del_W.dot(z.T)
    # z_dWt = dW_zt.T
    zt_dW = np.sqrt(np.sum((z.T.dot(del_W))**2))
    I_h = np.eye(d_h)
    A_h = (I_h -dW_zt/zt_dW)
    H_inv = A_h*H_inv*A_h.T + del_W.dot(del_W.T)/zt_dW
    '''
    
    params = [W, H_inv]
    return params

# Size of the problem
d_x = 3
d_y = 3

order1 = 10
order2 = 1

# Generate instances
N = 1000 # total number of examples
L = 20 # batch size
# Sample actual slope matrix
A = order1*np.random.randn(d_y, d_x)
# Sample intercept vector
B = order1*np.random.randn(d_y, 1)

# Sample independent variable
s_x = order2*np.random.randn(d_x, 1)
P_x = s_x.dot(s_x.T) + order2*np.random.rand(1)*np.eye(d_x)
sqrt_P_x = np.linalg.cholesky(P_x)
X = order2*np.random.randn(d_x, N) + sqrt_P_x.dot(np.random.randn(d_x, N))
X_test = order2*np.random.randn(d_x, N) + sqrt_P_x.dot(np.random.randn(d_x, N))
X_test_aug = np.vstack((X_test, np.ones((1, N))))

# Sample dependent variable
s_y = order2*np.random.randn(d_y, 1)
P_y = s_y.dot(s_y.T) + order2*np.random.rand(1)*np.eye(d_y)
sqrt_P_y = np.linalg.cholesky(P_y)
Y = A.dot(X) + B + sqrt_P_y.dot(np.random.randn(d_y, N))
Y_test = A.dot(X_test) + B + sqrt_P_y.dot(np.random.randn(d_y, N))

# Divide samples in batches
batches = [(X[:,L*i:L*i+L],Y[:,L*i:L*i+L]) for i in range(int(N/L))]

fig, axes = plt.subplots(d_x, d_y, figsize=[4*d_x,4*d_y])
fig.show()

# Compute overall parameters
A_a, B_a, _, _, _, _, _ = linear_regression(X, Y)
k_min, k_max = np.argmin(X, axis=1), np.argmax(X, axis=1)
X_a = np.zeros((d_x, 2))
for i in range(d_x):
    X_a[i,0] = X[i,k_min[i]]
    X_a[i,1] = X[i,k_max[i]]
Y_a = A_a.dot(X_a) + B_a

params = None
X_acc = np.zeros((d_x,0))
Y_acc = np.zeros((d_y,0))
N_acc = 0
t = time.time()
for batch in batches:
    X_b, Y_b = batch
    # Accumulate data
    X_acc = np.concatenate((X_acc, X_b), axis=1)
    Y_acc = np.concatenate((Y_acc, Y_b), axis=1)
    N_acc += X_b.shape[1]
    # x_m_acc = np.sum(X_acc, axis=1).reshape(-1,1)/N_acc
    # y_m_acc = np.sum(Y_acc, axis=1).reshape(-1,1)/N_acc
    # P_x_acc = (X_acc-x_m_acc).dot((X_acc-x_m_acc).T)/(N_acc-1)
    # P_yx_acc = (Y_acc-y_m_acc).dot((X_acc-x_m_acc).T)/(N_acc-1)
    params = incremental_linear_regression(X_b, Y_b, params)
    A_b, B_b, x_m_b, y_m_b, P_x_b, P_yx_b, N_acc = params
    # print('P_x = {:}'.format(P_x_acc))
    # print('P_x_b = {:}'.format(P_x_b))
    # print('P_yx = {:}'.format(P_yx_acc))
    # print('P_yx_b = {:}'.format(P_yx_b))
    # print('Error in P_x = {:}'.format(np.sum((P_x_acc-P_x_b)**2)))
    # print('Error in P_yx = {:}'.format(np.sum((P_yx_acc-P_yx_b)**2)))
    # print('Error in A = {:}'.format(np.sum((A_a-A_b)**2)))
    # print('Error in B = {:}'.format(np.sum((B_a-B_b)**2)))
    # Predictions
    Y_prd = A_b.dot(X_a) + B_b
    for i, j in product(range(d_x), range(d_y)):
        if isinstance(axes, np.ndarray):
            ax = axes[i, j]
        else:
            ax = axes
        ax.clear()
        fig.suptitle('Linear regressor: cases {:d} / {:d}'.format(N_acc, N))
        ax.plot(X_acc[i,:], Y_acc[j,:], 'k.', label='Data')
        ax.plot(X_a[i,:], Y_a[j,:], 'r', label='Overall regressor')
        ax.plot(X_a[i,:], Y_prd[j,:], 'b--', label='Prediction')
        ax.legend()
        ax.grid()
        fig.canvas.draw()
    print('Data cases {:d} / {:d}'.format(N_acc, N))
    sys.stdout.flush()
print('Finished in {:} s'.format(time.time()-t))
print('A = \n{:}, \nA_b = \n{:}'.format(A, A_b))
print('B = \n{:}, \nB_b = \n{:}'.format(B, B_b))
print('Error in A = {:}'.format(np.sum((A-A_b)**2)))
print('Error in B = {:}'.format(np.sum((B-B_b)**2)))

# MSE, cross-validation
W1 = np.vstack((A_b.T, B_b.T))
MSE1 = 2*L_c(X_test_aug, Y_test, W1)/N
print('Cross-validation mean square error: {:}'.format(MSE1))

## Other method
plt.close('all')
fig, axes = plt.subplots(d_x, d_y, figsize=[4*d_x,4*d_y])
fig.show()

# Compute overall parameters by least squares
W_a, H_inv_a = lr_least_squares(X, Y)

# Y_a = W_a[:-1,:].T.dot(X_a) + W_a[-1,:].reshape(-1,1)
Y_a = W_a.T.dot(np.vstack((X_a, np.ones((1,X_a.shape[1])))))

params = None
X_acc = np.zeros((d_x,0))
Y_acc = np.zeros((d_y,0))
N_acc = 0
t = time.time()
for batch in batches:
    X_b, Y_b = batch
    # Accumulate data
    X_acc = np.concatenate((X_acc, X_b), axis=1)
    Y_acc = np.concatenate((Y_acc, Y_b), axis=1)
    L = X_b.shape[1]
    N_acc += L
    # ind = random.sample(range(N_acc), min(L, N_acc))
    ind = list(range(max(0,N_acc-2*L),max(L, N_acc-L)))
    params = lr_incremental_least_squares(X_b, Y_b, params, X_acc[:,ind], Y_acc[:,ind])
    # params = lr_incremental_least_squares(X_b, Y_b, params)
    W_b, H_inv_b = params
    # Predictions
    Y_prd = W_b.T.dot(np.vstack((X_a, np.ones((1,X_a.shape[1])))))
    for i, j in product(range(d_x), range(d_y)):
        if isinstance(axes, np.ndarray):
            ax = axes[i, j]
        else:
            ax = axes
        ax.clear()
        fig.suptitle('Linear perceptron: cases {:d} / {:d}'.format(N_acc, N))
        ax.plot(X_acc[i,:], Y_acc[j,:], 'k.', label='Data')
        ax.plot(X_a[i,:], Y_a[j,:], 'r', label='Overall regressor')
        ax.plot(X_a[i,:], Y_prd[j,:], 'b--', label='Prediction')
        ax.legend()
        ax.grid()
        fig.canvas.draw()
    print('Data cases {:d} / {:d}'.format(N_acc, N))
    sys.stdout.flush()
print('Finished in {:} s'.format(time.time()-t))
W_t = np.vstack((A.T, B.T))
print('W_t = \n{:}, \nW_b = \n{:}'.format(W_t, W_b))
print('Error in W = {:}'.format(np.sum((W_t-W_b)**2)))
# MSE, cross-validation
MSE2 = 2*L_c(X_test_aug, Y_test, W_b)/N
print('Cross-validation mean square error: {:}'.format(MSE2))

plt.pause(0.001)
_ = prompt("Aperte [ENTER] pra finalizar: ")
plt.close("all")