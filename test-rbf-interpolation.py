import numpy as np
import matplotlib.pyplot as plt

use_chebyshev_nodes = True
use_samples = True
add_noise = True
incremental = True

a, b = (0, 1)
N = 20
N_s = 100


def f(x):
    return np.exp(x * np.cos(3.0 * np.pi * x))


def phi(x, x_i, epsilon=3.0):
    return np.exp(-(epsilon * (x - x_i)) ** 2)


def s(x, x_i, w_i, epsilon=3.0):
    n = x_i.shape[0]
    x_ = np.tile(x, (n, 1))
    x_i_ = x_i[:, None]
    return np.dot(w_i, phi(x_, x_i_, epsilon=epsilon))


if use_chebyshev_nodes:
    X_i = ((a + b) / 2) + ((b - a) / 2) * np.cos(
        (2 * np.arange(N) + 1) * np.pi / (2 * N)
    )
else:
    X_i = np.linspace(a, b, N)


Y_i = f(X_i)
if use_samples:
    X_s = np.random.rand(N_s)
    Y_s = f(X_s)
    if add_noise:
        Y_s += 0.25 * np.random.randn(N_s)
else:
    X_s = X_i
    Y_s = f(X_s)

X = np.linspace(a, b, 1001)
Y = f(X)

if incremental:
    indices = np.random.permutation(np.arange(N_s))
    N_b = 10
    batch_size = np.ceil(N_s / N_b).astype(int)
    batches = []
    for i in range(N_b):
        batch_indices = indices[i * batch_size:min((i + 1) * batch_size, N_s)]
        X_b = X_s[batch_indices]
        Y_b = f(X_b)
        batches.append((i, X_b, Y_b))
    K_prv = np.zeros((N, N))
    H_prv = np.zeros((N, N))
    W_prv = np.zeros((N, ))
    z_prv = np.zeros((N, ))
    Id = np.eye(N)

    # K = phi(np.tile(X_s, (N, 1)), X_i[:, None])
    # W_i = np.linalg.solve(K.dot(K.T), K.dot(Y_s))
    # Y_h = s(X, X_i, W_i)

    # H = K.dot(K.T)

    # _, X_b1, Y_b1 = batches[0]
    # _, X_b2, Y_b2 = batches[1]
    # K1 = phi(np.tile(X_b1, (N, 1)), X_i[:, None])
    # K2 = phi(np.tile(X_b2, (N, 1)), X_i[:, None])
    # H1 = K1.dot(K1.T)
    # H2 = K2.dot(K2.T)
    # G1 = np.linalg.pinv(H1)
    # P = H2.dot(G1)
    # G_ = G1.dot(Id - np.linalg.solve(Id + P, P))

    # W1 = np.linalg.solve(H1, K1.dot(Y_b1))
    # W2 = np.linalg.solve(H1 + H2, H1.dot(W1) + K2.dot(Y_b2))

    # print(f"|G - G_i| = {np.linalg.norm(W_i - W2)}")

    for i, X_b, Y_b in batches:
        K_cur = phi(np.tile(X_b, (N, 1)), X_i[:, None])
        H_cur = H_prv + K_cur.dot(K_cur.T)
        # W_cur = np.linalg.solve(
        #     H_cur,
        #     H_prv.dot(W_prv) + K_cur.dot(Y_b)
        # )
        z_cur = z_prv + K_cur.dot(Y_b)
        W_cur = np.linalg.solve(
            H_cur,
            z_cur
        )
        H_prv = H_cur
        W_prv = W_cur
        z_prv = z_cur
    W_i = W_cur
    Y_h = s(X, X_i, W_i)
    # Single batch (for comparison)
    K = phi(np.tile(X_s, (N, 1)), X_i[:, None])
    W_i_g = np.linalg.solve((K.dot(K.T) + 0 * Id), K.dot(Y_s))
    Y_h_g = s(X, X_i, W_i_g)
else:
    K = phi(np.tile(X_s, (N, 1)), X_i[:, None])
    W_i = np.linalg.solve(K.dot(K.T), K.dot(Y_s))
    Y_h = s(X, X_i, W_i)
plt.scatter(X_i, Y_i)
plt.plot(X, Y)
plt.plot(X, Y_h, linestyle='--')
plt.plot(X, Y_h_g, linestyle=':')
plt.show()
