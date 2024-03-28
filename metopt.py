import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

import scipy

GLOBAL_F_CNT = 0
GLOBAL_GRAD_F_CNT = 0


def grad(f, x, h=1e-5):
    global GLOBAL_GRAD_F_CNT
    GLOBAL_GRAD_F_CNT += 1
    return np.diag(np.eye(x.size) *
                   (f(x[:, np.newaxis] + h * np.eye(x.size)) - f(x[:, np.newaxis] - h * np.eye(x.size))) / (x.size * h)
                   )


def dichotomies(f, a, b, eps):
    global GLOBAL_F_CNT
    while abs(b - a) > eps:
        GLOBAL_F_CNT += 2
        c = (a + b) / 2
        delta = (b - a) / 8
        f1 = f(c - delta)
        f2 = f(c + delta)
        if f1 < f2:
            b = c
        else:
            a = c
    return (a + b) / 2


def golden_ratio(f, a, b, eps):
    global GLOBAL_F_CNT
    phi = (1 + sqrt(5)) / 2
    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi
    f1 = f(x1)
    f2 = f(x2)
    while abs(b - a) > eps:
        GLOBAL_F_CNT += 1
        if f1 > f2:
            a = x1
            x1 = x2
            x2 = b - (x1 - a)
            f2 = f(x2)
        else:
            b = x2
            x2 = x1
            x1 = a + (b - x2)
            f1 = f(x1)
    return (a + b) / 2


def grad_descent_with(func, dim):
    def inner(f, x, epochs=10, eps=0.1, eps_gradient=0.1):
        x = np.array(x, dtype=np.float64)
        history = np.zeros((epochs + 1, dim))
        history[0] = x
        for epoch in range(1, epochs + 1):
            grad_value = grad(f, x)
            lr = func(lambda t: f(x - t * grad_value), 0, 1, eps)
            x = x - lr * grad(f, x)
            history[epoch] = x
            if epoch > 0 and np.linalg.norm(f(history[epoch]) - f(history[epoch - 1])) < eps_gradient:
                return history[:epoch]
        return history

    return inner


def nelder_mead(function, start_values, epochs=1000, eps=1e-3, eps_gradient=1e-3):
    res = scipy.optimize.minimize(function, start_values, method='Nelder-Mead',
                                  options={'xatol': eps_gradient, 'disp': True, 'return_all': True, 'maxiter': epochs})
    vecs = res.get('allvecs')
    return vecs


dimension = 2

methods = {
    "dichotomies": grad_descent_with(dichotomies, dimension),
    "golden_ratio": grad_descent_with(golden_ratio, dimension),
    "nelder_mead": nelder_mead,
    "const_step": grad_descent_with(lambda *x: 1e-5, dimension)
}

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)


def draw(diap, eps, function, start_values, title, vecs):
    xk = history[:, 0]
    yk = history[:, 1]
    x = np.arange(-diap[0], diap[0], diap[0] / 100)
    y = np.arange(-diap[1], diap[1], diap[1] / 100)
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y])
    cp = ax2.contour(X, Y, Z, levels=sorted(set(function(point) for point in vecs)))
    ax2.clabel(cp, inline=1, fontsize=10)
    ax2.plot(xk, yk, color='blue')
    ax.plot(xk, yk, function(np.array([xk, yk])), color='blue')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='cyan', edgecolor='none', alpha=0.3)
    fig.suptitle(f"{title}, epsilon = {eps}, начальная точка: {start_values}")


def quadratic(x):
    sum = 0
    for i in range(len(x)):
        sum += (i + 1) * x[i] ** 2
    return sum

def noisy_function(x):
    return (x[0] - 1) ** 2 + (x[1] - 1) ** 2 + np.random.uniform(0, 0.01)


titles_function = ["(x - 1)^2 + (y - 1)^2",
                   "x^2 - x * y + 5 * y^2",
                   "x^2 + 1000*y^2 + x * y",
                   "(x^2 + y - 11)^2 + (x + y^2 - 7)^2",
                   "",
                   "",
                   "(x - 1)^2 + (y - 1)^2 + uniform(0, 0.01)"
                   ]

functions = [
    lambda x: (x[0] - 1) ** 2 + (x[1] - 1) ** 2,
    lambda x: x[0] ** 2 - x[0] * x[1] + 5 * x[1] ** 2,

    # плохо обусловленная
    lambda x: x[0] ** 2 + 1000 * x[1] ** 2,

    # мультимодальные Химмельблау
    lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2,

    # rosenbrock
    lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2,

    # функция для n переменных
    quadratic,

    noisy_function
]

diap = np.array([-100, 100])
start_values = np.array([80, 80])
eps2 = 1e-3
history = np.array(methods["nelder_mead"](functions[1], start_values, epochs=100000, eps=1e-3, eps_gradient=eps2))
print(f"Полученная точка: {history[-1]}")
print(
    f"Кол-во итераций для поиска (вызовы функции): {GLOBAL_F_CNT}, кол-во итераций для спуска (вызовы градиента): {GLOBAL_GRAD_F_CNT}, Число итераций {len(history)}")
draw(diap, eps2, functions[1], start_values, titles_function[1], history)

plt.show()