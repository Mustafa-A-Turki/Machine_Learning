import numpy as np 

x = np.array([
    [2020],
    [2021],
    [2022],
    [2023],
    [2024],
    [2025],
    [2026]
])

y = np.array([
        [400],
        [500],
        [550],
        [600],
        [750],
        [780],
        [800]
])


w = np.array([
    [1],
    [0.5]
])

x_new = np.hstack((np.ones(shape=x.shape),x))


lr = 0.0000001
m = len(x)

for i in range(10000):
    h = x_new @ w
    E = h - y
    w = w-(lr/m)*x_new.T@E

h = x_new @ w

print(w)
print("______________")
print(h)
