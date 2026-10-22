import numpy as np 

real_x = np.array([
    [2020],
    [2021],
    [2022],
    [2023],
    [2024],
    [2025],
    [2026]
])

x = real_x-2020

# # this function if x contain 1 feature
# def polyFeatures(x,order):
#     return np.hstack(([x**i for i in range(1,order+1)]))

from sklearn.preprocessing import PolynomialFeatures
def polyFeatures(x,order):
    poly = PolynomialFeatures(degree=order)
    return poly.fit_transform(x)

x_new = polyFeatures(x,3)

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
    [0.5],
    [0.3],
    [0.8]
])


lr = 0.0001
m = len(x)

for i in range(1000):
    h = x_new @ w
    E = h - y
    w = w-(lr/m)*x_new.T@E

h = x_new @ w

print(w)
print("______________")
print(h)
