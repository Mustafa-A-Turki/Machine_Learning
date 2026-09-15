import numpy as np 

# year, fuel price
year = np.array([
    [2020],
    [2021],
    [2022],
    [2023],
    [2024],
    [2025],
    [2026]
])

fuel_price = np.array([
    [7.00],
    [7.50],
    [8.00],
    [10.00],
    [13.50],
    [19.00],
    [24.00]
])

x_new = np.hstack((year,fuel_price))
x_new = np.hstack((np.ones(shape=(len(year),1)),x_new))

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
    [0.3]
])

lr = 0.0000001
m = len(x_new)

for i in range(1000):
    h = x_new @ w
    E = h - y
    w = w-(lr/m)*x_new.T@E

h = x_new @ w

print(w)
print("______________")
print(h)