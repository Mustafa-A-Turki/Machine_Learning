import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

data = pd.read_csv("data/IceCream.csv")

data_train_size = int(len(data)*0.7)

x_train = np.array(data['Temperature'][0:data_train_size])
y_train = np.array(data['Revenue'][0:data_train_size])

losses = []

def fit(x,y):
    x = x.reshape(len(x),1)
    y = y.reshape(len(y),1)

    x_new = np.hstack((np.ones(shape=x.shape),x))

    lr = 0.001
    m = len(x_new)

    w = np.random.rand(2,1)


    for i in range(100000):
        h = x_new@w
        E = h-y
        losses.append(np.mean(np.abs(E)))
        w = w-(lr/m)*x_new.T@E

    return w

w = fit(x_train,y_train)

def predict(temperature):
    print(w[0]+(w[1]*temperature))


def show_predictions(n):
    for i in range(n):
        r = np.random.randint(data_train_size,len(data))
        print(f"Temperature: {data['Temperature'][r]}")
        print(f"Predicted Sales: {w[0]+(w[1]*data['Temperature'][r])}")
        print(f"Actual Sales: {data['Revenue'][r]}")
        print("_"*40)

def evaluate():
    x_test = np.array(data['Temperature'][data_train_size:])
    y_test = np.array(data['Revenue'][data_train_size:])

    predictions = w[0]+w[1]*x_test

    E = predictions - y_test
    MAE = np.mean(np.abs(E))
    return MAE


def show_fitLine():
    plt.scatter(x_train,y_train)
    plt.plot(x_train,w[0]+w[1]*x_train,color='blue')
    plt.xlabel('Temperature')
    plt.ylabel('Revenue')
    plt.show()


def show_lossFun():
    plt.plot(np.arange(1,len(losses)+1),np.array(losses))
    plt.xlabel('Iteration')
    plt.ylabel('MAE')
    plt.show()

predict(40)

show_predictions(int(input("Enter number of test: ")))

print(evaluate())

show_fitLine()

show_lossFun()
