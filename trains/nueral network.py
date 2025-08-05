import numpy as np
import matplotlib.pyplot as plt

#nueral network:
def predict(x,theta,b):
    scores = x @ theta.T + b
    return scores

def softmax(z):
    e_z = np.exp(z - np.max(z , axis=1,keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def Relu(x):
    return np.maximum(0,x)

#cost function:
def cross_entropy(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


#data:0 for A-Z 1 for a-z  2 for 0-9
x = ['A','B','c','a','b','C','0','1','2','3']
y = [0,0,1,1,1,0,2,2,2,2]
#y one hot:
def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out

y = one_hot(y, 3)
#encode x:
x_encode = np.array([ord(c) for c in x])
#nomalize:
mean = np.mean(x_encode,axis=0)
std = np.std(x_encode,axis=0)
x_normal = (x_encode - mean) / std
x_normal = x_normal.reshape(-1,1)
#theta:
theta1 = np.random.randn(3, 1) * 0.01
theta2 = np.random.randn(3, 3) * 0.01
b1 = np.zeros((3,))
b2 = np.zeros((3,))
#train:
learning_rate = 0.125
epochs = 1000
loss_history = []
for epoch in range(epochs):
    #forward:
    h = x_normal @ theta1.T + b1
    h = np.maximum(0, h)
    scores = h @ theta2.T +b2
    y_pred = softmax(scores)
    loss = cross_entropy(y,y_pred)
    loss_history.append(loss)
    #backward:
    dscores = y_pred - y
    dtheta2 = dscores.T @ h
    db2 = np.sum(dscores,axis=0)

    dh = dscores @ theta2
    dh[h <= 0] = 0

    dtheta1 = dh.T @ x_normal
    db1 = np.sum(dh, axis=0)

    np.clip(dtheta1, -1, 1, out=dtheta1)
    np.clip(dtheta2, -1, 1, out=dtheta2)
    np.clip(db1, -1, 1, out=db1)
    np.clip(db2, -1, 1, out=db2)

    theta2 -= learning_rate * dtheta2
    b2 -= learning_rate * db2

    theta1 -= learning_rate * dtheta1
    b1 -= learning_rate * db1
    if epoch % 100 == 0:
        print(f"Epoch : {epoch} , Cost = {loss}")

#draw for test learning rate
# plt.plot(loss_history)
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title(f"Loss curve (Learning Rate = {learning_rate})")
# plt.grid(True)
# plt.show()

#test:
X_input = input("Enter a chractor : ")
X_encode = np.array([ord(v) for v in X_input])
X_normal = (X_encode - mean) / std
X_normal = X_normal.reshape(-1, 1)

h_test = np.maximum(0,X_normal @ theta1.T + b1)
scores_test = h_test @ theta2.T + b2
y_pred_test = softmax(scores_test)

y_pred_class = np.argmax(y_pred_test,axis=1)[0]

labels = {0: "Uppercase Letter (A-Z)", 1: "Lowercase Letter (a-z)", 2: "Digit (0-9)"}
print("Predicted Class:", labels[y_pred_class])

quit = input("Press eny key for quit.")