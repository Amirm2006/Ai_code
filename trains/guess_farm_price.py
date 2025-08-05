import numpy as np

def cost_function(y_true,y_pred):
    m = y_true.shape[0]
    return np.sum((y_true - y_pred)**2) / m

#data
X = np.array([[1,20,40],#800
              [1,30,50],#1500
              [1,10,20],#200
              [1,30,60]],dtype=np.float32)#1800

y = np.array([[150],
              [250],
              [50],
              [400]],dtype=np.float32)

tetha = np.zeros((X.shape[1], 1), dtype=np.float32)
"""print(tetha) 
[[-150. ] 
 [  -7.5] 
 [  12.5]]"""

m = y.shape[0]
# normalization:
X_no_bias = X[:, 1:]
means = X_no_bias.mean(axis=0)
stds = X_no_bias.std(axis=0, ddof=0)
X_scaled = (X_no_bias - means) / stds

# بازسازی X با بایاس
X = np.hstack([np.ones((m, 1)), X_scaled])


learning_rate = 0.5
epochs = 5000
m = len(y)

for epoch in range(epochs):
    h = X @ tetha
    grad = (1/m) * X.T @ (h - y)
    tetha -= learning_rate * grad

    if epoch % 100 == 0:
        loss = cost_function(y,h)
        print('epoch:',epoch,'loss',loss)
print("tetha:\n",tetha,"\n")

#test:
length_input = float(input("Enter length:"))
width_input = float(input("Enter width:"))
length_input_new = (length_input - 22.5) / (8.291562 + 1e-8)
width_input_new = (width_input - 42.5) / (14.790199 + 1e-8)
x_input = np.array([[1,length_input_new,width_input_new]])

price = x_input @ tetha
np.set_printoptions(formatter={'all':lambda x:'%.2f'%x})
print(f"price to dollar = {price.flatten()[0]}")