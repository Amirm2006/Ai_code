#Hello, this model predicts whether your input is English letters or a number.
#import packages:
import numpy as np

#region data:
#letters a-z A-Z and digits 0 1 ... 9:
digits = [chr(i) for i in range(48,58)]
letters = [chr(i)for i in range(65,91)] + [chr(i) for i in range(97,123)]
#ascii codes:
x_digits = np.array([ord(c) for c in digits])
x_letters = np.array([ord(c) for c in letters])
#class 0 and 1:
y_digits = np.zeros(len(x_digits), dtype=int)
y_letters = np.ones(len(x_letters), dtype=int)
#data concatenate:
X = np.concatenate([x_digits,x_letters])
y = np.concatenate([y_digits,y_letters])
#nomarliziation:
X = X.reshape(-1,1) / 200
#class numbers:
num_classes = 2
#weight and baias:
np.random.seed(0)
W = np.random.randn(1,num_classes)
b = np.zeros((1,num_classes))
#endregion

#region functions:
def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def cross_antropy(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    return np.sum(log_likelihood) / m

def predict(X):
    scores = np.dot(X, W) + b
    probs = softmax(scores)
    return np.argmax(probs, axis=1), probs
#end region

#region train:
learning_rate = 0.5
epochs = 5000

# One-hot for lables:
y_one_hot = np.zeros((len(y), num_classes))
y_one_hot[np.arange(len(y)), y] = 1

# train
for epoch in range(epochs):
    #guess:
    scores = np.dot(X,W) + b
    probs = softmax(scores)

    #loss function:
    loss = cross_antropy(y,probs)

    #gradian:
    dscores = probs - y_one_hot
    dW = np.dot(X.T, dscores) / len(X)
    db = np.sum(dscores, axis=0, keepdims=True) / len(X)

    #update:
    W -= learning_rate * dW
    b -= learning_rate * db

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

#endregion

#region test
def classify(x):
    if str(x).isdigit():
        ascii_val = ord(str(x))#digit
    elif len(str(x)) == 1:
        ascii_val = ord(str(x))#letter
    else:
        return "Invalid input" , None

    x_input = np.array([ascii_val / 200.0])
    predict_class , probs = predict(x_input)
    return "Digits" if predict_class[0] == 0 else "Letters" , probs

#example:
x_input = input("Hi , give me a chractor(letters or digits):")
if x_input.isdigit():
    print(classify(int(x_input)))
elif x_input.isalpha():
    print(classify(x_input))
else:
    print("!!!!Invalid data!!!!")

quit = input("Press eny key to quit.")
#endregion