#This model geusses the price of homes in Pardisan in Qom
#mainprogram:
def main():
    print("Hi , this model geusses the price of homes in Pardisan in Qom , let's go \n")
    for i in range(1,4):
        print(i)
        sleep(1)

#costfunction
def costfunction(x,y,tetha):
    error = x @ tetha - y
    return 0.5 * (error.T @ error)[0,0]

#Import packages:
import numpy as np
import numpy.linalg as np_lin
from time import sleep

#data:
#         size  bed   floor    year
x_train = np.array([[1,76,2,2,1398],
              [1,65,1,5,1391],
              [1,81,2,5,1394],
              [1,98,2,3,1404],
              [1,126,3,4,1398]])

#prices:
y_train = np.array([#miliard toman(000,000)
                    [2080],
                    [2050],
                    [2160],
                    [4500],
                    [4450]])

#matrix I:
I = np.eye(5)
I[0][0] = 0

#tetha(regulaziation):
tetha = np_lin.inv(x_train.T @ x_train + 0.001 * I) @ x_train.T @ y_train

#print(tetha):
"""[[-2.10836718e+05]
[ 5.36753401e+01]
[-7.90713012e+02]
[ 1.86725355e+02]
[ 1.50299748e+02]]"""

#mainprogram call:
main()

#cost function for data_train:
print("\nCosts for data train : ",costfunction(x_train,y_train,tetha),"\n")

#new data guesses:
size  = int(input("Enter size : "))
bedrooms = int(input("Enter bedrooms : "))
floor = int(input("Enter floor : "))
year = int(input("Enter year : "))
geuss = -2.10836718e+05 + 5.36753401e+01 * size - 7.90713012e+02 *bedrooms + 1.86725355e+02 * floor + 1.50299748e+02 * year
print("The geuss price : ",int(geuss),"000,000")
quit = input("Enter eny key for quit.")

"""
My model needs to understand what floor this house is on.
Right now it can only understand what floor it is on!
"""
