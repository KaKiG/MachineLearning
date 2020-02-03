import numpy as np
from my_ml.LinearRegression import LinearRegression
import my_ml.Ridge as MR
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

exp_data = np.loadtxt(fname="expression.txt")
SNP_data = np.loadtxt(fname="SNPs.txt")
gene_list = np.loadtxt(fname="gene_list.txt", dtype="str")
strain_list = np.loadtxt(fname="strain_list.txt", dtype="str")

###(a)
coeffs = []
lin_reg = LinearRegression()
for i in range(SNP_data.shape[1]):
    lin_reg.fit_normal(SNP_data[:, i].reshape(-1, 1),
                       exp_data[:, 394].reshape(-1, 1))
    coeff = lin_reg.coef_.tolist()
    coeffs.append(coeff[0][0])


###(b)
lin_reg2 = LinearRegression()
lin_reg2.fit_normal(SNP_data[:, :2], exp_data[:, 394].reshape(-1, 1))
print(lin_reg2.coef_)

###(c)
lin_ridge = Ridge(alpha=0.2)
coeffs3 = []
lin_ridge.fit(SNP_data, exp_data[:,394])
coeffs3 = lin_ridge.coef_
print(coeffs3)

coeffs3my =[]
lin_ridge_my = MR.Ridge(alpha=0.2)
coeffs3my = []
lin_ridge_my.fit_normal(SNP_data, exp_data[:,394])
coeffs3my = lin_ridge_my.coef_
print(coeffs3my)



###(d)
lin_ridge2 = Ridge(alpha=200)
coeffs4 = []
lin_ridge2.fit(SNP_data, exp_data[:,394])
coeffs4 = lin_ridge2.coef_
print(coeffs4)

coeffs4my =[]
lin_ridge_my2 = MR.Ridge(alpha=200)
coeffs4my = []
lin_ridge_my2.fit_normal(SNP_data, exp_data[:,394])
coeffs4my = lin_ridge_my2.coef_
print(coeffs4my)
##test-------------------------------------------------------------------------------------------
# lin_ridge_my = MR.Ridge(0)
# lin_ridge_sk = Ridge(0)
# coeffs_my = []
# coeffs_sk = []
#
# def costFunc(X_train, y_train, theta, alpha):
#     X = np.hstack([np.ones((len(X_train), 1)), X_train])
#     theta = np.array(theta)
#     return np.sum((y_train-X.dot(theta)).T.dot(y_train-X.dot(theta))) + alpha * np.sum((theta ** 2))
#
# X_train = SNP_data[:,0].reshape(-1,1)
# y_train = exp_data[:, 394].reshape(-1,1)
# lin_ridge_my.fit_normal(X_train, y_train)
# lin_ridge_sk.fit_normal(X_train, y_train)
#
# coeffs_my.append(lin_ridge_my.intercept_)
# coeffs_my.append(lin_ridge_my.coef_)
#
# coeffs_sk.append(lin_ridge_sk.intercept_)
# coeffs_sk.append(lin_ridge_sk.coef_)
# print(coeffs_my, coeffs_sk)
# print(costFunc(X_train, y_train, coeffs_my, 200))
# print(costFunc(X_train, y_train, coeffs_sk, 200))


# for i in range (SNP_data.shape[1]-1200):
#     lin_ridge.fit_normal(SNP_data[:,i].reshape(-1,1), exp_data[:, 394].reshape(-1,1))
#     coeff = lin_ridge.coef_.tolist()
#     coeffs4.append(coeff[0][0])
