import numpy as np

# load and preprocess data
with open("genes.txt", "r") as file:
    genes = file.readlines()
expr = np.loadtxt("model.txt")
geneDic = {}
for i in range(len(genes)):
    geneDic[genes[i][:-1]] = i

# order is NCAPG FOS FOXA1 ESR1
order = ["NCAPG", "FOS", "FOXA1", "ESR1"]
orderIndex = []

for i in range(len(order)):
    orderIndex.append(geneDic[order[i]])

samples = np.zeros((100, 4))

# taking 100 samples in this order
for i in range(100):
    for j in range(4):
        if j != 0:
            samples[i, j] = np.random.normal(
                loc=expr[orderIndex[j], 17]+expr[orderIndex[j], orderIndex[j-1]]*samples[i,j-1], scale=np.sqrt(expr[orderIndex[j], 18])
            )
        else:
            samples[i, j] = np.random.normal(
                loc=expr[orderIndex[j], 17], scale=np.sqrt(expr[orderIndex[j], 18])
            )

# compute mean and variance for ESR1
mean, variance = np.mean(samples[:, 3]), np.var(samples[:, 3])
print(mean)
print(variance)

# ----------------------------------------------------------
# function for solving linear regression with formula
def LinearRegression(X_train, y_train):
    # combine the training data with "1" array
    X = np.hstack([np.ones((len(X_train), 1)), X_train])

    # based on the closed form expression, compute the result directly
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_train)

    # set coeffs to variable
    intercept = theta[0]
    coef = theta[1:]

    return intercept, coef


# compute beta_i for each gene
beta = []
for i in range(4):
    y_train = samples[:, i].reshape(-1, 1)
    if i == 0:
        beta_i = 0
    else:
        X_train = samples[:, i - 1].reshape(-1, 1)
        beta_0, beta_i = LinearRegression(X_train, y_train)
    beta.append(beta_i)
print(beta)
