import numpy as np
import matplotlib.pyplot as plt


# functions for Kmeans clustering
def EluDist(x, y):
    return np.sqrt(np.sum((x - y) * (x - y), axis=1))


def Loss(centroid, X, label):
    loss = 0
    for i in range(len(label)):
        loss += np.sum((centroid[label[i] - 1, :] - X[i, :]) ** 2)
    return loss


def Kmeans(k, X, label):
    centroid = np.zeros((k, X.shape[1]))
    for i in range(1, k + 1):
        ind = []
        for j in range(len(label)):
            if label[j] == i:
                ind.append(j)
        centroid[i - 1, :] = np.sum(X[ind], axis=0) / len(ind)

    for i in range(len(label)):
        label[i] = np.argmin(EluDist(centroid, X[i, :])) + 1

    return centroid, label


def Clustering(k, X, init_label):
    label = init_label
    while True:
        prev = label.copy()
        centroid, label = Kmeans(k, X, label)
        if np.array_equal(label, prev):
            break

    loss = Loss(centroid, X, label)
    return centroid, label, loss


# load data and parameter initialization
label = np.loadtxt("test.txt").astype(int)
data = np.loadtxt("expression.txt")
data = data.T
k = 5
label_r = np.random.randint(low=1, high=6, size=data.shape[0])

# clustering (problem 2b)
centroid, res_label, loss = Clustering(k, data, label)
centroid_r, res_label_r, loss_r = Clustering(k, data, label_r)

# plot correlation matrix (raw data)
plt.figure(dpi=300)
plt.rc("font", size=6.5)
plt.imshow(np.corrcoef(data), cmap="viridis")
plt.xlabel("gene expression")
plt.ylabel("gene expression")
plt.title("Gene Expression Correlation Heatmap (no grouping)")
plt.savefig("figure/raw_corr.png")

# plot correlation matrix (clustered data)
plt.figure(dpi=300)
plt.rc("font", size=6.5)
plt.imshow(np.corrcoef(data[np.argsort(label_r)]), cmap="viridis")
plt.xlabel("gene expression")
plt.ylabel("gene expression")
plt.title("Gene Expression Correlation Heatmap (grouping, K = 5)")
plt.savefig("figure/5means_corr.png")

# ten times running for random initialization (problem 2c) and plotting
plt.rc("font", size=15)
fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(17, 8))
for i in range(10):
    label_r = np.random.randint(low=1, high=6, size=data.shape[0])
    centroid_r, res_label_r, loss_r = Clustering(k, data, label_r)
    ax = axes[int(i / 5), int(i % 5)]
    ax.imshow(np.corrcoef(data[np.argsort(res_label_r)]), cmap="viridis")
    ax.title.set_text("{0:.4f}".format(loss_r))
fig.suptitle("Gene Expression, K = 5")
fig.text(0.5, 0.04, "Genes", ha="center")
fig.text(0.04, 0.5, "Genes", va="center", rotation="vertical")
plt.savefig("figure/5means_corr_10ran_2.png")

# different k with ten times random initializaiton
# compute average loss for visualization to find elbow point (2d)
average_loss = []
for i in range(3, 11):
    loss_sum = 0
    plt.rc("font", size=15)
    fig, axes = plt.subplots(
        nrows=2, ncols=5, sharex=True, sharey=True, figsize=(17, 8)
    )
    for s in range(10):
        label_r = np.random.randint(low=1, high=i + 1, size=data.shape[0])
        centroid_r, res_label_r, loss_r = Clustering(i, data, label_r)
        loss_sum += loss_r
        ax = axes[int(s / 5), int(s % 5)]
        ax.imshow(np.corrcoef(data[np.argsort(res_label_r)]), cmap="viridis")
        ax.title.set_text("{0:.4f}".format(loss_r))
    fig.suptitle("Gene Expression, K = " + str(i))
    fig.text(0.5, 0.04, "Genes", ha="center")
    fig.text(0.04, 0.5, "Genes", va="center", rotation="vertical")
    plt.savefig("figure/" + str(i) + "means_corr_10ran_3.png")
    average_loss.append(loss_sum / 10.0)

# plot the loss vs. number of k figure (2d)
x = [3, 4, 5, 6, 7, 8, 9, 10]
plt.figure(dpi=300)
plt.plot(x, average_loss, "bx-", linewidth=0.5)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Objective value")
plt.title("Objective Value vs. Numebr of K")
plt.savefig("figure/best_k.png")
