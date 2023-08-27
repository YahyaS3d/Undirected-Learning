import numpy as np
import matplotlib.pyplot as plt

def count_pro(X, U, S):
    n = X.shape[0]
    value = (1 / (2 * np.pi) ** (n / 2) * np.linalg.det(S)) * np.exp(-0.5 * np.dot((X - U), np.linalg.inv(S)).dot((X - U).T))
    return value

def Expectation():
    global save_pro
    for i in range(1000):
        total = 0
        for j in range(k):
            pro = count_pro(point[i], U[j], S[j]) * W[j]
            save_pro[i, j] = pro
            total += pro

        if total != 0:  # Check if total is nonzero
            for g in range(k):
                responsibility_Matrix[i, g] = save_pro[i, g] / total
        else:
            responsibility_Matrix[i, :] = 0  # Set the entire row to zero if total is zero

def Maximization():
    global W, U, S
    for j in range(k):
        sum1 = 0
        sum2 = 0
        am = 0
        for i in range(1000):
            sum1 += responsibility_Matrix[i, j] * point[i]
            sum2 += responsibility_Matrix[i, j]
            am += responsibility_Matrix[i, j] * np.outer((point[i] - U[j]), (point[i] - U[j]))

        if not np.isnan(sum2) and sum2 != 0:
            W[j] = sum2 / 1000
            U[j] = sum1 / sum2
            S[j] = am / sum2
        else:
            W[j] = 0
            U[j] = U_initial[j]
            S[j] = S_initial[j]

k = 2
u1 = np.array([-1, -1])
u2 = np.array([1, 1])
U = np.array([u1, u2])
U_initial = np.copy(U)

S1 = np.array([[0.8, 0], [0, 0.8]])
S2 = np.array([[0.75, -0.2], [-0.2, 0.6]])
S = np.array([S1, S2])
S_initial = np.copy(S)

W = np.array([700, 300])
responsibility_Matrix = np.zeros((1000, k))
save_pro = np.zeros((1000, k))

data_group1 = np.random.multivariate_normal(u1, S1, 700)
data_group2 = np.random.multivariate_normal(u2, S2, 300)
point = np.concatenate((data_group1, data_group2))

# Expectation-Maximization iterations
log_likelihoods = []  # Track log-likelihoods for convergence
prev_log_likelihood = None
diff = 1e6  # Initialize the log-likelihood difference to a large value

for iteration in range(10):
    Expectation()
    Maximization()

    # Expectation step: Calculate log-likelihood
    log_likelihood = 0.0
    for i in range(1000):
        total = 0
        for j in range(k):
            pro = count_pro(point[i], U[j], S[j]) * W[j]
            save_pro[i, j] = pro
            total += pro
        if total != 0:  # Check if total is non-zero
            log_likelihood += np.log(total)

    # Print the log-likelihood after each iteration
    print(f"Iteration {iteration+1}, Log-Likelihood: {log_likelihood}")
    log_likelihoods.append(log_likelihood)

    # Check convergence based on log-likelihood difference
    if prev_log_likelihood is not None:
        diff = np.abs(log_likelihood - prev_log_likelihood)
        if diff < 1e-6:
            print("Converged.")
            break

    prev_log_likelihood = log_likelihood

# Plot log-likelihood convergence
plt.plot(log_likelihoods)
plt.title("Log-Likelihood Convergence")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.show()

# Plotting the results
plt.scatter(data_group1[:, 0], data_group1[:, 1], c='blue', label='Group 1')
plt.scatter(data_group2[:, 0], data_group2[:, 1], c='green', label='Group 2')
plt.scatter(U[:, 0], U[:, 1], c='red', marker='x', label='Cluster Centers')
plt.title("Gaussian Mixture Model")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
