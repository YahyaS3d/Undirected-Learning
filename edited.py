import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for Group 1
mean_group1 = [-1, -1]
cov_group1 = [[0.8, 0], [0, 0.8]]

# Parameters for Group 2
mean_group2 = [1, 1]
cov_group2 = [[0.75, -0.2], [-0.2, 0.6]]

# Generate data for Group 1
data_group1 = np.random.multivariate_normal(mean_group1, cov_group1, 700)

# Generate data for Group 2
data_group2 = np.random.multivariate_normal(mean_group2, cov_group2, 300)

# Combine the data from both groups
data = np.concatenate((data_group1, data_group2))

# True labels for the data
true_labels = np.concatenate((np.zeros(700), np.ones(300)))

# Specify the number of clusters (k = 2 in this case)
k = 2

# Expectation-Maximization Algorithm
max_iters = 100
# Threshold
tolerance = 1e-4

# Initialize the means randomly
np.random.seed(42)
means = np.random.randn(k, 2)

# Initialize the covariances randomly
covariances = np.array([np.eye(2)] * k)

# Initialize the mixing coefficients uniformly
mixing_coeffs = np.ones(k) / k

# Expectation-Maximization Algorithm
for _ in range(max_iters):
    # Expectation Step
    responsibilities = np.zeros((len(data), k))
    for j in range(k):
        responsibilities[:, j] = mixing_coeffs[j] * (
            1 / np.sqrt(2 * np.pi * np.linalg.det(covariances[j]))) * np.exp(
            -0.5 * np.sum(np.dot((data - means[j]), np.linalg.inv(covariances[j])) * (data - means[j]), axis=1))

    # Normalize the responsibilities
    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

    # Maximization Step
    prev_means = np.copy(means)
    prev_covariances = np.copy(covariances)
    prev_mixing_coeffs = np.copy(mixing_coeffs)

    for j in range(k):
        Nj = np.sum(responsibilities[:, j])
        means[j] = np.sum(responsibilities[:, j][:, None] * data, axis=0) / Nj
        covariances[j] = (np.dot((responsibilities[:, j][:, None] * (data - means[j])).T,
                                (data - means[j])) / Nj) + np.eye(2) * 1e-6
        mixing_coeffs[j] = Nj / len(data)

    # Check convergence
    mean_diff = np.sum(np.abs(means - prev_means))
    cov_diff = np.sum(np.abs(covariances - prev_covariances))
    mixing_coeffs_diff = np.sum(np.abs(mixing_coeffs - prev_mixing_coeffs))

    if mean_diff < tolerance and cov_diff < tolerance and mixing_coeffs_diff < tolerance:
        break

# Get the predicted labels for each data point
predicted_labels_gmm = np.argmax(responsibilities, axis=1)

# Create a colormap for true labels
cmap_true = cm.get_cmap('Set1')

# Create a colormap for predicted labels
cmap_predicted = cm.get_cmap('Set2')

# Plot the data with true labels and predicted labels
plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap=cmap_true, marker='o', label='True Labels', alpha=0.7)
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels_gmm, cmap=cmap_predicted, marker='^', label='GMM Predicted Labels', alpha=0.7)
plt.xlabel('Group 1')
plt.ylabel('Group 2')
plt.title('Data with True and Predicted Labels')
plt.legend()

plt.show()
