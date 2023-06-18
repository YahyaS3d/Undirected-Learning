# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture
#
# # Set random seed for reproducibility
# np.random.seed(42)
#
# # Parameters for Group 1
# mean_group1 = [-1, -1]
# cov_group1 = [[0.8, 0], [0, 0.8]]
#
# # Parameters for Group 2
# mean_group2 = [1, 1]
# cov_group2 = [[0.75, -0.2], [-0.2, 0.6]]
#
# # Generate data for Group 1
# data_group1 = np.random.multivariate_normal(mean_group1, cov_group1, 700)
#
# # Generate data for Group 2
# data_group2 = np.random.multivariate_normal(mean_group2, cov_group2, 300)
#
# # Plot the data
# plt.scatter(data_group1[:, 0], data_group1[:, 1], label='Group 1', alpha=0.7)
# plt.scatter(data_group2[:, 0], data_group2[:, 1], label='Group 2', alpha=0.7)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Generated Data')
# plt.legend()
# plt.show()
#
# # Combine the data from both groups
# data = np.concatenate((data_group1, data_group2))
#
# # Specify the number of clusters (k = 2 in this case)
# k = 2
#
# # Initialize the KMeans model
# kmeans = KMeans(n_clusters=2, n_init='auto')
#
# # Fit the model to the data
# kmeans.fit(data)
#
# # Get the predicted labels for each data point
# predicted_labels = kmeans.labels_
#
# # Separate the data into predicted Group 1 and Group 2
# predicted_group1 = data[predicted_labels == 0]
# predicted_group2 = data[predicted_labels == 1]
#
# # Plot the predicted clusters
# plt.scatter(predicted_group1[:, 0], predicted_group1[:, 1], label='Predicted Group 1', alpha=0.7)
# plt.scatter(predicted_group2[:, 0], predicted_group2[:, 1], label='Predicted Group 2', alpha=0.7)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Predicted Clusters (k-means)')
# plt.legend()
# plt.show()
#
#
# # Initialize the GaussianMixture model
# gmm = GaussianMixture(n_components=k, random_state=42)
#
# # Fit the model to the data
# gmm.fit(data)
#
# # Get the predicted labels for each data point
# predicted_labels = gmm.predict(data)
#
# # Separate the data into predicted Group 1 and Group 2
# predicted_group1 = data[predicted_labels == 0]
# predicted_group2 = data[predicted_labels == 1]
#
# # Plot the predicted clusters
# plt.scatter(predicted_group1[:, 0], predicted_group1[:, 1], label='Predicted Group 1', alpha=0.7)
# plt.scatter(predicted_group2[:, 0], predicted_group2[:, 1], label='Predicted Group 2', alpha=0.7)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Predicted Clusters (GMM)')
# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
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

# Initialize the KMeans model
kmeans = KMeans(n_clusters=2, n_init='auto')

# Fit the model to the data
kmeans.fit(data)

# Get the predicted labels for each data point
predicted_labels_kmeans = kmeans.labels_

# Initialize the GaussianMixture model
gmm = GaussianMixture(n_components=k, random_state=42)

# Fit the model to the data
gmm.fit(data)

# Get the predicted labels for each data point
predicted_labels_gmm = gmm.predict(data)

# Create a colormap for true labels
cmap_true = plt.colormaps.get_cmap('Set1')

# Create a colormap for predicted labels
cmap_predicted = plt.colormaps.get_cmap('Set2')

# Plot the data with true labels and predicted labels
plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap=cmap_true, marker='o', label='True Labels', alpha=0.7)
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels_kmeans, cmap=cmap_predicted, marker='s', label='KMeans Predicted Labels', alpha=0.7)
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels_gmm, cmap=cmap_predicted, marker='^', label='GMM Predicted Labels', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data with True and Predicted Labels')
plt.legend()

# Mark the differences between true labels and predicted labels
misclassified_kmeans = data[true_labels != predicted_labels_kmeans]
misclassified_gmm = data[true_labels != predicted_labels_gmm]

plt.scatter(misclassified_kmeans[:, 0], misclassified_kmeans[:, 1], color='red', marker='x', label='Misclassified (KMeans)')
plt.scatter(misclassified_gmm[:, 0], misclassified_gmm[:, 1], color='blue', marker='*', label='Misclassified (GMM)')

plt.legend()
plt.show()


