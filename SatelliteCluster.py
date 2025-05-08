import numpy as np
from PIL import Image
from tabulate import tabulate

def image_open(filename):
    image = Image.open(filename)
    image = image.convert('L')  
    image_array = np.array(image)
    binary = np.where(image_array > 127, 1, 0)   
    coordinates = np.column_stack(np.where(binary == 1))
    return coordinates

def initial_centroids(X, k):
    
    np.random.seed(0)
    initial_indices = np.random.choice(len(X), k, replace=False)
    return X[initial_indices]
def clusters(X, centroids):
    labels = []
    for x in X:
        distances = [np.linalg.norm(x - centroid) for centroid in centroids]
        min_index = np.argmin(distances)
        labels.append(min_index)
    return np.array(labels)

def update_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        # Initialize variables to store sum and count for each dimension
        sum_dimension = np.zeros(X.shape[1])  # X.shape[1] gives the number of dimensions/features
        count = 0
        # Iterate through each data point and accumulate sum and count for points belonging to cluster i
        for j in range(len(X)):
            if labels[j] == i:
                sum_dimension += X[j]
                count += 1
        # Calculate the mean for each dimension
        if count != 0:
            mean = sum_dimension / count
        else:
            mean = np.zeros(X.shape[1])  # If no points belong to the cluster, set centroid to zero vector
        new_centroids.append(mean)
    return np.array(new_centroids)
def silhouette(X, labels, centroids):
    sil_scores = []
    for i in range(len(X)):
        # Calculate average distance from point i to other points in the same cluster
        distances_within = []
        for j in range(len(X)):
            if labels[j] == labels[i] and i != j:
                distances_within.append(np.linalg.norm(X[j] - X[i]))
        mean_distance_within = np.mean(distances_within) if distances_within else 0
        
        # Calculate average distance from point i to points in other clusters
        distances_between = []
        for j in range(len(X)):
            if labels[j] != labels[i]:
                distances_between.append(np.mean(np.linalg.norm(X[j] - X[i])))
        mean_distance_between = min(distances_between) if distances_between else 0
        
        # Calculate silhouette score for point i
        sil_score = (mean_distance_between - mean_distance_within) / max(mean_distance_within, mean_distance_between) if mean_distance_between != 0 else 0
        sil_scores.append(sil_score)
    
    return np.mean(sil_scores)
def kmeans(X, k, max_iters=100):
    centroids = initial_centroids(X, k)
    for _ in range(max_iters):
        labels = clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

def find_clusters(X, k_range):
    best_score = -1
    best_k = None
    for k in k_range:
        centroids, labels = kmeans(X, k)
        score = silhouette(X, labels, centroids)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score
def calculate_centroid_distances(centroids):
    num_centroids = len(centroids)
    distances = np.zeros((num_centroids, num_centroids))
    for i in range(num_centroids):
        for j in range(i + 1, num_centroids):
            distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])
            distances[j, i] = distances[i, j]
    return distances
# Load and process images
filenames = ['1.png', '2.png', '3.png', '4.png']
results = []
for filename in filenames:
    coordinates = image_open(filename)
    best_k, best_score = find_clusters(coordinates, range(2, 11))
    centroids, labels = kmeans(coordinates, best_k)
    results.append((filename, best_k, centroids, labels, best_score))

# Print the results
# Continuing from the previous code where we loop through filenames and perform clustering:


# Print the results in a table format
# Print the results in a table format
for result in results:
    filename, best_k, centroids, labels, best_score = result
    distances = calculate_centroid_distances(centroids)
    print(f"Image: {filename}, Best k: {best_k}")
    print("Centroid distances:")

    # Print distances between centroids
    num_centroids = len(centroids)
    for i in range(num_centroids):
        for j in range(i + 1, num_centroids):
            print(f"Distance of centroid {i+1} from {j+1}: {distances[i, j]}")
    print()



