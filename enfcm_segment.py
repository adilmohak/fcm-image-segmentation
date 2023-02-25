# import numpy as np
# import matplotlib.pyplot as plt
# import SimpleITK as sitk
# from skimage.color import label2rgb
# from skimage.filters import gaussian
# from sklearn.cluster import KMeans


# class EnFCMSegmenter:
#     def __init__(self, image, num_clusters=4, m=2, epsilon=0.01, sigma=1):
#         """
#         Initializes the EnFCMSegmenter with an input image and hyperparameters.

#         Args:
#             image (numpy array): Input 2D image array.
#             num_clusters (int): Number of clusters for KMeans clustering (default=4).
#             m (float): Fuzziness parameter for EnFCM (default=2).
#             epsilon (float): Regularization parameter for EnFCM (default=0.01).
#             sigma (float): Standard deviation for Gaussian smoothing (default=1).
#         """
#         self.image = image
#         self.num_clusters = num_clusters
#         self.m = m
#         self.epsilon = epsilon
#         self.sigma = sigma
    
#     def preprocess(self):
#         """
#         Preprocesses the input image by smoothing it with a Gaussian filter and
#         flattening it to a 1D array for clustering.

#         Returns:
#             numpy array: Flattened image array.
#         """
#         # Smooth the image using a Gaussian filter
#         smoothed = gaussian(self.image, sigma=self.sigma)
        
#         # Convert the image to a 1D array for clustering
#         flattened = np.reshape(smoothed, (-1, 1))
        
#         return flattened
    
#     def segment(self):
#         """
#         Segments the input image using the EnFCM algorithm with KMeans clustering.

#         Returns:
#             tuple: A tuple of (labeled image, segmentation mask).
#                 The labeled image is a color image where each segment is colored
#                 with a different color. The segmentation mask is a binary image
#                 where each pixel is assigned a label (0 or 1) based on whether it
#                 belongs to the foreground or background.
#         """
#         # Preprocess the image
#         flattened = self.preprocess()
        
#         # Apply the EnFCM algorithm
#         kmeans = KMeans(n_clusters=self.num_clusters)
#         kmeans.fit(flattened)
#         centers = kmeans.cluster_centers_.squeeze()
#         labels = kmeans.labels_
#         w = 1.0 / (1.0 + (np.square(flattened - centers[labels]) / self.epsilon))
#         u = np.power(w, 1.0 / (self.m - 1))
#         u = u / np.sum(u, axis=1)[:, np.newaxis]
#         segmentation = np.reshape(u[:, 1], self.image.shape)
        
#         # Create a labeled image for visualization
#         labeled = label2rgb(segmentation, self.image, colors=[(0, 0, 1), (1, 0, 0)], alpha=0.5)
        
#         return labeled, segmentation

# if __name__ == "__main__":
#     # Load the input image
#     input_image = sitk.ReadImage('./data/brain_tumor_dataset/no/1 no.jpeg')
#     input_array = sitk.GetArrayFromImage(input_image)

#     # Select a slice to segment
#     slice_idx = 50
#     slice_image = input_array[slice_idx, :]

#     # Initialize the EnFCM segmenter with the slice image
#     enfcm_segmenter = EnFCMSegmenter(slice_image)

#     # Segment the image using EnFCM
#     labeled, segmentation = enfcm_segmenter.segment()

#     # Visualize the results
#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

#     axes[0].imshow(slice_image, cmap='gray')
#     axes[0].set_title('Input image')

#     axes[1].imshow(labeled)
#     axes[1].set_title('Labeled image')

#     axes[2].imshow(segmentation, cmap='gray')
#     axes[2].set_title('Segmentation mask')

#     plt.show()


# # # Load an example image
# # image = sitk.ReadImage('example_image.nii.gz')
# # image_array = sitk.GetArrayFromImage(image)

# # # Choose hyperparameters
# # num_clusters = 3
# # m = 2
# # epsilon = 0.01
# # sigma = 1

# # # Create an instance of the EnFCMSegmenter class
# # segmenter = EnFCMSegmenter(image_array, num_clusters=num_clusters, m=m, epsilon=epsilon, sigma=sigma)

# # # Segment the image
# # labeled_image, segmentation_mask = segmenter.segment()

# # # Plot the results
# # fig, ax = plt.subplots(1, 2)
# # ax[0].imshow(labeled_image











# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.color import label2rgb
# import skfuzzy as fuzz


# class ENFCM_Segmenter:
#     def __init__(self, image, num_clusters=4, m=2, w=0.5, beta=100, gamma=1, max_iter=100, threshold=1e-5):
#         """
#         Initializes the ENFCM_Segmenter with the given parameters.

#         Args:
#         - image: input image to be segmented.
#         - num_clusters: number of clusters to be used in the Enhanced Fuzzy C-Means algorithm.
#         - m: fuzzy exponent parameter used in the Enhanced Fuzzy C-Means algorithm.
#         - w: weight factor used in the Enhanced Fuzzy C-Means algorithm.
#         - beta: shape parameter used in the Enhanced Fuzzy C-Means algorithm.
#         - gamma: scale parameter used in the Enhanced Fuzzy C-Means algorithm.
#         - max_iter: maximum number of iterations for the Enhanced Fuzzy C-Means algorithm.
#         - threshold: threshold for stopping the Enhanced Fuzzy C-Means algorithm.
#         """
#         self.image = image
#         self.num_clusters = num_clusters
#         self.m = m
#         self.w = w
#         self.beta = beta
#         self.gamma = gamma
#         self.max_iter = max_iter
#         self.threshold = threshold
    
#     def preprocess(self):
#         """
#         Preprocesses the input image by normalizing its pixel intensities and reshaping it into a 2D array.

#         Returns:
#         - flattened: 2D array containing the normalized pixel intensities of the input image.
#         """
#         # Normalize pixel intensities to range [0, 1]
#         normalized = (self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image))
#         # Reshape image into a 2D array
#         flattened = np.reshape(normalized, (-1, 1))
#         return flattened
    
#     def segment(self):
#         """
#         Segments the input image using the Enhanced Fuzzy C-Means algorithm.

#         Returns:
#         - image: original input image.
#         - labeled: labeled image for visualization.
#         - segmentation: segmented image.
#         """
#         # Preprocess the image
#         flattened = self.preprocess()
        
#         # Apply the ENFCM algorithm
#         cntr, u, u0, d, jm, p, fpc = fuzz.cluster.enfcmeans(
#             flattened.T, self.num_clusters, self.m, self.w, self.beta, self.gamma, error=self.threshold, maxiter=self.max_iter, init=None)
#         segmentation = np.reshape(u[1], self.image.shape)
        
#         # Create a labeled image for visualization
#         labeled = label2rgb(segmentation, self.image, colors=[(0, 0, 1), (1, 0, 0)], alpha=0.5)
        
#         return self.image, labeled, segmentation

# if __name__ == "__main__":
#     # Load the input image
#     image = cv2.imread('./data/brain_tumor_dataset/yes/Y1.jpg', 0)

#     # Create an instance of the ENFCM_Segmenter
#     segmenter = ENFCM_Segmenter(image, num_clusters=4, m=2, w=0.5, beta=100, gamma=1, max_iter=100, threshold=1e-5)

#     # Segment the input image
#     original, labeled, segmentation = segmenter.segment()

#     # Display the original and segmented image using matplotlib
#     fig, axs = plt.subplots(1, 2, figsize=(10, 10))
#     axs[0].imshow(original, cmap='gray')
#     axs[0].set_title('Original Image')
#     axs[1].imshow(segmentation)
#     axs[1].set_title('Segmented Image')
#     plt.show()














import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import skfuzzy as fuzz


class ENFCM_Segmenter:
    def __init__(self, image, num_clusters=4, m=2, w=0.1, max_iter=100, threshold=1e-5):
        """
        Initializes the ENFCM_Segmenter with the given parameters.

        Args:
        - image: input image to be segmented.
        - num_clusters: number of clusters to be used in the ENFCM algorithm.
        - m: fuzzy exponent parameter used in the ENFCM algorithm.
        - w: spatial weighting factor used in the ENFCM algorithm.
        - max_iter: maximum number of iterations for the ENFCM algorithm.
        - threshold: threshold for stopping the ENFCM algorithm.
        """
        self.image = image
        self.num_clusters = num_clusters
        self.m = m
        self.w = w
        self.max_iter = max_iter
        self.threshold = threshold
    
    def preprocess(self):
        """
        Preprocesses the input image by normalizing its pixel intensities and reshaping it into a 2D array.

        Returns:
        - flattened: 2D array containing the normalized pixel intensities of the input image.
        """
        # Normalize pixel intensities to range [0, 1]
        normalized = (self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image))
        # Reshape image into a 2D array
        flattened = np.reshape(normalized, (-1, 1))
        return flattened
    
    def segment(self):
        """
        Segments the input image using the ENFCM algorithm.

        Returns:
        - image: original input image.
        - labeled: labeled image for visualization.
        - segmentation: segmented image.
        """
        # Preprocess the image
        flattened = self.preprocess()
        
        # Apply the ENFCM algorithm
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.enfcmeans(
            flattened.T, self.num_clusters, self.m, self.w, error=self.threshold, maxiter=self.max_iter, init=None)
        segmentation = np.reshape(u[1], self.image.shape)
        
        # Create a labeled image for visualization
        labeled = label2rgb(segmentation, self.image, colors=[(0, 0, 1), (1, 0, 0)], alpha=0.5)
        
        return self.image, labeled, segmentation

if __name__ == "__main__":
    # Load the input image
    image = cv2.imread('./data/brain_tumor_dataset/yes/Y1.jpg', 0)

    # Create an instance of the ENFCM_Segmenter
    segmenter = ENFCM_Segmenter(image, num_clusters=4, m=2, w=0.1, max_iter=100, threshold=1e-5)

    # Segment the input image
    original, labeled, segmentation = segmenter.segment()

    # Display the original and segmented image using matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(segmentation)
    axs[1].set_title('Segmented Image')
    plt.show()
