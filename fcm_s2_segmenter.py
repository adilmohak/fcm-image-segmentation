import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import skfuzzy as fuzz


class FCM_S2_Segmenter:
    def __init__(self, image, num_clusters=4, m=2, max_iter=100, threshold=1e-5):
        """
        Initializes the FCM_S2_Segmenter with the given parameters.

        Args:
        - image: input image to be segmented.
        - num_clusters: number of clusters to be used in the Fuzzy C-Means algorithm.
        - m: fuzzy exponent parameter used in the Fuzzy C-Means algorithm.
        - max_iter: maximum number of iterations for the Fuzzy C-Means algorithm.
        - threshold: threshold for stopping the Fuzzy C-Means algorithm.
        """
        self.image = image
        self.num_clusters = num_clusters
        self.m = m
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
        Segments the input image using the Fuzzy C-Means algorithm.

        Returns:
        - image: original input image.
        - labeled: labeled image for visualization.
        - segmentation: segmented image.
        """
        # Preprocess the image
        flattened = self.preprocess()
        
        # Apply the FCM_S2 algorithm
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            flattened.T, self.num_clusters, self.m, error=self.threshold, maxiter=self.max_iter)
        segmentation = np.reshape(np.argmax(u, axis=0), self.image.shape)
        
        # Create a labeled image for visualization
        labeled = label2rgb(segmentation, self.image, colors=[(0, 0, 1), (1, 0, 0)], alpha=0.5)
        
        return self.image, labeled, segmentation


if __name__ == "__main__":
    # Load the input image
    image = cv2.imread('./data/brain_tumor_dataset/yes/Y1.jpg', 0)

    # Create an instance of the FCM_S2_Segmenter
    segmenter = FCM_S2_Segmenter(image, num_clusters=4, m=2, max_iter=100, threshold=1e-5)

    # Segment the input image
    original, labeled, segmentation = segmenter.segment()

    # Display the original and segmented image using matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(segmentation)
    axs[1].set_title('Segmented Image Using FCM_S2 Algorithm')
    plt.show()
