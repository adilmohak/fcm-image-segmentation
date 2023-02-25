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
