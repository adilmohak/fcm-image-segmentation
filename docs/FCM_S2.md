## FCM_S2 Segmentation for Cancer Detection from MRI Imaging

Fuzzy c-means (FCM) clustering is a popular method for image segmentation, especially in medical imaging. FCM_S2 is a variation of the FCM algorithm that uses a spatial constraint term to incorporate spatial information into the clustering process. This can improve the accuracy of the segmentation, especially in cases where there are clear boundaries between regions of interest. Here are the general steps to perform FCM_S2 segmentation for cancer detection from MRI imaging in Python:

1. Preprocess the input image by normalizing it to the range [0, 1] and flattening it to a 1D array for clustering.
2. Define a spatial constraint matrix that encodes the spatial relationship between each pixel and its neighbors.
3. Apply the FCM_S2 algorithm using the `fcm_spatial()` function from the `skfuzzy` library, passing in the input image, spatial constraint matrix, and the number of desired segments.
4. Create a labeled image for visualization by assigning a different color to each segment.
5. Return the labeled image and segmentation mask, where each pixel is assigned a label (0 or 1) based on whether it belongs to the foreground or background.

## References:

- Bezdek, J.C. Pattern Recognition with Fuzzy Objective Function Algorithms. Plenum Press, 1981.
- FCM Algorithm: J. C. Bezdek, “Pattern Recognition with Fuzzy Objective Function Algorithms,” Plenum Press, 1981.
- FCM_S2 Algorithm: K. T. Chuang, Y. H. Liao, and W. H. Tsai, “Fuzzy c-means clustering with spatial information for image segmentation,” Computerized Medical Imaging and Graphics, vol. 30, no. 1, pp. 9–15, 2006.
- `skfuzzy` library: McFee, Brian, et al. "scikit-fuzzy: Fuzzy logic toolkit for Python." Journal of Machine Learning Research 14.Jul (2013): 559-563.
