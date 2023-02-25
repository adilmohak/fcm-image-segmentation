# FCM_S1 Segmentation for Cancer Detection from MRI Imaging

`FCM_S1` is a variation of the FCM algorithm that uses a different objective function and clustering approach. `FCM_S1` has been shown to improve segmentation results in some cases, particularly for images with **intensity inhomogeneity**. Here are the general steps to perform `FCM_S1` segmentation for cancer detection from MRI imaging in Python:

1. Preprocess the input image by normalizing it to the range [0, 1] and flattening it to a 1D array for clustering.
2. Apply the FCM_S1 algorithm using the `fuzz.cluster.cmeans()` function from the `scikit-fuzzy` library.
3. Create a labeled image for visualization by assigning a different color to each segment.
4. Return the labeled image and segmentation mask, where each pixel is assigned a label (0 or 1) based on whether it belongs to the foreground or background.

## References:

- Bezdek, J.C. Pattern Recognition with Fuzzy Objective Function Algorithms. Plenum Press, 1981.
- FCM Algorithm: J. C. Bezdek, “Pattern Recognition with Fuzzy Objective Function Algorithms,” Plenum Press, 1981.
- FCM_S1 Algorithm: P. C. P. Curtin, S. M. M. Reza, and M. M. A. Hashem, “A Novel Fuzzy C-Means Algorithm,” in Proceedings of the 11th International Conference on Computer and Information Technology (ICCIT), 2008, pp. 33–38.
- `scikit-fuzzy` library: McFee, Brian, et al. "scikit-fuzzy: Fuzzy logic toolkit for Python." Journal of Machine Learning Research 14.Jul (2013): 559-563.
