import numpy as np
import imageio
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import math
import statistics
import time

# Image path and name processing
image_name = input("Enter noisy image filename: ")
image_path = "images/noisy/" + image_name
short_image_name = image_path.split("/")[-1].split(".")[0]

# read in the image as an (n,m,3) rgb pixel matrix
noisy_data = imageio.imread(image_path, as_gray=False, pilmode="RGB")
# noisy_data = noisy_data.resize((300, 200,3), Image.ANTIALIAS)

# Converts RGB color matrix to a Hue-Saturation-Lightness (HSV) matrix 
# Prevents the median filter on a color-image from creating false colors
hsv_matrix = matplotlib.colors.rgb_to_hsv(noisy_data)

def mean_filter(arr):
    """Mean filter implementation using the convolution method"""
    z_matrix = np.apply_along_axis(np.convolve, 0, arr, np.array([1/3, 1/3, 1/3]))
    return np.apply_along_axis(np.convolve, 1, z_matrix, np.array([1/3, 1/3, 1/3]))


def median_filter(arr):
    """ Non-linear Median filter for a (n,m,1) graphics matrix"""
    def median(i,j):
        """
        Single-point median filter implementation using loops
        This function computes the median in a 3x3 grid around the input grid coordinates (i,j)
        """
        lower = -1 * math.floor(r/2)
        upper = math.floor(r/2) + 1
        elements = [arr[i+x][j+y] for x in range(lower, upper) for y in range(lower, upper)]
        return statistics.median(elements) if elements else 0.5
    r = 3 # the number of values to sample for the median)
    x_len, y_len = np.shape(arr)
    y = np.zeros([x_len, y_len])
    # computes the median for each grid point and writes to a new matrix y
    for i in range(r, x_len-r):
        for j in range(r, y_len - r):
            y[i][j] = median(i,j)
    return y
    



# apply the mean filter to each of the HSV channels
mean_start = time.time()
mean_hsv_matrix = np.stack([mean_filter(hsv_matrix[:,:,i]) for i in range(hsv_matrix.shape[2])], axis=2)
mean_rgb_matrix = matplotlib.colors.hsv_to_rgb(mean_hsv_matrix)
plt.imsave("images/output/" + short_image_name + "_mean_filtered.jpg", np.divide(mean_rgb_matrix, 255.0))
print("mean filter (using convolution) runtime is " + str(time.time() - mean_start))



# apply the median filter to each of the HSV channels
median_start = time.time()
median_hsv_matrix = np.stack([median_filter(hsv_matrix[:,:,i]) for i in range(hsv_matrix.shape[2])], axis=2)

median_rgb_matrix = matplotlib.colors.hsv_to_rgb(median_hsv_matrix)
plt.imsave("images/output/" + short_image_name + "_median_filtered.jpg", np.divide(median_rgb_matrix, 255.0))
print("median filter (using loops) runtime is " + str(time.time() - median_start))
