import numpy as np
import imageio
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import math
import statistics
import time


image_name = "chicago.jpg"
image_path = "images/input/" + image_name
short_image_name = image_path.split("/")[-1].split(".")[0]

# noisy_data = loadmat("images/input/noisy_richb.mat")["x"]
# plt.imsave("original_richb.jpg", noisy_data, cmap="gray")
noisy_data = imageio.imread(image_path,
as_gray=False, pilmode="RGB")

# Converts RGB color matrix to a Hue-Saturation-Lightness (HSV) matrix 
# Prevents the median filter on a color-image from creating false colors
hsv_matrix = matplotlib.colors.rgb_to_hsv(noisy_data)


# mean filter implementation using the convolution method 
# mean_start = time.time()

# print("mean filter (using convolution) runtime is " + str(time.time() - mean_start))
# plt.imsave("mean_richb.jpg", output, cmap="gray")


def mean_filter(arr):
    z_matrix = np.apply_along_axis(np.convolve, 0, arr, np.array([1/3, 1/3, 1/3]))
    return np.apply_along_axis(np.convolve, 1, z_matrix, np.array([1/3, 1/3, 1/3]))


def median_filter(arr):
    """ Median filter for a (n,m,1) graphics matrix"""
    def median(i,j):
        """
        Single-point median filter implementation using loops
        This function computes the median in a 3x3 grid around the input grid coordinates (i,j)
        """
        elements = []
        for x in range(-1 * math.floor(r/2), math.floor(r/2)+1):
            for y in range(-1 * math.floor(r/2), math.floor(r/2)+1):
                if (i+x) > 0 and (i+x) < x_len and (y+j) > 0 and (y+j) < y_len:
                    elements.append(arr[i + x][j + y]) 
            return statistics.median(elements) if elements else 0.5
    r = 4
    x_len, y_len = np.shape(arr)
    y = np.zeros([x_len, y_len])
    # computes the median for each grid point and writes to a new matrix y
    for i in range(x_len):
        for j in range(y_len):
            y[i][j] = median(i,j)
    return y
    



# apply the mean filter to each of the HSV channels
mean_hsv_matrix = np.stack([mean_filter(hsv_matrix[:,:,i]) for i in range(hsv_matrix.shape[2])], axis=2)
mean_rgb_matrix = matplotlib.colors.hsv_to_rgb(mean_hsv_matrix)
plt.imsave("images/output/mean_filtered_" + short_image_name + ".jpg", np.divide(mean_rgb_matrix, 255.0))


# apply the median filter to each of the HSV channels
median_hsv_matrix = np.stack([median_filter(hsv_matrix[:,:,i]) for i in range(hsv_matrix.shape[2])], axis=2)
median_rgb_matrix = matplotlib.colors.hsv_to_rgb(median_hsv_matrix)
plt.imsave("images/output/median_filtered_" + short_image_name + ".jpg", np.divide(median_rgb_matrix, 255.0))

# median_start = time.time()

# print("median filter (using loops) runtime is " + str(time.time() - median_start))
# plt.imsave("median_richb.jpg", y, cmap="gray")

