import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from skimage import feature
from scipy import ndimage as nd
import queue


def gaussian_smoothing(image, sigma):
    # kernel size should and odd number bigger or equal to 2.54*sigma
    # computing gaussian blur with a sigma parameter which determines how smoothed the image will be
    ceil_sigma = np.int(np.ceil(2.54 * sigma))
    kernel_shape = ceil_sigma if ceil_sigma % 2 == 1 else ceil_sigma + 1
    grid_shape = kernel_shape // 2
    kernel = np.zeros((kernel_shape, kernel_shape)).astype(np.float32)
    for i in range(0, kernel_shape):
        for j in range(0, kernel_shape):
            i_k = i - grid_shape
            j_k = j - grid_shape
            kernel[i, j] = np.exp(-(i_k * i_k + j_k * j_k) / (2 * sigma * sigma)) / (2 * np.pi * sigma * sigma)
    smoothed_image = nd.filters.convolve(image, kernel)
    smoothed_image *= 1.0 / smoothed_image.max()  # normalizing to a range 0.0-1.0 of floats
    return smoothed_image


def calculate_gradients(image):
    # computing gradient magnitude and angle of an smoothed image and normalizing its output
    kernel_horizontal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_vertical = np.flip(kernel_horizontal.T, axis=0)
    new_image_horizontal = nd.filters.convolve(image, kernel_horizontal)
    new_image_vertical = nd.filters.convolve(image, kernel_vertical)
    gradient_magnitude = np.sqrt(np.square(new_image_horizontal) + np.square(new_image_vertical))
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 1.0
    new_image_vertical[new_image_vertical == 0] = 0.000001
    gradient_angle = np.arctan2(new_image_horizontal, new_image_vertical)
    gradient_angle = np.rad2deg(gradient_angle)  # changing radians to degrees
    return gradient_magnitude, gradient_angle


def non_maximum_suppression(gradient_magnitude, gradient_angle):
    # computing non maximum suppression on image's magnitude.
    # based on angle, getting 2 neighbours of an pixel and if magnitude value of the checked pixel is higher than
    # values of neighbours then this pixel is a potential edge. Otherwise value set to 0.
    potential_edges = np.zeros(gradient_magnitude.shape)
    gradient_angle[gradient_angle < 0] += 180
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            a = 1.0
            b = 1.0
            if (0 <= gradient_angle[i, j] < 22.5) or (157.5 <= gradient_angle[i, j] <= 180):
                a = gradient_magnitude[i, j - 1]
                b = gradient_magnitude[i, j + 1]
            elif 22.5 <= gradient_angle[i, j] < 67.5:
                a = gradient_magnitude[i + 1, j + 1]
                b = gradient_magnitude[i - 1, j - 1]
            elif 67.5 <= gradient_angle[i, j] < 112.5:
                a = gradient_magnitude[i - 1, j]
                b = gradient_magnitude[i + 1, j]
            elif 112.5 <= gradient_angle[i, j] < 157.5:
                a = gradient_magnitude[i + 1, j - 1]
                b = gradient_magnitude[i - 1, j + 1]

            if (gradient_magnitude[i, j] >= a) and (gradient_magnitude[i, j] >= b):
                potential_edges[i, j] = gradient_magnitude[i, j]
            else:
                potential_edges[i, j] = 0
    potential_edges = potential_edges / potential_edges.max() * 1.0
    return potential_edges


def otsu_threshold(image):
    # computing otsu thresholding based on a histogram of an image.
    image_norm = image.copy()
    image_norm *= 255 / image_norm.max()
    image_norm = image_norm.astype(np.uint8)
    histogram = np.zeros(256)
    flattened_image = image_norm.flatten()

    threshold = max_var = sum = sumB = q1 = 0
    size = image_norm.shape[0] * image_norm.shape[1]

    for i in range(size):
        value = flattened_image[i]
        histogram[value] += 1

    for i in range(histogram.shape[0]):
        sum += i * histogram[i]

    for t in range(histogram.shape[0]):
        q1 += histogram[t]
        if q1 == 0:
            continue
        q2 = size - q1
        sumB += t * histogram[t]
        u1 = sumB / q1
        if q2 == 0:
            q2 = 1
        u2 = (sum - sumB) / q2

        sigma_square = q1 * q2 * (u1 - u2) ** 2
        if sigma_square > max_var:
            threshold = t
            max_var = sigma_square
    return threshold


def double_threshold(potential_edges, gradient_magnitude):
    # if the potential edge value is higher than threshold value received from otsu, then it is an actual edge
    # but if the value is between upper and lower thresholds, then it is an potential edge to be checked by hysteresis
    # this thresholding allows to get rid of noise ane pixels of image that are for sure not a part of an edge.
    upper_threshold = otsu_threshold(potential_edges)
    lower_threshold = upper_threshold // 3
    potential_edges *= 255 / potential_edges.max()
    strong_edges = np.zeros(gradient_magnitude.shape)
    weak_edges = np.zeros(gradient_magnitude.shape)
    for i in range(potential_edges.shape[0]):
        for j in range(potential_edges.shape[1]):
            if potential_edges[i, j] > upper_threshold:
                strong_edges[i, j] = 255
            elif upper_threshold >= potential_edges[i, j] >= lower_threshold:
                weak_edges[i, j] = 50
    return strong_edges, weak_edges


def edge_hysteresis(strong_edges, weak_edges):
    edges = strong_edges + weak_edges
    pixel_queue = queue.Queue()
    def get_neighbours(coordinate):
        neighbours = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if (i, j) == (0, 0):
                    continue
                # getting all 8 adjacent pixels of an checked one
                neighbours.append((coordinate[0] + i, coordinate[1] + j))
        return neighbours
    for i in range(strong_edges.shape[0]):
        for j in range(strong_edges.shape[1]):
            if strong_edges[i, j] != 0:
                pixel_queue.put((i, j))  # forming an queue of all pixels that are actual edges
    while not pixel_queue.empty():
        check_pixel = pixel_queue.get()
        pixel_neighbours = get_neighbours(check_pixel)
        for neighbour in pixel_neighbours:
            if edges[neighbour] == 50.0:
                # if a neighbour of an strong/actual edge is a weak edge, then it means it's actually a real one so it's
                # value is set to 255, which is the same as strong edge value.
                edges[neighbour] = 255.0
                pixel_queue.put(neighbour)  # queuing a weak-turned-to-strong neighbour in order to check his neighbours
    edges[edges == 50.0] = 0  # getting rid of all remaining weak edges, because they are not an actual ones.
    return edges


def canny(image, sigma=1.0):
    # function connecting all subfunctions into a one.
    gauss = gaussian_smoothing(image, sigma)
    magnitude, angle = calculate_gradients(gauss)
    potential_edges = non_maximum_suppression(magnitude, angle)
    strong_edges, weak_edges = double_threshold(potential_edges, magnitude)
    edges = edge_hysteresis(strong_edges, weak_edges)
    return edges


if __name__ == "__main__":
    image = color.rgb2gray(io.imread("CT_Lungs.png"))
    skimage_canny = feature.canny(image, sigma=3)
    own_canny = canny(image, sigma=3)
    plt.figure(dpi=300)
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Image")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(skimage_canny, cmap='gray')
    plt.title("Canny skimage")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(own_canny, cmap='gray')
    plt.title("Canny own")
    plt.axis('off')
    plt.show()

"""
The result of own canny function is a bit different from an skimage.feature.canny.
It is not due to bad coding/algorithm but Otsu Thresholding.
Even though my Otsu gives same result as built-in skimage.filters.threshold_otsu, which is 72, it's not the same as in
skimage. To have an edges close to skimage, the threshold value should be around 63-66. 
But all in all the result is pretty satisfying.
"""