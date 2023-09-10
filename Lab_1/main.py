import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np

file_path = "flowers.jpg"  # Replace with the path to your image

# Function to load an image
def load_image(file_path):
    try:
        img = mpimg.imread(file_path)
        return img
    except Exception as e:
        print(f"An error occurred while loading the image: {str(e)}")
        return None

# Function to display an image
def display_image(image, cmap="viridis"):
    try:
        plt.imshow(image, cmap=cmap)
        plt.axis('off')  # Turn off axis labels

        plt.show()

    except Exception as e:
        print(f"An error occurred while displaying the image: {str(e)}")

# Function to convert a color image to grayscale using the specified formula
def color_to_grayscale(image):
    try:
        # Ensure the image has three color channels (RGB)
        print(image.shape[2])
        if image.shape[2] == 3:
            # Apply the formula to calculate grayscale values
            grayscale_image = np.dot(image[..., :3], [0.3, 0.59, 0.11])
            #grayscale_image = grayscale_image.astype(np.uint8)  # Convert to 8-bit unsigned integer
            return grayscale_image
        else:
            print("Input image is not a color image (RGB).")
            return None
    except Exception as e:
        print(f"An error occurred during color to grayscale conversion: {str(e)}")
        return None

# Function to display the brightness histogram of an image
def display_brightness_histogram(gray_image):
    try:
        # Ensure the image is in grayscale
        if len(gray_image.shape) == 2:
            # Calculate the histogram
            histogram, bins = np.histogram(gray_image.flatten(), bins=256, range=(0, 256))

            # Plot the histogram
            plt.figure(figsize=(8, 4))
            plt.title("Brightness Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.bar(bins[:-1], histogram, width=1, color='gray')
            plt.xlim([0, 256])
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
        else:
            print("Input image must be grayscale.")
    except Exception as e:
        print(f"An error occurred while displaying the histogram: {str(e)}")

# Function to apply logarithmic correction to an image
def apply_logarithmic_correction(gray_image, c):
    try:
        # Ensure the image is in grayscale
        if len(gray_image.shape) == 2:
            # Apply the logarithmic correction
            corrected_image = c * np.log(1 + gray_image)

            # Normalize the pixel values to the 0-255 range
            corrected_image = ((corrected_image - corrected_image.min()) / (corrected_image.max() - corrected_image.min()) * 255).astype(np.uint8)

            return corrected_image
        else:
            print("Input image must be grayscale.")
            return None
    except Exception as e:
        print(f"An error occurred during logarithmic correction: {str(e)}")
        return None

# Function to apply the Roberts operator filter to an image
def apply_roberts_operator(gray_image):
    try:
        # Ensure the image is in grayscale
        if len(gray_image.shape) == 2:
            # Define the Roberts operator kernels
            kernel_x = np.array([[1, 0], [0, -1]])
            kernel_y = np.array([[0, 1], [-1, 0]])

            # Convolve the image with the Roberts operator kernels
            gradient_x = np.abs(ndimage.convolve(gray_image, kernel_x ))
            gradient_y = np.abs(ndimage.convolve( gray_image, kernel_y))

            # Calculate the gradient magnitude
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

            # Normalize the gradient magnitude to the 0-255 range
            gradient_magnitude = ((gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min()) * 255).astype(np.uint8)

            return gradient_magnitude
        else:
            print("Input image must be grayscale.")
            return None
    except Exception as e:
        print(f"An error occurred during Roberts operator filtering: {str(e)}")
        return None

# Example usage:
if __name__ == "__main__":
    # Load the image
    loaded_image = load_image(file_path)

    if loaded_image is not None:
        # Display the original image
        display_image(loaded_image)

        # Convert the color image to grayscale
        grayscale_image = color_to_grayscale(loaded_image)

        if grayscale_image is not None:
            # Display the brightness histogram of the grayscale image
            display_brightness_histogram(grayscale_image)

            # Apply logarithmic correction to the grayscale image
            corrected_image = apply_logarithmic_correction(grayscale_image, 50)

            if corrected_image is not None:
                # Display the brightness histogram of the corrected image
                display_brightness_histogram(corrected_image)
                display_image(corrected_image, cmap = "gray")

                # Apply the Roberts operator filter to the grayscale image
                roberts_image = apply_roberts_operator(grayscale_image)

                if roberts_image is not None:
                    # Display the brightness histogram of the Roberts operator filtered image
                    display_brightness_histogram(roberts_image)
                    display_image(roberts_image, cmap = "gray")

