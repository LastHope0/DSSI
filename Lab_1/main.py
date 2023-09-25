import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import TextBox, Button
from scipy import ndimage
import numpy as np

# Define global variables to store file path and c value
file_path = "test/5.jpg"
c_value = 50.0

# Callback function for file path input
def on_file_path_change(text):
    global file_path
    file_path = text  # Update the global file path

# Callback function for c value input
def on_c_value_change(text):
    global c_value
    try:
        c_value = float(text)  # Update the global c value
    except ValueError:
        pass  # Ignore non-numeric input

# Callback function for the generate plot button
def generate_plot(event):
    # Reload the image using the updated file path
    loaded_image = load_image(file_path)

    if loaded_image is not None:
        # Create a grid for displaying images and histograms
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))

        # Display the original image
        axs[0, 0].imshow(loaded_image)
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')

        # Convert the color image to grayscale
        grayscale_image = color_to_grayscale(loaded_image)

        if grayscale_image is not None:
            # Display the grayscale image
            axs[0, 1].imshow(grayscale_image, cmap="gray")
            axs[0, 1].set_title('Grayscale Image')
            axs[0, 1].axis('off')

            # Display the brightness histogram of the grayscale image
            display_brightness_histogram(axs[1, 1], grayscale_image, 'Grayscale Histogram')

            # Apply logarithmic correction to the grayscale image
            corrected_image = apply_logarithmic_correction(grayscale_image, c_value)

            if corrected_image is not None:
                # Display the corrected image
                axs[0, 2].imshow(corrected_image, cmap='gray')
                axs[0, 2].set_title(f'Corrected Image (c={c_value})')
                axs[0, 2].axis('off')

                # Display the brightness histogram of the corrected image
                display_brightness_histogram(axs[1, 2], corrected_image, f'Corrected Histogram (c={c_value})')

                # Apply the Roberts operator filter to the grayscale image
                roberts_image = apply_roberts_operator(grayscale_image)
                if roberts_image is not None:
                    # Display the Roberts operator filtered image
                    axs[0, 3].imshow(roberts_image, cmap='gray')
                    axs[0, 3].set_title('Roberts Image')
                    axs[0, 3].axis('off')

                    # Display the brightness histogram of the Roberts operator filtered image
                    display_brightness_histogram(axs[1, 3], roberts_image, 'Roberts Histogram')

                    # Adjust spacing between subplots
                    plt.tight_layout()

                    # Show the combined plot
                    plt.show()

# Function to load an image
def load_image(file_path):
    try:
        img = mpimg.imread(file_path)
        return img
    except Exception as e:
        print(f"An error occurred while loading the image: {str(e)}")
        return None

# Convolve
def convolve_2d(image, kernel):
    # Get dimensions of the input image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding required for valid convolution
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Create an output array to store the result
    output = np.zeros((image_height, image_width))

    # Pad the input image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Perform 2D convolution
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest (ROI) from the padded image
            roi = padded_image[i:i + kernel_height, j:j + kernel_width]

            # Apply the convolution operation
            output[i, j] = np.sum(roi * kernel)

    return output

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
        if len(image.shape) == 2:
            return image
        if len(image.shape) == 3:
            # Apply the formula to calculate grayscale values
            grayscale_image = np.dot(image[..., :3], [0.3, 0.59, 0.11])         
            print(grayscale_image)
            return grayscale_image
        else:
            print("Input image is not a color image (RGB).")
            return None
    except Exception as e:
        print(f"An error occurred during color to grayscale conversion: {str(e)}")
        return None

# Function to display the brightness histogram of an image
def display_brightness_histogram(ax, gray_image, title):
    try:
        # Ensure the image is in grayscale
        if len(gray_image.shape) == 2:
            
            # Initialize an array to store the histogram
            histogram = list(range(256)) # 256 bins for pixel intensities ranging from 0 to 255
            
            # Calculate the histogram manually
            for pixel_value in gray_image.flatten().astype(np.uint8):
                histogram[pixel_value] += 1
            
            # Plot the histogram
            bins = list(range(256))
            ax.bar(bins, histogram, width=1, color='gray')
            ax.set_xlim([0, 256])
            ax.set_title(title)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
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
            min = corrected_image.min()
            max = corrected_image.max()
            if(max > 255):
                corrected_image = ((corrected_image - min) / (max - min) * 255).astype(np.uint8)

            return corrected_image
        else:
            print("Input image must be grayscale.")
            return None
    except Exception as e:
        print(f"An error occurred during logarithmic correction: {str(e)}")
        return None

# Function to apply the Roberts operator filter +to an image
def apply_roberts_operator(gray_image):
    try:
        # Ensure the image is in grayscale
        if len(gray_image.shape) == 2:
            # Define the Roberts operator kernels
            kernel_x = np.array([[1, 0], [0, -1]])
            kernel_y = np.array([[0, 1], [-1, 0]])

            # Convolve the image with the Roberts operator kernels
            gradient_x = np.abs(convolve_2d(gray_image, kernel_x ))
            gradient_y = np.abs(convolve_2d( gray_image, kernel_y))

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


if __name__ == "__main__":
    
    # Create a separate figure for input elements
    input_fig = plt.figure(figsize=(10,3))

    # Create text input boxes for file path and c value
    file_path_text_box_ax = plt.axes([0.3, 0.6, 0.4, 0.1])
    file_path_text_box = TextBox(file_path_text_box_ax, 'Path:', initial=file_path)
    file_path_text_box.on_submit(on_file_path_change)

    c_value_text_box_ax = plt.axes([0.3, 0.4, 0.4, 0.1])
    c_value_text_box = TextBox(c_value_text_box_ax, 'c:', initial=str(c_value))
    c_value_text_box.on_submit(on_c_value_change)

    # Create a button to generate the plot
    generate_plot_button_ax = plt.axes([0.3, 0.2, 0.4, 0.1])
    generate_plot_button = Button(generate_plot_button_ax, 'Generate')
    generate_plot_button.on_clicked(generate_plot)

    # Show the input elements figure
    plt.show()