import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

# from skimage.feature import hog
# from skimage import color, exposure
# images are divided up into vehicles and non-vehicles

car_file_names = glob.glob('data/vehicles/*/*.png')
notcar_file_names = glob.glob('data/non-vehicles/*/*.png')


if __name__ == 'main':
    # Define a function to return some characteristics of the dataset
    def data_look(car_list, notcar_list):
        data_dict = {}
        # Define a key in data_dict "n_cars" and store the number of car images
        data_dict["n_cars"] = len(car_list)
        # Define a key "n_notcars" and store the number of notcar images
        data_dict["n_notcars"] = len(notcar_list)
        # Read in a test image, either car or notcar
        example_img = mpimg.imread(car_list[0])
        # Define a key "image_shape" and store the test image shape 3-tuple
        data_dict["image_shape"] = example_img.shape
        # Define a key "data_type" and store the data type of the test image.
        data_dict["data_type"] = example_img.dtype
        # Return data_dict
        return data_dict


    data_info = data_look(car_file_names, notcar_file_names)

    print('Your function returned a count of',
          data_info["n_cars"], ' cars and',
          data_info["n_notcars"], ' non-cars')
    print('of size: ', data_info["image_shape"], ' and data type:',
          data_info["data_type"])
    # Just for fun choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, len(car_file_names))
    notcar_ind = np.random.randint(0, len(notcar_file_names))

    # Read in car / not-car images
    car_image = mpimg.imread(car_file_names[car_ind])
    notcar_image = mpimg.imread(notcar_file_names[notcar_ind])

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
