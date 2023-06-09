import cv2
import os
import numpy as np
import shutil

# Define the input and output directories
input_dir = '/workspaces/Covid-detection-FL/covid_and_non_xray_400_100/dataset_split/test/Normal'
output_dir = '/workspaces/Covid-detection-FL/after_preprocessing'


def do_it(input_dir,output_dir):
    # Define the size for resizing the images
    img_size = 224

    # Loop through each image in the input directory
    for filename in os.listdir(input_dir):
        # Read the image
        img = cv2.imread(os.path.join(input_dir, filename))

        # Resize the image to img_size x img_size
        img_resized = cv2.resize(img, (img_size, img_size))

        # Normalize the pixel intensity values to a range of 0-1
        img_normalized = img_resized / 255.0

        # Apply image augmentation techniques such as flip and rotation
        img_flipped = cv2.flip(img_normalized, 1)
        img_rotated = cv2.rotate(img_normalized, cv2.ROTATE_90_CLOCKWISE)

        # Apply image filtering techniques to remove noise
        img_blurred = cv2.GaussianBlur(img_normalized, (3, 3), 0)

        # # Apply image segmentation techniques to separate the lungs from the background
        # gray = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # mask = np.zeros((img_normalized.shape[0], img_normalized.shape[1]), dtype=np.uint8)
        # for i in range(len(contours)):
        #     if cv2.contourArea(contours[i]) > 100:
        #         cv2.drawContours(mask, [contours[i]], 0, (255, 255, 255), -1)
        # img_segmented = cv2.bitwise_and(img_normalized, img_normalized, mask=mask)

        # Save the preprocessed images in the output directory
        cv2.imwrite(os.path.join(output_dir, 'resized_' + filename), img_resized)
        cv2.imwrite(os.path.join(output_dir, 'normalized_' + filename), img_normalized * 255.0)
        cv2.imwrite(os.path.join(output_dir, 'flipped_' + filename), img_flipped * 255.0)
        cv2.imwrite(os.path.join(output_dir, 'rotated_' + filename), img_rotated * 255.0)
        cv2.imwrite(os.path.join(output_dir, 'blurred_' + filename), img_blurred * 255.0)
        # cv2.imwrite(os.path.join(output_dir, 'segmented_' + filename), img_segmented * 255.0)




def copy_directory(src_dir, dest_dir):
    try:
        shutil.copytree(src_dir, dest_dir)
        print(f"Directory copied from {src_dir} to {dest_dir}")
    except Exception as e:
        print(f"Error while copying directory from {src_dir} to {dest_dir}: {e}")



def create_folder(directory, folder_name):
    """
    Creates a new folder with the given name within the specified directory.

    Args:
    - directory (str): The path to the directory where the new folder should be created.
    - folder_name (str): The name of the new folder.

    Returns:
    - None
    """
    # Use os.path.join() to join the directory path and folder name
    new_folder_path = os.path.join(directory, folder_name)

    # Use os.makedirs() to create the new folder
    try:
        os.makedirs(new_folder_path)
        print(f"Folder '{folder_name}' created successfully in '{directory}'")
    except FileExistsError:
        print(f"Folder '{folder_name}' already exists in '{directory}'")
    except Exception as e:
        print(f"Error while creating folder '{folder_name}' in '{directory}': {e}")




create_folder('/workspaces/Covid-detection-FL/after_preprocessing/dataset_split/client1','train/COVID')
create_folder('/workspaces/Covid-detection-FL/after_preprocessing/dataset_split/client1','train/Normal')
do_it('/workspaces/Covid-detection-FL/dataset_split/client1/train/COVID','/workspaces/Covid-detection-FL/after_preprocessing/dataset_split/client1/train/COVID')
do_it('/workspaces/Covid-detection-FL/dataset_split/client1/train/Normal','/workspaces/Covid-detection-FL/after_preprocessing/dataset_split/client1/train/Normal')



create_folder('/workspaces/Covid-detection-FL/after_preprocessing/dataset_split/client2','train/COVID')
create_folder('/workspaces/Covid-detection-FL/after_preprocessing/dataset_split/client2','train/Normal')
do_it('/workspaces/Covid-detection-FL/dataset_split/client2/train/COVID','/workspaces/Covid-detection-FL/after_preprocessing/dataset_split/client2/train/COVID')
do_it('/workspaces/Covid-detection-FL/dataset_split/client2/train/Normal','/workspaces/Covid-detection-FL/after_preprocessing/dataset_split/client2/train/Normal')

copy_directory('/workspaces/Covid-detection-FL/dataset_split/client2/test/COVID','/workspaces/Covid-detection-FL/after_preprocessing/dataset_split/client2/test/COVID')
copy_directory('/workspaces/Covid-detection-FL/dataset_split/client1/test/COVID','/workspaces/Covid-detection-FL/after_preprocessing/dataset_split/client1/test/COVID')
copy_directory('/workspaces/Covid-detection-FL/dataset_split/client2/test/Normal','/workspaces/Covid-detection-FL/after_preprocessing/dataset_split/client2/test/Normal')
copy_directory('/workspaces/Covid-detection-FL/dataset_split/client1/test/Normal','/workspaces/Covid-detection-FL/after_preprocessing/dataset_split/client1/test/Normal')
