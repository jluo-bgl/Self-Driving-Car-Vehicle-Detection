import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle


def show_image(image, cmap=None):
    plt.figure()
    plt.imshow(image, cmap)


def calibrateCamera(img_size):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            show_image(img)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist


def save_camera_calibration(mtx, dist):
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("./output_images/camera_calibration_pickle.p", "wb"))


def load_camera_calibration(file_name):
    data = pickle.load(open(file_name, "rb"))
    return data["mtx"], data["dist"]


def undistort(mtx, dist, img, dst_file_name):
    """
        given camera mtx, dist and image, this function undistort img and display them side by side
    """
    img_size = (img.shape[1], img.shape[0])
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # cv2.imwrite(dst_file_name, dst)

    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    print("Visualize undistortion")
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.savefig(dst_file_name)


if __name__ == "__main__":
    img = cv2.imread('./camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])
    camera_matrix, distortion = calibrateCamera(img_size)
    save_camera_calibration(camera_matrix, distortion)
    undistort(camera_matrix, distortion, img, './output_images/calibration1_undist.png')

