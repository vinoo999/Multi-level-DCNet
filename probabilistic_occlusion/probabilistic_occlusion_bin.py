from mlxtend.data import loadlocal_mnist
import sys
import numpy as np
import random
import imageio

out_dir = "" # Absolute or relative path to output directory
mnist_data_dir = "mnist_bin/" # Absolute or relative path to directory
                              # containing binary MNIST dataset files
occlusion_prob = .8 # 80% chance of occlusion of any given pixel

# Load the data into numpy arrays
X, y = loadlocal_mnist(
        images_path= mnist_data_dir + 't10k-images-idx3-ubyte',
        labels_path= mnist_data_dir + 't10k-labels-idx1-ubyte')

# Check the dimensions of the arrays for accuracy
print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('Digits:  0 1 2 3 4 5 6 7 8 9')
print('labels: %s' % np.unique(y))
print('Class distribution: %s' % np.bincount(y))

# Track progress in subjecting dataset to probabilistic occlusion
it_count = 0.0
tot = float(X.shape[0] * X.shape[1])

# Iterate over the numpy arrays representing the input images
for img_index in range(0, len(X)):
    for pix_index in range(0, len(X[img_index])):
        it_count += 1.0

        # For each pixel in the image, there is an occlusion_prob chance
        # that it will be set to zero (black)
        if random.random() < occlusion_prob:
            X[img_index][pix_index] = 0

        sys.stdout.write('\r')
        sys.stdout.write("{}%".format(round(it_count*100.0/tot, 2)))
        sys.stdout.flush()

# Save the numpy array representing the first image in the dataset (after
# subjecting it to probabilistic occlusion) as a PNG for testing purposes.
imageio.imwrite(out_dir + 'sample.png', np.split(X[0], 28))

# Output the transformed images in CSV format. Note: this is for demo purposes
# only, in practice we will use the transformed numpy array directly
np.savetxt(fname= out_dir + 'transformed_images.csv',
           X=X, delimiter=',', fmt='%d')
np.savetxt(fname= out_dir + 'labels.csv',
           X=y, delimiter=',', fmt='%d')
