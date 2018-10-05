from matplotlib import pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

print("Number of training examples:", mnist.train.num_examples)
print("Number of validation examples:", mnist.validation.num_examples)
print("Number of testing examples:", mnist.test.num_examples)

for i in range(3):
	first_image = mnist.test.images[i]
	print mnist.test.labels[i]
	first_image = np.array(first_image, dtype='float')
	pixels = first_image.reshape((28, 28))
	plt.imshow(pixels, cmap='gray')
	plt.show()

#indices = [i for i, x in enumerate(my_list) if x == "whatever"]