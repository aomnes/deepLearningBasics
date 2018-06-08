# MNIST

Data set: train.npy, test.npy, valid.npy

* 3 layers: Accuracy `0.824`

		Convolution: 32 filters, with padding, kernel size x = 5, y = 5
		Max pooling: strides = 2, windows size x = 2, y = 2
		Dropout rate = 0.4
		
		Convolution: 64 filters, with padding, kernel size x = 5, y = 5
		Max pooling: strides = 2, windows size x = 2, y = 2
		Dropout rate = 0.4
		
		Feed forward: 1024 units Dropout rate = 0.4
		
* 5 layers: Accuracy `0.8225`

		Convolution: 32 filters, with padding, kernel size x =5, y = 5
		Max pooling: strides = 2, windows size x = 2, y = 2
		Dropout rate = 0.4
		
		Convolution: 64 filters, with padding, kernel size x = 5, y = 5
		Max pooling: strides = 2, windows size x = 2, y = 2
		Dropout rate = 0.4
		
		Convolution: 128 filters, with padding, kernel size x = 5, y = 5
		Dropout rate = 0.4
		
		Convolution: 128 filters, with padding, kernel size x = 5, y = 5
		Dropout rate = 0.4
		
		Feed forward: 1024 units Dropout rate = 0.4
		
* 7 layers: Accuracy `0.8295`

		Convolution: 32 filters, with padding, kernel size x = 5, y = 5
		Max pooling: strides = 2, windows size x = 2, y = 2
		Dropout rate = 0.4
		
		Convolution: 64 filters, with padding, kernel size x = 5, y = 5
		Max pooling: strides = 2, windows size x = 2, y = 2
		Dropout rate = 0.4
		
		Convolution: 128 filters, with padding, kernel size x = 5, y = 5
		Dropout rate = 0.4
		
		Convolution: 128 filters, with padding, kernel size  x= 5, y = 5
		Dropout rate = 0.4
		
		Convolution: 320 filters, with padding, kernel size x = 5, y = 5
		Dropout rate = 0.4
		
		Convolution: 320 filters, with padding, kernel size x = 5, y = 5
		Dropout rate = 0.4
		
		Feed forward: 1024 units Dropout rate = 0.4


# CIFAR-10

Data set: extra2-test_img.npy, extra2-train.npy, extra2-valid.npy

* 3 layers: Accuracy `0.4849`

		Convolution: filter= 96, kernel x = 3, y = 3, padding
		Max pooling: strides = 2, windows size x = 2, y = 2
		
		Convolution: filter= 96, kernel x = 3, y = 3, padding, stride = 2
		Max pooling: strides = 2, windows size x = 2, y = 2
		Dropout: rate = 0.2

		Feed Forward layer: units = 2048
		
* 5 layers: Accuracy `0.6766`

		Convolution: filter = 96, kernel x = 3, y = 3, padding
		Convolution: filter = 96, kernel x = 3, y = 3, padding, stride = 2
		Dropout: rate = 0.2
		
		Convolution: filter = 192, kernel x = 3, y = 3, padding
		Convolution: filter = 192, kernel x = 3, y = 3, padding, stride = 2
		Dropout: rate = 0.5

		Feed Forward layer: units = 2048
		
		
* 7 layers: Accuracy `0.6579`

		Convolution: filter = 96, kernel x = 3, y = 3, padding
		Convolution: filter = 96, kernel x = 3, y = 3, padding, stride = 2
		Dropout: rate = 0.2
		
		Convolution: filter = 192, kernel x = 3, y = 3, padding
		Convolution: filter = 192, kernel x = 3, y = 3, padding, stride = 2
		Dropout: rate = 0.5
		
		Convolution: filter= 256, kernel x = 3, y = 3, padding
		Max Pooling: size windows x=2, y=2, strides=2
		
		Convolution: filter= 256, kernel x = 3, y = 3, padding, stride=2
		Max Pooling: size windows x=2, y=2, strides=2
		Dropout: rate=0.5

		Feed Forward layer: units = 2048
