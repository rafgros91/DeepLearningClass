# Adapted from python code in 
# https://andrewkruger.github.io/projects/2017-08-05-keras-convolutional-neural-network-for-cifar-100

library(keras)

# Load Cifar100 dataset
cifar100 <- dataset_cifar100()
x_train <- cifar100$train$x/255
x_test <- cifar100$test$x/255
y_train <- to_categorical(cifar100$train$y, num_classes = 100)
y_test <- to_categorical(cifar100$test$y, num_classes = 100)

# Creating Model
model <- keras_model_sequential()
model %>%
	# First stack
	layer_conv_2d(filter = 128, kernel_size = c(3,3), padding = "same", input_shape = c(32, 32, 3)) %>%
	layer_activation("elu") %>%
	layer_conv_2d(filter = 128, kernel_size = c(3,3)) %>%
	layer_activation("elu") %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	layer_dropout(0.1) %>%
	# Second stack
	layer_conv_2d(filter = 256, kernel_size = c(3,3), padding = "same") %>%
	layer_activation("elu") %>%
	layer_conv_2d(filter = 256, kernel_size = c(3,3)) %>%
	layer_activation("elu") %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	layer_dropout(0.25) %>%
	# Third stack
	layer_conv_2d(filter = 512, kernel_size = c(3,3), padding = "same") %>%
	layer_activation("elu") %>%
	layer_conv_2d(filter = 512, kernel_size = c(3,3)) %>%
	layer_activation("elu") %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	layer_dropout(0.5) %>%
	# Final Dense Layer
	layer_flatten() %>%
	layer_dense(1024) %>%
	layer_activation("elu") %>%
	layer_dropout(0.5) %>%
	layer_dense(100) %>%
	layer_activation("softmax")
	
# Compile Model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

# Fit
history <- model %>% fit(
    x_train, y_train,
    batch_size = 32,
    epochs = 100,
    validation_data = list(x_test, y_test),
    shuffle = TRUE
  )