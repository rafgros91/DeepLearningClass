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
	# Hidden Conv2d 
	layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(32, 32, 3)) %>%
	layer_activation("relu") %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	
	# Second hidden layer
	layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
	layer_activation("relu") %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	
	# Flatten max filtered output into feature vector
	layer_flatten() %>%
	
	# Dropout layer to try to avoid overfitting
	layer_dropout(0.5) %>%
	
	# Outputs from dense layer are projected onto 100 unit output layer
	layer_dense(100) %>%
	layer_activation("softmax")
	
# Compile Model
model %>% compile(
	optimizer = 'rmsprop',
	loss = 'categorical_crossentropy',
	metrics = c('accuracy')
	)

	summary(model)

# Fit
history <- model %>% fit(
    x_train, y_train,
    batch_size = 32,
    epochs = 50,
    validation_data = list(x_test, y_test),
    shuffle = TRUE
  )

plot(history)
