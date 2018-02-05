library(keras)

data(iris)

n = nrow(iris)
p = ncol(iris)

# Split the data in two subsamples, n/3 2n/3 approximately

Ind.test = c(sample(1:50,n/9),sample(51:100,n/9),sample(101:150,n/9))
Learn = iris[-Ind.test,]
Test = iris[Ind.test,]
	
print(dim(Learn))
print(dim(Test))
	
# create model
model <- keras_model_sequential()

# add layers and compile the model
model %>%
	layer_dense(units = 32, activation = 'relu', input_shape = c(4)) %>%
	layer_dense(units = 3, activation = 'softmax') %>%
	compile(
		optimizer = 'rmsprop',
		loss = 'categorical_crossentropy',
		metrics = c('accuracy')
		)

# Generate train data
x_train <- as.matrix(Learn[1:4])
y_train <- Learn[,5]

# Convert labels to categorical one-hot encoding
one_hot_labels_train <- to_categorical(as.numeric(y_train)-1, num_classes = 3)

# Generate test data
x_test <- as.matrix(Test[1:4])
y_test <- Test[,5]

# Convert test labels to one-hot
one_hot_labels_test <- to_categorical(as.numeric(y_test)-1, num_classes = 3)

# Train the model with batch of 32 samples
model %>% fit(x_train, one_hot_labels_train, epochs=10, batch_size=32)

# Evaluate the model
model %>% evaluate(x_test, one_hot_labels_test)