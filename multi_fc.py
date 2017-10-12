from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

# Silences some CPU computation warnings that TensorFlow throws my way
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save

image_size = 28
num_labels = 10

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

# Computes accuracy of predictions based on one-hot encoded vectors for labels
def get_accuracy(preds, labels):
	return 100.0 * np.mean(np.argmax(preds, 1) == np.argmax(labels, 1))

num_nodes = 1024
batch_size = 128

# Used to create multiple instances of fully connected layers for the network.
def fc_layer(input, size_in, size_out, name="fc"):
	with tf.name_scope(name):
		w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1),
						name="W")
		b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
		logits = tf.matmul(input, w) + b
		tf.summary.histogram("weights", w)
		tf.summary.histogram("biases", b)
		tf.summary.histogram("activations", logits)
		dict = {"logits": logits, "weights": w, "biases": b}
		return dict

# Takes in these hyperparameters in order to test multiple models through TensorBoard
def mnist_model(num_hidden, hparam_string, learning_rate=0.1, num_steps=10000):

	graph = tf.Graph()
	with graph.as_default():

		# Sets the datasets used in training, validation and testing
		tf_train_dataset = tf.placeholder(tf.float32, shape=(
			None, image_size * image_size), name="input")
		tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels),
									 name="labels")
		tf_valid_dataset = tf.constant(valid_dataset)
		tf_test_dataset = tf.constant(test_dataset)
		example_image = tf.reshape(tf_train_dataset,
								   [-1, image_size, image_size, 1])
		tf.summary.image("input", example_image, 1)

		# Start assembling the network
		network = []

		input_layer = fc_layer(tf_train_dataset, image_size * image_size, num_nodes,
							 name="input_layer")
		network.append(input_layer)
		input_act = tf.nn.relu(input_layer["logits"])

		hidden_layer_1 = fc_layer(input_act, num_nodes, num_nodes, name="hidden_layer_1")
		network.append(hidden_layer_1)
		hl1_act = tf.nn.relu(hidden_layer_1["logits"])

		# TODO: Need to find a more modular way to do this
		if num_hidden == 2:
			hidden_layer_2 = fc_layer(hl1_act, num_nodes,  num_nodes, name="hidden_layer_2")
			network.append(hidden_layer_2)
			hl2_act = tf.nn.relu(hidden_layer_2["logits"])

			output_layer = fc_layer(hl2_act, num_nodes, num_labels, name="output_layer")

		elif num_hidden == 3:

			hidden_layer_2 = fc_layer(hl1_act, num_nodes,  num_nodes, name="hidden_layer_2")
			network.append(hidden_layer_2)
			hl2_act = tf.nn.relu(hidden_layer_2["logits"])

			hidden_layer_3 = fc_layer(hl2_act, num_nodes,  num_nodes, name="hidden_layer_3")
			network.append(hidden_layer_3)
			hl3_act = tf.nn.relu(hidden_layer_3["logits"])

			output_layer = fc_layer(hl3_act, num_nodes, num_labels, name="output_layer")

		elif num_hidden == 4:
			hidden_layer_2 = fc_layer(hl1_act, num_nodes,  num_nodes, name="hidden_layer_2")
			network.append(hidden_layer_2)
			hl2_act = tf.nn.relu(hidden_layer_2["logits"])

			hidden_layer_3 = fc_layer(hl2_act, num_nodes,  num_nodes, name="hidden_layer_3")
			network.append(hidden_layer_3)
			hl3_act = tf.nn.relu(hidden_layer_3["logits"])

			hidden_layer_4 = fc_layer(hl3_act, num_nodes,  num_nodes, name="hidden_layer_3")
			network.append(hidden_layer_4)
			hl4_act = tf.nn.relu(hidden_layer_4["logits"])

			output_layer = fc_layer(hl4_act, num_nodes, num_labels, name="output_layer")
		else:
			output_layer = fc_layer(hl1_act, num_nodes, num_labels, name="output_layer")

		output_logits = output_layer["logits"]

		with tf.name_scope("loss"):
			# Keeping L2 regularization out of this model for other hyperparameter optimizations
			"""
			beta = 0.01
			l2_loss = 0
			for layer in network:
				l2_loss += tf.nn.l2_loss(layer["weights"])
		    l2_loss *= beta
			"""
			loss = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(logits=output_logits,
														labels=tf_train_labels))
			tf.summary.scalar("loss", loss)

		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
		train_prediction = tf.nn.softmax(logits=output_logits)

		with tf.name_scope("accuracy"):
			correct = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(tf_train_labels, 1))
			acc = 100 * tf.reduce_mean((tf.cast(correct, tf.float32)))
			# acc = 100.0 * np.mean(np.argmax(train_prediction) == np.argmax(tf_train_labels))
			tf.summary.scalar("accuracy", acc)

		def run_through_network(dataset):
			input = dataset
			for layer in network:
				w = layer["weights"]
				b = layer["biases"]
				run = tf.nn.relu(tf.matmul(input, w) + b)
				input = run
			output = tf.matmul(input, output_layer["weights"]) + output_layer["biases"]
			prediction = tf.nn.softmax(output)
			return prediction

		valid_prediction = run_through_network(tf_valid_dataset)
		test_prediction = run_through_network(tf_test_dataset)

		merged_summary = tf.summary.merge_all()

	with tf.Session(graph=graph) as sess:

		tf.global_variables_initializer().run()
		writer = tf.summary.FileWriter('./logs/varied_hidden_layers/'
									   + hparam_string, graph=sess.graph)

		print("Starting run for " + hparam_string + ": ")
		for step in range(num_steps):
			offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
			batch_data = train_dataset[offset:(offset + batch_size), :]
			batch_labels = train_labels[offset:(offset + batch_size), :]

			feed_dict = {tf_train_dataset: batch_data,
						 tf_train_labels: batch_labels}

			# Run the optimizer and retrieve useful stats on the training
			_, l, predictions = sess.run([optimizer, loss, train_prediction],
							 feed_dict=feed_dict)
			if step % 500 == 0:
				summary = sess.run(merged_summary, feed_dict=feed_dict)
				writer.add_summary(summary, step)
				print("Minibatch loss at step {}: {}".format(step, l))
				print("Minibatch accuracy: {:.1f}".format(
					get_accuracy(predictions, batch_labels)))
				print("Validation accuracy: {:.1f}".format(
					get_accuracy(valid_prediction.eval(), valid_labels)))

		print("Test accuracy: {:.1f}".format(get_accuracy(test_prediction.eval(), test_labels)))

# Creates TensorBoard logs for each set of hyperparameters to compare performance
for num_hidden in [2, 3, 4]:
	for lr in [0.01, 0.05, 0.1]:
		hparam_string = "num_hidden = {}, num_nodes_constant".format(num_hidden)
		mnist_model(num_hidden, hparam_string, lr, num_steps=10000)
