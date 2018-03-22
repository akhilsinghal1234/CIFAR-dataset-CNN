# import tensorflow as tf
import numpy as np
import pickle
from sklearn.cross_validation import train_test_split

train_path = '../cifar-10-batches-py/data_batch_'
test_path = '../cifar-10-batches-py/test_batch'

def to_img(raw):
	img_size = 32
	num_channels = 3
	raw_float = np.array(raw, dtype=float) / 255.0
	images = raw_float.reshape([-1, num_channels, img_size, img_size])
	images = images.transpose([0, 2, 3, 1])
	return images[0]

def as_array(data):
	d = []
	for i in range(len(data)):
		for j in range(len(data[i])):
			d.append(data[i][j])
	return d

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def split_train(train_d,train_y):
	index = 2
	valid_d = train_d[index]
	valid_y = train_y[index]
	train_d.pop(index)
	train_y.pop(index)
	valid_d = valid_d.tolist()
	return valid_d,valid_y,train_d,train_y 

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def one_hot(y):
	Y = np.zeros((len(y),10))
	Y[np.arange(len(y)),y] = 1
	return Y

def as_array1(data):
	d = []
	for i in range(len(data)):
		for j in range(len(data[i])):
			d.append(to_img(data[i][j]))
	return d

def get_data():
	train_d,train_y,test_d,test_y = [],[],[],[]
	for i in range(1,6):
		path = train_path + str(i)
		dict = unpickle(path)
		print(dict.get(b'batch_label'))
		train_d.append(dict.get(b'data'))
		train_y.append(dict.get(b'labels'))

	path = test_path
	dict = unpickle(path)
	print(dict.get(b'batch_label'))
	test_d.append(dict.get(b'data'))
	test_y.append(dict.get(b'labels'))

	img_y = as_array(train_y)
	test_img_y = as_array(test_y)
	img_Y = one_hot(img_y)
	test_img_Y = one_hot(test_img_y)

	img_X,test_img_X = [],[]
	img_X = as_array1(train_d)
	test_img_X = as_array1(test_d)

	t_train, v_train,  t_test, v_test = train_test_split(img_X, img_Y, test_size=0.20, random_state=42)
	print('---------data loaded-----------')

	return t_train,t_test,v_train,v_test, test_img_X, test_img_Y,test_img_y
	

