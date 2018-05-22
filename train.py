#!/usr/bin/env python
"""Chainer example: train a CNN on NDLKANA dataset
Modified from examples/mnist programs of Chainer 1.9.0.
"""
from __future__ import print_function
import argparse
import os
import time

import numpy as np
import six

import chainer
from chainer import computational_graph
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import data
import net


def isdir(arg):
	if not os.path.isdir(arg):
		msg = "directory %s not found" % arg
		raise argparse.ArgumentTypeError(msg)
	return arg


parser = argparse.ArgumentParser(description='Chainer example: NDLKANA')
parser.add_argument('--initmodel', '-m', default='',
					help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
					help='Resume the optimization from snapshot')
parser.add_argument('--epoch', '-e', default=20, type=int,
					help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=100,
					help='learning minibatch size')
parser.add_argument('--datadir', '-d', type=isdir, required=True,
					help='data directory')
parser.add_argument('--fspec', '-f', default='*.png',
					help='file spec of images')
parser.add_argument('--testratio', '-t', type=float, default=1.0/7.0,
					help='ratio of test data')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
print(args)
basename = os.path.split(args.datadir)[1]

batchsize = args.batchsize
n_epoch = args.epoch

# Prepare dataset
print('load NDLKANA dataset')
ndlkana = data.load_ndlkana_data(args.datadir, args.fspec,
									args.testratio)
ndlkana['data'] = ndlkana['data'].astype(np.float32)
ndlkana['data'] /= 255
ndlkana['target'] = ndlkana['target'].astype(np.int32)
n_test = ndlkana['testsize']
n_train = ndlkana['data'].shape[0] - n_test
print("n_train={} n_test={}".format(n_train, n_test))

x_train, x_test = np.split(ndlkana['data'], [n_train])
y_train, y_test = np.split(ndlkana['target'], [n_train])

# Prepare CNN model, defined in net.py
model = L.Classifier(net.NdlkanaCNN())
print("n_class={}".format(data.n_class))
xp = np
if args.gpu >= 0:
    # Make a specified GPU current
    chainer.backends.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()  # Copy the model to the GPU
    xp = chainer.backends.cuda.cupy

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)
if args.gpu >= 0:
	optimizer.target.to_gpu(args.gpu)

# Init/Resume
if args.initmodel:
	print('Load model from', args.initmodel)
	serializers.load_npz(args.initmodel, model)
if args.resume:
	print('Load optimizer state from', args.resume)
	serializers.load_npz(args.resume, optimizer)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
	print('epoch', epoch)

	# training
	perm = np.random.permutation(n_train)
	sum_accuracy = 0
	sum_loss = 0
	start = time.time()
	for i in six.moves.range(0, n_train, batchsize):
		x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
		t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))

		# Pass the loss function (Classifier defines it) and its arguments
		optimizer.update(model, x, t)

		if epoch == 1 and i == 0:
			with open(basename + '.dot', 'w') as o:
				variable_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0',
								  'style': 'filled'}
				function_style = {'shape': 'record', 'fillcolor': '#6495ED',
								  'style': 'filled'}
				g = computational_graph.build_computational_graph(
					(model.loss, ),
					variable_style=variable_style,
					function_style=function_style)
				o.write(g.dump())
			print('graph generated')

		sum_loss += float(model.loss.data) * len(t.data)
		sum_accuracy += float(model.accuracy.data) * len(t.data)
	end = time.time()
	elapsed_time = end - start
	throughput = n_train / elapsed_time
	print('train mean loss={:.5f}, accuracy={:.5f}, throughput={:.1f} images/sec'.format(
		sum_loss / n_train, sum_accuracy / n_train, throughput))

	# evaluation
	sum_accuracy = 0
	sum_loss = 0
	for i in six.moves.range(0, n_test, batchsize):
		x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]))
		t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]))
		with chainer.no_backprop_mode():
			loss = model(x, t)
		sum_loss += float(loss.data) * len(t.data)
		sum_accuracy += float(model.accuracy.data) * len(t.data)

	print('test  mean loss={:.5f}, accuracy={:.5f}'.format(
		sum_loss / n_test, sum_accuracy / n_test))

# Save the model and the optimizer
print('save the model')
serializers.save_npz(basename + '.model', model)
print('save the optimizer')
serializers.save_npz(basename + '.state', optimizer)
