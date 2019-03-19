import time
import socket
import os

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

import tensorflow as tf

import signal
import sys
import time

graph_global = None
tensor = None

NUM_ITERACOES = 10
NUM_IMAGES = 1
SOMA = 0
IMAGE_SIZE = (12, 8)

def inserirImagens(qtd):
	global tensor
	global NUM_IMAGES

	aux = []
	for i in range(qtd):
		image = Image.open("/my_images/" + str(i) + ".jpg")
		#print(image.getdata())
		(im_width, im_height) = image.size
		image = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
		aux.append(image)

	aux = np.array(aux)
	print("Inserindo um vetor de imagens de shape")
	print(aux.shape)

	tensor = aux

# Gambiarra para armazenar o graph (que é muito grande) no agente
def loadModel():
	global graph_global
	# We load the protobuf file from the disk and parse it to retrieve the 
	# unserialized graph_def
	with tf.gfile.GFile("/map_misc_finder2/frozen_inference_graph.pb", "rb") as f:
        f = f.encode('utf8','surrogateescape')
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	# Then, we import the graph_def into a new Graph and returns it 
	with tf.Graph().as_default() as graph:
		# The name var will prefix every op/nodes in your graph
		# Since we load everything in a new graph, this is not needed
		tf.import_graph_def(graph_def, name="prefix")

	graph_global = graph
	#time.sleep(10)


def inferencia(sess):
	# Carrega alguma imagem para teste
	#image = Image.open(PATH + "/imgs/2.jpg")
	#print(image.getdata())
	#(im_width, im_height) = image.size
	#image = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


	#im = Image.open('imgs/2.jpg')
	#im = im.resize((300, 300), Image.ANTIALIAS)
	#draw = ImageDraw.Draw(im)

	global NUM_IMAGES
	global NUM_ITERACOES
	global SOMA

	#print("ok, criou a sessão")
	#self.set_attr(bufferResposta = sess.run(y, feed_dict={x: np.expand_dims(image, 0)}))
	
	for i in range(5):
		#inserirImagens(NUM_IMAGES)


		image = Image.open("/my_images/image" + str(i) + ".jpg")
		#print(image.getdata())
		(im_width, im_height) = image.size
		image = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

		start = time.time()
		res = sess.run(y, feed_dict={x: np.expand_dims(image, 0)})
		end = time.time()
		print("Tempo decorrido: ")
		tempo_total = end - start
		SOMA = SOMA + tempo_total
		print(end - start)

		plt.figure(figsize=IMAGE_SIZE)
		plt.imshow(image)
		plt.show()
			

	print("Média de tempo para " + str(NUM_ITERACOES) + " com " + str(NUM_IMAGES) + " imagens: " + str(SOMA / NUM_ITERACOES))


loadModel()


#image = np.zeros((300,300,3))
x = graph_global.get_tensor_by_name('prefix/image_tensor:0')
y = graph_global.get_tensor_by_name('prefix/detection_boxes:0')

with tf.Session(graph=graph_global) as sess:
	inferencia(sess)