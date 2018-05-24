#Ethan Hartzell
#backpropogation implementation
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from math import log, exp, e
import random
#start with .arff file reader
random.seed(0)

#reads in and stores info about a .arff file
class arff_file:
	def __init__(self,filename):
		self.relation = ""		#relation name
		self.attributes = []	#feature names
		self.classes = []
		self.data = []	
		self.labels = []		#examples
		self.file = filename
		f = open(filename)		#open the file
		tmp = f.readlines()
		data = False
		#store the relevent parts of the file
		for line in tmp:		#for each line
		    ls = line.split()	#for each word
		    for i, word in enumerate(ls):
				if data:
					if line.upper() != "@DATA":
						self.data.append(line.split(",")[:-1])	#append all but last word
						self.labels.append(line.split(",")[-1].replace("\n",""))
					continue
				if word.upper() == "@Relation":
					self.relation = ls[i+1]	#relation gets next word
					continue
				if word.upper() == "@ATTRIBUTE":
					if "class" not in line:
						self.attributes.append(line)	#save the whole line for the attribute
					elif "class" in line.split():
						for c in line.split(","):
							get = c.replace("}","")
							get = get.replace("@attribute class {","")
							get = get.replace("\n","")
							self.classes.append(get)
				if word.upper() == "@DATA":
					data = True
					continue
		f.close()
		for count, example in enumerate(self.data):
			for c2,num in enumerate(example):
				self.data[count][c2] = float(num)
	def get_data(self):
		return self.data
	def get_labels(self):
		return self.labels
	def replace_data(self,nd):
		self.data = nd
	def onehot_encoding(self):
		encodings = {}
		for count,c in enumerate(self.classes):
			code = np.zeros(len(self.classes))
			code[count] = 1
			code = code.tolist()
			encodings[c] = tuple(code)
		return encodings
	def onehot_decoding(self):
		decodings = {}
		for count,c in enumerate(self.classes):
			code = np.zeros(len(self.classes))
			code[count] = 1
			code = code.tolist()
			decodings[tuple(code)] = c
		return decodings


#now the classifier

class neural_net:
	def __init__(self, dset, d, w):	#takes the training set with the number of hidden nodes
		self.num_input_nodes = len(dset.attributes)
		self.num_output_nodes = len(dset.classes)
		self.depth = d
		self.width = w
		self.weights = defaultdict(float) #will store (node,node) => weight
		#input nodes named by i0, i1, i2 etc. 
		#hidden nodes named h00, h01, h02, h03, h10, h11, h12 (hdw -- which layer then which one in that layer)
		#output nodes named o1,o2,o3 etc

		self.examples = dset.get_data()
		self.labels = dset.get_labels()
		self.encodings = dset.onehot_encoding()	#this dictionary helps convert labels to onehot encodings
		self.decodings = dset.onehot_decoding()	#this dictionary helps convert onehot encodings back into labels
		self.init_weights()
		
	#randomly initializes the weights
	def init_weights(self):
		if self.depth > 0:
			for i in range(0,self.num_input_nodes):	#for each input node connect to first hidden layer
				for j in range(0,self.width):
					self.weights[("i"+str(i),"h0"+str(j))] = random.uniform(-.1,.1)
			if self.depth != 1:
				for i in range(0,self.depth-1):		#for each hidden layer except the last, 
					for j in range(0,self.width):	#for each node in layer
						for k in range(0,self.width):	#connect to each hnode in next layer
							self.weights[("h"+str(i)+str(j),"h"+str(i+1)+str(k))] = random.uniform(-.1,.1)
			for i in range(0,self.width):	#for each node in last hidden layer, connect to output
				for j in range(0,self.num_output_nodes):
					self.weights[("h"+str(self.depth-1)+str(i),"o"+str(j))] = random.uniform(-.1,.1)
		elif self.depth == 0:
			for i in range(0,self.num_input_nodes):	#just connect input to output directly
				for j in range(0,self.num_output_nodes):
					self.weights[("i"+str(i),"o"+str(j))] = random.uniform(-.1,.1)
		
	#the activation function, in this case sigmoid
	def activation(self,x):
		return 1.0/(1.0+e**(-1*(max(-50.0,x))))
		#return 1.0/(1.0+e**(-1*x))
	#the derivative of the activation function
	def sigprime(self,x):
		return self.activation(x)*(1.0-self.activation(x))
	#trains the neural network using backpropagation, returns array containing error rate after each iteration
	def train(self,iterations,learning_rate=0.1):
		mistakes = []
		exi = self.examples
		
		for i in range(0,iterations):	#loop "iterations" times
			print "iteration: " + str(i)
			for count, ex in enumerate(exi):	#for each example
				nodes = defaultdict(float)		#keep a dictionary of node values
				derived_nodes = defaultdict(float)
					#keep a dictionary of nodes put through the derivative of the activation function
				#backpropagation algorithm
				#here is the forward pass
				if self.depth > 0:
					for inode in range(0,self.num_input_nodes): #do the forward pass from input to first hidden layer
						for hnode in range(0,self.width):
							nodes["h0"+str(hnode)] += ex[inode]*self.weights[("i"+str(inode),"h0"+str(hnode))] 
					#now put those sums through activation function
					for hnode in range(0,self.width):
						derived_nodes["h0"+str(hnode)] = self.sigprime(nodes["h0"+str(hnode)])
						nodes["h0"+str(hnode)] = self.activation(nodes["h0"+str(hnode)])
					#now do forward pass from h0 to h1, and so on until last hidden layer
					if self.depth != 1:
						for layer in range(0,self.depth-1):	
							for node in range(0,self.width):
								for nextlayernode in range(0,self.width):
									nodes["h"+str(layer+1)+str(nextlayernode)] += nodes["h"+str(layer)+str(node)] * self.weights[("h"+str(layer)+str(node),"h"+str(layer+1)+str(nextlayernode))] 
							#put those sums through activation function
							for nextlayernode in range(0,self.width):
								derived_nodes["h"+str(layer+1)+str(nextlayernode)] = self.sigprime(nodes["h"+str(layer+1)+str(nextlayernode)])
								nodes["h"+str(layer+1)+str(nextlayernode)] = self.activation(nodes["h"+str(layer+1)+str(nextlayernode)])
					#now finish forward pass from hfinal to output layer
					for node in range(0,self.num_output_nodes):
						for hnode in range(0,self.width):
							nodes["o"+str(node)] += nodes["h"+str(self.depth-1)+str(hnode)] * self.weights[("h"+str(self.depth-1)+str(hnode),"o"+str(node))]
					#need put through activation function
					for onode in range(0,self.num_output_nodes):
						derived_nodes["o"+str(onode)] = self.sigprime(nodes["o"+str(onode)])
						nodes["o"+str(onode)] = self.activation(nodes["o"+str(onode)])
				elif self.depth == 0:
					#do forward pass from input to output later
					for inode in range(0,self.num_input_nodes):
						for onode in range(0,self.num_output_nodes):
							nodes["o"+str(onode)] += ex[inode]*self.weights[("i"+str(inode),"o"+str(onode))]
					for onode in range(0,self.num_output_nodes):
						derived_nodes["o"+str(onode)] = self.sigprime(nodes["o"+str(onode)])
						nodes["o"+str(onode)] = self.activation(nodes["o"+str(onode)])
				if self.depth == 1 and self.width == 3:
					if i == iterations-1:
						print ex
						for hnode in nodes.keys():
							if "h" in hnode:
								print str(hnode) + " = " + str(nodes[hnode])
				outputs = np.zeros(self.num_output_nodes)
				for onode in range(0,self.num_output_nodes):
					outputs[onode]=nodes["o"+str(onode)]
				label = np.argmax(outputs)
				encoding = np.zeros(self.num_output_nodes)
				encoding[label] = 1
				choice = self.decodings[tuple(encoding)]
				if choice != self.labels[count]:
					#now for the backward pass
					#calculate delta at each output node
					for onode in range(0,self.num_output_nodes):
						nodes["o"+str(onode)] = -1*derived_nodes["o"+str(onode)]*(self.encodings[self.labels[count]][onode] - nodes["o"+str(onode)])
					if self.depth > 0:	#if we have ANY hidden layers
						#now go through and update every weight from last layer to output layer and calculate delta there
						#do last layer first because num output nodes may differ from width of hnode layers
						for hnode in range(0,self.width):
							delta = derived_nodes["h"+str(self.depth-1)+str(hnode)]
							dsum = 0.0
							for onode in range(0,self.num_output_nodes):	#update weights between last hidden layer and output layer
								dsum += self.weights[("h"+str(self.depth-1)+str(hnode),"o"+str(onode))] * nodes["o"+str(onode)]
								self.weights[("h"+str(self.depth-1)+str(hnode),"o"+str(onode))] -= learning_rate*nodes["h"+str(self.depth-1)+str(hnode)]*nodes["o"+str(onode)]
							nodes["h"+str(self.depth-1)+str(hnode)] = delta * dsum
						#update the weights between hidden layers
						if self.depth > 2:	#if more than two hidden layers
							for layer in range(self.depth-2,-1,-1):	#from the second to last layer to the first
								for hnode_upper in range(0,self.width):	#update the weights between this layer and the one above
									delta = derived_nodes["h"+str(layer)+str(hnode)]
									dsum = 0.0
									for hnode in range(0,self.width):
										dsum += self.weights[("h"+str(layer)+str(hnode),"h"+str(layer+1)+str(hnode_upper))] * nodes["h"+str(layer+1)+str(hnode_upper)]
										self.weights[("h"+str(layer)+str(hnode),"h"+str(layer+1)+str(hnode_upper))] -= learning_rate*nodes["h"+str(layer)+str(hnode)]*nodes["h"+str(layer+1)+str(hnode_upper)]
									nodes["h"+str(layer)+str(hnode_upper)] = delta * dsum
						#update weights between hidden layers in case of two hlayers
						elif self.depth == 2:	#if only two layers
							#print "depth is definitely two"
							for hnode in range(0,self.width):
								delta = derived_nodes["h0"+str(hnode)]
								dsum = 0.0
								for hnode_upper in range(0,self.width):	#update nodes between first and second layer
									dsum += self.weights[("h0"+str(hnode),"h1"+str(hnode_upper))] * nodes["h1"+str(hnode_upper)]
									#print self.weights[("h0"+str(hnode),"h1"+str(hnode_upper))]
									self.weights[("h0"+str(hnode),"h1"+str(hnode_upper))] -= learning_rate*nodes["h0"+str(hnode)]*nodes["h1"+str(hnode_upper)]
									#print self.weights[("h0"+str(hnode),"h1"+str(hnode_upper))]					
								nodes[("h0"+str(hnode))] = delta*dsum
						#update weights between input layer and first hidden layer
						for inode in range(0,self.num_input_nodes):
							for hnode in range(0,self.width):
								self.weights[("i"+str(inode),"h0"+str(hnode))] -= learning_rate*ex[inode]*nodes["h0"+str(hnode)]
							#no need to update nodes here, backpropogation is finally over
					elif self.depth == 0:
						#update weights between input and output units
						for inode in range(0,self.num_input_nodes):
							for onode in range(0,self.num_output_nodes):
								self.weights[("i"+str(inode),"o"+str(onode))] -= learning_rate*ex[inode]*nodes["o"+str(onode)]
			mistakes.append(self.test(self.examples,self.labels))
		#print "the weights!"
		#print self.weights
		return mistakes

	#returns the net's best guest on an input
	def classify(self,example): #example comes as list of strings of numbers
		ex = example
		nodes = defaultdict(float)
		#backpropagation algorithm
		#here is the just the forward pass
		if self.depth > 0:
			for inode in range(0,self.num_input_nodes): #do the forward pass from input to first hidden layer
				for hnode in range(0,self.width):
					nodes["h0"+str(hnode)] += ex[inode]*self.weights[("i"+str(inode),"h0"+str(hnode))] 
			#now put those sums through activation function
			for hnode in range(0,self.width):
				nodes["h0"+str(hnode)] = self.activation(nodes["h0"+str(hnode)])
			#now do forward pass from h0 to h1, and so on until last hidden layer
			if self.depth != 1:
				for layer in range(0,self.depth-1):	#need to put sums through activation function before moving to next layer
					for node in range(0,self.width):
						for nextlayernode in range(0,self.width):
							nodes["h"+str(layer+1)+str(nextlayernode)] += nodes["h"+str(layer)+str(node)] * self.weights[("h"+str(layer)+str(node),"h"+str(layer+1)+str(nextlayernode))] 
					#put those sums through activation function
					for nextlayernode in range(0,self.width):
						nodes["h"+str(layer+1)+str(nextlayernode)] = self.activation(nodes["h"+str(layer+1)+str(nextlayernode)])
			#now finish forward pass from hfinal to output layer
			for node in range(0,self.num_output_nodes):
				for hnode in range(0,self.width):
					nodes["o"+str(node)] += nodes["h"+str(self.depth-1)+str(hnode)] * self.weights[("h"+str(self.depth-1)+str(hnode),"o"+str(node))]
			#need put through activation function
			for onode in range(0,self.num_output_nodes):
				nodes["o"+str(onode)] = self.activation(nodes["o"+str(onode)])
		elif self.depth == 0:
			#do forward pass from input to output later
			for inode in range(0,self.num_input_nodes):
				for onode in range(0,self.num_output_nodes):
					nodes["o"+str(onode)] += ex[inode]*self.weights[("i"+str(inode),"o"+str(onode))]
			for onode in range(0,self.num_output_nodes):
				nodes["o"+str(onode)] = self.activation(nodes["o"+str(onode)])
		outputs = np.zeros(self.num_output_nodes)
		for onode in range(0,self.num_output_nodes):
			outputs[onode]=nodes["o"+str(onode)]
		label = np.argmax(outputs)
		encoding = np.zeros(self.num_output_nodes)
		encoding[label] = 1
		return self.decodings[tuple(encoding)]
	#tests the classifier against a test set, returns an error rate
	def test(self,unlabeled_data,labels):
		error_count = 0.0
		total = len(labels)
		for count, example in enumerate(unlabeled_data):
			ex = self.classify(example)
			#print ex
			if ex != labels[count]:
				error_count += 1
		return error_count/total

#now do the experiments
exp = raw_input("Do you want to perform experiment 1 using 838.arff? Enter y or n to display train and display hidden node representation after 3000 iterations:    ")
if exp == "y":
	data = arff_file("838.arff")
	net = neural_net(data,1,3)
	#test = arff_file("optdigits_test")
	mistakes = []
	mistakes = net.train(3000)

	plt.plot(range(1,len(mistakes)+1),mistakes)
	plt.title("Mistakes over #iterations on 838 network")
	plt.xlabel("# of iterations")
	plt.ylabel("percent errors")
	plt.show()


#next experiment

d2 = arff_file("optdigits_train.arff")
d3 = arff_file("optdigits_test.arff")
exp2 = raw_input("Enter y or n if you want to run experiment 2 (200 iterations, width 5-40 on a 3-deep network) This may take a while.:       ")
if exp2 == "y":
	train_error = []
	test_error = []
	iter_error = []
	for width in range(5,45,5):
		newnet = neural_net(d2,3,width)
		iter_error.append(newnet.train(200))
		train_error.append(newnet.test(d2.get_data(),d2.get_labels()))
		test_error.append(newnet.test(d3.get_data(),d3.get_labels()))
	for r in iter_error:
		plt.plot(range(1,201),r)
	plt.title("Error rate over iterations on d=3 networks of different widths")
	plt.xlabel("iterations")
	plt.ylabel("percent errors")
	plt.show()

	plt.plot(range(5,45,5),train_error)
	plt.plot(range(5,45,5),test_error)
	plt.xlabel("width")
	plt.ylabel("percent error")
	plt.title("Error over widths on d=3 networks")
	plt.show()
#final experiment
exp3 = raw_input("if you want to run exp 3 on the varying depth, enter y (this may take a while). Otherwise, enter n:    ")
if exp3 == "y":
	trainerror2 = []
	testerror2 = []
	itererror2 = []
	for depth in range(0,6):
		newnet = neural_net(d2,depth,10)
		itererror2.append(newnet.train(200))
		trainerror2.append(newnet.test(d2.get_data(),d2.get_labels()))
		testerror2.append(newnet.test(d3.get_data(),d3.get_labels()))
	#print itererror2
	for p in itererror2:
		plt.plot(range(1,201),p)
	plt.title("error over iterations on w=10 networks of different depths")
	plt.xlabel("iterations")
	plt.ylabel("percent error")
	plt.show()
	"""print trainerror2
				print testerror2"""
	plt.plot(range(0,6),trainerror2)
	plt.plot(range(0,6),testerror2)
	plt.title("error over depth on w=10 networks of different depths")
	plt.xlabel("depth")
	plt.ylabel("percent error")
	plt.show()
while True:
	personal_exp = raw_input("To run your own experiment, specify training_file, test_file, depth, width, iterations, and learning rate separated by spaces (otherwise enter q to quit:    ")
	if personal_exp == "q":
		break
	else:
		training = arff_file(personal_exp.split()[0])
		testing = arff_file(personal_exp.split()[1])
		newnet = neural_net(training,personal_exp.split()[2],personal_exp.split()[3])
		newnet.train(personal_exp.split()[4],personal_exp.split()[5])
		print newnet.test(testing.get_data(),testing.get_labels())
		print "this is the error percent on your test set given your training set and parameters"