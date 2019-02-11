
# Demonstrate gradient descent for simple linear regression.

import sys
import csv



# try importing matplotlib
try:
	import matplotlib.pyplot as plt
	MATPLOTLIB_IMPORTED = True
except:
	MATPLOTLIB_IMPORTED = False

# determine python version
if sys.version_info[0] > 2:
	PYTHON_3 = True
else:
	PYTHON_3 = False

# import training data
training_data = []
with open("...",'r') as f:
	datareader = csv.reader(f)
	if PYTHON_3:
		headers = next(datareader)
	else:
		headers = datareader.next()
	for row in datareader:
		x = float(row[0])
		y = float(row[1])
		training_data.append([x,y])
print("Loaded {0} training data points.".format(len(training_data)))


# define functions
def model(x,m,b):
	# define the linear model
	return m*x + b

def calculate_loss(points,m,b):
	# define the loss function as the residual squared
	loss = 0
	for p in points:
		residual = p[1] - model(p[0],m,b)
		loss += residual * residual
	average_loss = loss / len(points)
	return average_loss

def adjust_parameters(points,m,b,learning_rate):
	# adjust model parameters using partial derivatives
	m_gradient = 0
	b_gradient = 0
	for p in points:
		m_gradient +=  -2 * (p[1] - model(p[0],m,b)) * p[0]
		b_gradient +=  -2 * (p[1] - model(p[0],m,b)) 
	new_m = m - learning_rate * m_gradient / len(points)
	new_b = b - learning_rate * b_gradient / len(points)
	return (new_m, new_b)


# define initial model parameters and learning rate 
m = 0
b = 0
learning_rate = 0.001

# train model: calculate cost with current parameters and adjust accordingly
loss_list = []
parameter_list = []
for i in range(100):
	parameter_list.append([m,b])
	loss = calculate_loss(training_data,m,b)
	loss_list.append(loss)
	print("Step: {:2d}, Loss: {:04.2f}, m: {:04.3f}, b: {:06.5f}".format(i,loss,m,b))
	m,b = adjust_parameters(training_data,m,b,learning_rate)

# display training process on graphs
if MATPLOTLIB_IMPORTED:

	f = plt.figure(figsize=(14,4))

	# plot results of training
	plt.subplot(1,3,1)
	plt.plot(loss_list,'.')
	plt.xlabel('Step #')
	plt.ylabel('Loss')

	# plot results of parameter evolution
	m_values = [p[0] for p in parameter_list]
	b_values = [p[1] for p in parameter_list]
	plt.subplot(1,3,2)
	plt.scatter(m_values,b_values)
	plt.xlabel('Parameter: m')
	plt.ylabel('Parameter: b')

	# plot results
	x_values = [p[0] for p in training_data]
	y_values = [p[1] for p in training_data]
	x_values_model = range(13)
	y_values_model = [model(x,m,b) for x in x_values_model]
	plt.subplot(1,3,3)
	plt.scatter(x_values,y_values)
	plt.plot(x_values_model,y_values_model,'r-')
	plt.xlabel('Miles')
	plt.ylabel('Ride price ($)')
	plt.ylim(0,20)
	plt.xlim(0,12)
	plt.show()


