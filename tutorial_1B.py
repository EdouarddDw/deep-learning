#question 1

import torch
a = torch.tensor(list(range(9)))
print("a")

#Predict the size, storage offset, and stride of a.
print(a.size())           # The shape of the tensor.
print(a.storage_offset()) # Where the tensor starts in the underlying memory block.
print(a.stride())        # How you move in memory to get to the next element along each dimension.
# The expected output is:
# torch.Size([9])
# 0
# (1,)

#question b
print("\nquestion b")
b = a.view(3, 3)
print(a)
print(b)

#what does view do?
# The view function in PyTorch is used to reshape a tensor without changing its data.

# Predict the size, offset, and stride
print(b.size())           
print(b.storage_offset()) 
print(b.stride())

#pridited output:
# torch.Size([3, 3])
# 0
# (3, 1)

#Do a and b share the same underlying storage?
print(a.storage().data_ptr() == b.storage().data_ptr())
#yes they do

#How does this differ from creating a copy?
# Creating a copy of a tensor allocates new memory and duplicates the data, so the original and the copy do not share the same underlying storage. Changes made to one will not affect the other.

#question c
print("\nquestion c")
c = b[1:, 1:]
print(b)
print(c)

# Predict the size, offset, and stride
print(c.size())
print(c.storage_offset())
print(c.stride())
#predicted output:
# torch.Size([2, 2])
# 4
# (3, 1)
#Why can slicing change the stride but not necessarily the storage?
# Slicing a tensor creates a new view of the original tensor that may have different strides to reflect the new shape and layout of the data. However, the underlying storage remains the same, so the storage offset may change based on where the slice starts in the original tensor.

#question d
print("\nquestion d")

#I chose the sigmoid function

#Apply it element-wise to a. Why does this result in an error?
#d = torch.sigmoid_(a)
#print(d)
#RuntimeError: result type Float can't be cast to the desired output type Long

#What operation is required to make the function work?
d = torch.sigmoid_(a.float())
print(d)

#Is there an in-place version of this operation? What does “in-place” mean in this context?
# Yes, the in-place version of the sigmoid operation is torch.sigmoid_. In-place means that the operation modifies the original tensor directly without creating a new tensor.


# question 2

# question a
#using PyTorch tensors, define a scalar-valued function using PyTorch operations
#(e.g. addition, multiplication, and a non-smooth operation such as max).
print("\nquestion 2a")

# Create leaf tensors that require gradients
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# Define a scalar-valued function using PyTorch operations
# multiplication + a non-smooth op (max) + sum to ensure a scalar
z = torch.max(x * y, torch.tensor([0.0])).sum()
print(z)

# x and y should have requires_grad=True because we want dz/dx and dz/dy.

# question b
print("\nquestion 2b")
# Execute the backward pass (z is a scalar)
z.backward()

print("x.grad:", x.grad)
print("y.grad:", y.grad)

# .grad exists (and is populated) for leaf tensors with requires_grad=True.
# z is an intermediate result, so z.grad is not populated unless you call z.retain_grad().

#Why are gradients accumulated rather than overwritten?
# Gradients are accumulated rather than overwritten to support scenarios where multiple backward passes are performed on the same graph, allowing for the aggregation of gradients from different paths in the computation graph.

#question C
#What happens if you call backward() twice without resetting gradients?
print("\nquestion 2c")
# Create leaf tensors that require gradients
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)
# Define a scalar-valued function using PyTorch operations
z = torch.max(x * y, torch.tensor([0.0])).sum()
# Execute the backward pass (z is a scalar)
z.backward()
print("First backward call:")
print("x.grad:", x.grad)
print("y.grad:", y.grad)
# Call backward again without resetting gradients
#z.backward()
#print("Second backward call:")
#print("x.grad:", x.grad)
#print("y.grad:", y.grad)
# RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
# To avoid this error, you can reset the gradients before the second backward call:

#What happens if you detach part of the graph?
print("\nquestion 2C detach")
# Create leaf tensors that require gradients
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)
# Define a scalar-valued function using PyTorch operations with detachment
z = torch.max(x * y.detach(), torch.tensor([0.0])).sum()
# Execute the backward pass (z is a scalar)
z.backward()
print("After detaching y:")
print("x.grad:", x.grad)
print("y.grad:", y.grad)

#y.grad: None
# Detaching part of the graph (y in this case) means that gradients will not be computed for that part. As a result, y.grad remains None because y was detached from the computation graph, preventing gradient flow back to it.

#How do these behaviors relate to the computational graph view from Session 1A?

#Question 3
print("\nquestion 3")
#linear regression with manual gradients 

w = torch.ones((), requires_grad=True)
b = torch.zeros((), requires_grad=True)

# Define the linear model
def model(x, w, b):
    return w * x + b


# Define the mean squared error loss function
def mean_squared_error(y_pred, y_true):
    squared_diffs = (y_pred - y_true) ** 2
    return squared_diffs.mean()


# train data
x = torch.tensor([1., 3., 5., 7., 9., 11., 13., 15.])
y = torch.tensor([3.6, 8.5, 10.8, 12.5, 20.8, 26.2, 25.6, 28.3])

#graph x and y in a scatter plot
import matplotlib.pyplot as plt
plt.scatter(x.numpy(), y.numpy())
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of training data')
plt.show()

# calculate initial loss
y_pred = model(x, w, b)
loss = mean_squared_error(y_pred, y)
print(f'Initial loss: {loss.item()}')

# Manually compute gradients using finite differences
delta = 0.1
loss_rate_of_change_w = \
    (mean_squared_error(model(x, w + delta, b), y) - mean_squared_error(model(x, w - delta, b), y)) / (2 * delta)

learning_rate = 1e-2

w = w - learning_rate * loss_rate_of_change_w

#same for b
loss_rate_of_change_b = \
    (mean_squared_error(model(x, w, b + delta), y) - mean_squared_error(model(x, w, b - delta), y)) / (2 * delta)

b = b - learning_rate * loss_rate_of_change_b

print(f'Updated w: {w.item()}, b: {b.item()}')
y_pred = model(x, w, b)
loss = mean_squared_error(y_pred, y)
print(f'Loss after one update: {loss.item()}')


# same but in a loop
# Train for multiple epochs
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(x, w, b)
    loss = mean_squared_error(y_pred, y)
    # Manually compute gradients using finite differences
    loss_rate_of_change_w = \
        (mean_squared_error(model(x, w + delta, b), y) - mean_squared_error(model(x, w - delta, b), y)) / (2 * delta)
    loss
    rate_of_change_b = \
        (mean_squared_error(model(x, w, b + delta), y) - mean_squared_error(model(x, w, b - delta), y)) / (2 * delta)
    # Update parameters
    w = w - learning_rate * loss_rate_of_change_w
    b = b - learning_rate * rate_of_change_b
    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')    
# After training, print final parameters
print(f'Final w: {w.item()}, b: {b.item()}')
# Reflect on the gradient computation process
"""
• Which parts of the gradient computation were straightforward?
• Which parts were confusing or error-prone?
• What assumptions from the book were implicit in your implementation?
"""


# Straightforward parts:
# - Defining the linear model and loss function using PyTorch operations.
# - Using finite differences to approximate gradients.
# Confusing or error-prone parts:
# - Ensuring the correct application of the finite difference formula.
# - Choosing an appropriate delta value for finite differences.
# Implicit assumptions from the book:
# - The linearity of the model and the differentiability of the loss function.

#question 4
print("\nquestion 4")

#reemplement model using nn.Linear
import torch.nn as nn

linear_model = nn.Linear(1, 1) # 1 input feature, 1 output feature

    # Reshape x to be a 2D tensor with one column
x_reshaped = x.view(-1, 1)
y_reshaped = y.view(-1, 1)

linear_loss = []
for epoch in range(1000):

    # Forward pass
    y_pred = linear_model(x_reshaped)
    
    # Compute loss
    loss = mean_squared_error(y_pred, y_reshaped)
    
    # Backward pass and optimization
    linear_model.zero_grad()  # Zero the gradients
    loss.backward()           # Backpropagate the loss
    with torch.no_grad():     # Update weights without tracking gradients
        for param in linear_model.parameters():
            param -= learning_rate * param.grad
    
    linear_loss.append(loss.item())
    
    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

#nspect the parameters created by nn.Linear and their associated gradients.
for name, param in linear_model.named_parameters():
    if param.requires_grad:
        print(f'Parameter: {name}, Value: {param.data}, Gradient: {param.grad}')

#nn.linear abstracts away the manual gradient computation and parameter updates, making the code cleaner and less error-prone. It automatically handles the creation of weights and biases, as well as their gradients during backpropagation.
#conceptually similar to the manual implementation, but with added convenience and efficiency.

#question 5
print("\nquestion 5")
#extend model by adding a hidden layer with a non-linear activation function (e.g., ReLU) between the input and output layers.
import torch.nn.functional as F
class ExtendedModel(nn.Module):
    def __init__(self):
        super(ExtendedModel, self).__init__()
        self.hidden = nn.Linear(1, 5)  # Hidden layer with 5 neurons
        self.output = nn.Linear(5, 1)   # Output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # Apply ReLU activation
        x = self.output(x)
        return x
extended_model = ExtendedModel()

# Reshape x to be a 2D tensor with one column
x_reshaped = x.view(-1, 1)
y_reshaped = y.view(-1, 1)

extend_loss = []
for epoch in range(1000):

    # Forward pass
    y_pred = extended_model(x_reshaped)
    
    # Compute loss
    loss = mean_squared_error(y_pred, y_reshaped)
    
    # Backward pass and optimization
    extended_model.zero_grad()  # Zero the gradients
    loss.backward()             # Backpropagate the loss
    with torch.no_grad():       # Update weights without tracking gradients
        for param in extended_model.parameters():
            param -= learning_rate * param.grad
    
    extend_loss.append(loss.item())
    
    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# observe change in loss curve 

plt.figure()
plt.plot(linear_loss, label="Linear model")
plt.plot(extend_loss, label="Linear + ReLU")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss curves comparison")
plt.show()

# change in gradient magnitudes
for name, param in extended_model.named_parameters():
    if param.requires_grad:
        print(f'Parameter: {name}, Value: {param.data}, Gradient: {param.grad}')
# The addition of a hidden layer with a non-linear activation function allows the model to capture more complex relationships in the data, potentially leading to better performance. The loss curve for the extended model may show a more significant decrease compared to the simple linear model, indicating improved learning. Gradient magnitudes may also vary, reflecting the increased complexity of the model.

#d 
# change the activation function to tanh
class TanhModel(nn.Module):
    def __init__(self):
        super(TanhModel, self).__init__()
        self.hidden = nn.Linear(1, 5)  # Hidden layer with 5 neurons
        self.output = nn.Linear(5, 1)   # Output layer

    def forward(self, x):
        x = torch.tanh(self.hidden(x))  # Apply tanh activation
        x = self.output(x)
        return x
tanh_model = TanhModel()
# Reshape x to be a 2D tensor with one column
x_reshaped = x.view(-1, 1)
y_reshaped = y.view(-1, 1)

tanh_loss = []
for epoch in range(1000):
    
    # Forward pass
    y_pred = tanh_model(x_reshaped)
    
    # Compute loss
    loss = mean_squared_error(y_pred, y_reshaped)
    
    # Backward pass and optimization
    tanh_model.zero_grad()  # Zero the gradients
    loss.backward()         # Backpropagate the loss
    with torch.no_grad():   # Update weights without tracking gradients
        for param in tanh_model.parameters():
            param -= learning_rate * param.grad
    
    tanh_loss.append(loss.item())
    
    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# add 2 hidden layers
class TwoHiddenLayersModel(nn.Module):
    def __init__(self):
        super(TwoHiddenLayersModel, self).__init__()
        self.hidden1 = nn.Linear(1, 5)  # First hidden layer with 5 neurons
        self.hidden2 = nn.Linear(5, 5)  # Second hidden layer with 5 neurons
        self.output = nn.Linear(5, 1)    # Output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # Apply ReLU activation to first hidden layer
        x = F.relu(self.hidden2(x))  # Apply ReLU activation to second hidden layer
        x = self.output(x)
        return x


two_hidden_model = TwoHiddenLayersModel()
# Reshape x to be a 2D tensor with one column
x_reshaped = x.view(-1, 1)
y_reshaped = y.view(-1, 1)
two_hidden_loss = []
for epoch in range(1000):
    
    # Forward pass
    y_pred = two_hidden_model(x_reshaped)
    
    # Compute loss
    loss = mean_squared_error(y_pred, y_reshaped)
    
    # Backward pass and optimization
    two_hidden_model.zero_grad()  # Zero the gradients
    loss.backward()               # Backpropagate the loss
    with torch.no_grad():         # Update weights without tracking gradients
        for param in two_hidden_model.parameters():
            param -= learning_rate * param.grad
    
    two_hidden_loss.append(loss.item())
    
    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        


# observe change in loss curve
plt.figure()
plt.plot(linear_loss, label="Linear model")
plt.plot(extend_loss, label="Linear + ReLU")
plt.plot(tanh_loss, label="Linear + Tanh")
plt.plot(two_hidden_loss, label="2 Hidden Layers + ReLU")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 500)
plt.legend()
plt.title("Loss curves comparison")
plt.show()
