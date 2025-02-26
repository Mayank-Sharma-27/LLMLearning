Before Diving deep into neural networks its a suggestion to revise linear algebra.

# Linear Algebra

## Vectors

A vector is a quantity that has both magnitude and direction. Vectors are crucial in neural networks as inputs, weights, and other parameters are often represented as vectors. Operations like dot products and matrix multiplications are fundamental in neural network computations

Resource : [Good Youtube Playlist](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

# Neural Networks Fundamentals

The most common neuron model used today is one called the sigmoid neuron, but before understanding that. I will try to understand what are perceptrons and what gaps that they had which lead to people learning sigmoid neurons?

Perceptrons were [developed](http://books.google.ca/books/about/Principles_of_neurodynamics.html?id=7FhRAAAAMAAJ) in the 1950s and 1960s by the scientist [Frank Rosenblatt](http://en.wikipedia.org/wiki/Frank_Rosenblatt), inspired by earlier [work](http://scholar.google.ca/scholar?cluster=4035975255085082870) by [Warren McCulloch](http://en.wikipedia.org/wiki/Warren_McCulloch) and [Walter Pitts](http://en.wikipedia.org/wiki/Walter_Pitts).

A perceptron takes a binary inputs x1,x2,… and produces a single binary output.

![perceptron.png](perceptron.png)

In the example shown the perceptron has three inputs, x1,x2,x3 In general it could have more or fewer inputs. The perceptron’s output is determined by whether the weighted sum of the inputs is greater than some threshold value.

x1+x2+x3 < threshold then the output will be 0 otherwise it will be 1. But in this example the weight of each input is 1.

### What do we mean by weight?

A way you can think about the perceptron is that it's a device that makes decisions by weighing up evidence. Let us understand this with an example:

Suppose you are on a weight loss journey and someone just put a box of sweets in front of you. Now you may want to decide whether of not to eat the sweets based on the following 3 factors:

1. Did complete your calorie goals for the day?
2. Does your favourite sweet is part of the sweets which are offered?
3. Are the sweets present low in calories?

We can represent these factors by corresponding binary variables x1,x2 and x3. For instance if you like the sweet we will have x1=1 if you don’t like it x1= . Similarly for others as well we will have x2=1 if your favourite sweet is present and x2=0 if its not there, similar with x3 as well.

Now suppose you are a person who absolutely loves sweets and you are happy to eat even if it has your favourite sweet present even if its not low calorie or you have completed the calorie limit for the day. One way to do this is choose a weight W2 as 6 and W1 and W3 as 2 respectively. The larger value of W2 indicates that the type of sweet matters a lot to you, much more than the other 2 weights. Finally suppose you choose the threshold as 5 for the perceptron network.

With these choices, the perceptron implements the desired decision-making model, outputting 1 whenever the sweet present is favourite, and 0 whenever its not. It makes no difference to the output whether you have hit the calorie limit or the sweets present are low calorie.

From this example we understand that the neuron’s output is determined by weighted sum

∑j wjxj is less than or greater than some _threshold value_. Just like the weights, the threshold is a real number which is a parameter of the neuron. To put it in more precise algebraic terms:

![function.png](function.png)

### Sigmoid Neurons

Let us now try to understand why we needed signmoid neurons. Suppose we have a network of perceptrons to solve a specific set of problem. Let us say we are uploading and image to read handwritten letters. We want the network to learn weights and biases so that the output from the network correctly classifies the letter. Suppose we want the network to understand the difference b/w 'd' and 'b',we observe that the network in unable to recognise this and we make a slight change in the weight so that instead of classifying the image as a 'b' it classifies the image as a 'd'. Similarly we would want to change the weight and biases in other cases as well.
![[perceptron_weight_change.png]](perceptron_weight_change.png)

The problem is that this isn't what happens when our network contains perceptrons. In fact, a small change in the weights or bias of any single perceptron in the network can sometimes cause the output of that perceptron to completely flip, say from 0 to 1. That flip may then cause the behaviour of the rest of the network to completely change in some very complicated way. So while your "b" might now be classified correctly, the behaviour of the network on all the other images is likely to have completely changed in some hard-to-control way. That makes it difficult to see how to gradually modify the weights and biases so that the network gets closer to the desired behaviour. Perhaps there's some clever way of getting around this problem. But it's not immediately obvious how we can get a network of perceptrons to learn.

We fix this by introducing a new type of neuron called Sigmoid neuron. They are similar to perceptrons but they can be modified in such as way that small changes in their weights and biases cause only a small change in their output.

![perceptron.png](perceptron.png)

In case of sigmoid neuron the inputs will we x1,x2,x3, but instead of being just 0 or 1, these inputs can also take any values between 0 and 1. So for instance, 0.138 is a valid input for a sigmoid neuron. Similar to perceptron, the sigmoid neuron has weights for each input, w1,w2,w3... and an overall bias b.

Unlike perceptrons, which output either 0 or 1, sigmoid neurons output a value between 0 and 1, calculated using the sigmoid function:
![[sigmoid_function.png]](sigmoid_function.png)

The output of the sigmoid function with weights would look like this :
![](https://miro.medium.com/v2/resize:fit:381/1*hsxr27HSqlBJcw9IARQZbQ.png)

Now **to visualize this function**, we can take some values of x and y and plot it to see what it looks like, for example in the below case, we are plotting (‘**wx + b**’) on the x-axis and ‘y’ value on the y-axis.
![[sigmoid_graph.png]](sigmoid_graph.png)

If ‘**wx + b**’ is 0, then the equation(**y**) is reduced to:

![[Screenshot 2024-10-06 at 2.13.01 PM.png]](Screenshot 2024-10-06 at 2.13.01 PM.png)

Let’s try some other value:

![[Screenshot 2024-10-06 at 2.14.14 PM.png]](Screenshot 2024-10-06 at 2.14.14 PM.png)

This example is taken from the blog : https://prvnk10.medium.com/sigmoid-neuron-ad0ec6f9a3e2

To understand the similarity to the perceptron model, suppose z≡w⋅x+bz≡w⋅x+b is a large positive number. Then e−z≈0 and so σ(z)≈1 In other words, when z=w⋅x+b is large and positive, the output from the sigmoid neuron is approximately 1, just as it would have been for a perceptron. Suppose on the other hand that z=w⋅x+b is very negative. Then e−z→∞ and σ(z)≈0. So when z=w⋅x+b is very negative, the behaviour of a sigmoid neuron also closely approximates a perceptron. It's only when w⋅x+b is of modest size that there's much deviation from the perceptron model

So the sigmoid function looks something like this:

![[sigmoid_function_graph.png]](sigmoid_function_graph.png)

### Neural Networks

![[neural_network_architecture.png]](neural_network_architecture.png)

In the neural network architecture, the leftmost layer is called the input layer and the rightmost is called the output layer. The middle layers are called hidden layers.

The design of the input and output layers in a network is often straightforward. For example, suppose we're trying to determine if a handwritten image depicts a "9" or not. A natural way to design the network is to encode the intensities of the image pixels into the input neurons. If the image is a 64 by 64 greyscale image, then we'd have 4,096=64×64 input neurons, with the intensities scaled appropriately between 0 and 1. The output layer will contain just a single neuron, with output values of less than 0.5 indicating "input image is not a 9", and values greater than 0.5 indicating "input image is a 9".

### Gradient Descent

To train the neural network we feed the neural network with a training labled data. For example if we feed in a input x and we expected an input y we will feed that the network. This is how we make sure that the network learns overtime. Suppose after training the data we feed in the input to the network and check what was the output given by the network. If the network gets this wrong we try to adjust the weight and biases again and try.

Let us understand this with a example, Suppose we have a model which predicts the price of house based on some input parameters such as pincode and size of the house. We train the model based on the training data and then feed an input to the model to predict the price. Suppose the prediction made by the model is that house will be sold by 1 million, but in reality the house was sold at the price of 1.5 million. This means that we need to adjust the weights and biases again and train the model. In technical terms we will say that the `cost function` is very large in this case.

#### Cost Function

![[cost-of-difference.png]](cost-of-difference.png)
In neural networks, a cost function measures the error between predicted values and actual values. A common example of a cost function is **Mean Squared Error (MSE)**, which is defined as the average of the squared differences between predicted and actual values

Gradient Descent is a first-order optimization algorithm used to minimize the cost function by iteratively moving in the direction of the steepest descent (i.e., the negative of the gradient). The step size is determined by the **learning rate**. Too large a learning rate may cause the model to overshoot, while too small may make the training slow

Read more here
https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/
Youtube videos : [1](https://www.youtube.com/watch?v=i62czvwDlsw&t=20s)
[2](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2)

### Back Propagation

Till now we have learnt how the architecture of neural network looks like where we have input layer, hidden layers and the output layer. The input is traversed through hidden layers with weight and biases to produce an output and this is called forward propagation. We saw in the above section that the final output might not be correct. In that case network needs to learn from its mistakes. Back Propagation is an algorithm used to train the neural networks applying the error correction principle.

![[backpropagation.png]](backpropagation.png)

Next Step before starting Zero to Hero Course : https://www.youtube.com/watch?v=VMj-3S1tku0&t=7334s is to learn about PyTorch

## Implementing Video 1 in the series

Some questions that I will keep on documenting

### What is Numpy?

NumPy is a Python library used for working with arrays.
It also has functions for working in domain of linear algebra, fourier transform, and matrices.

### What is a derivative?

The derivative is the slope of the tangent line of a curve. The derivative the slope of a specific line. For a linear function like y = mx + b, the derivative is just m. Every point on the curve has the same local slope.

For quadratic functions like y = x2 the derivative is 2x. So at any point the slope of the tangent line is 2x. At x = 0, at the bottom of the parabola the derivative is 2\*0 = 0. It's a flat constant line that touches the parabola at x = 0. At x = 1 the derivative is 2 times 1 or 2. At x = 2 the derivative is 2 times 2 = 4. At x = 3 the derivative is 2 times 3 = 6. So though points on y = x2 are always the square of the x value, the slope at that point is increasing by 2 times the value of x at that point.

The derivative is the instantaneous rate of a change. If you are traveling in a car the derivative of your distance with respect to time is your velocity. Miles per hour or kilometers per hour is your rate of change. The value of the speed on the speedometer is the rate of change right now. Your average rate of change is the total distance you traveled divided by the total time traveled over the period of interest.

In the code(Check the micrograd_from_scratch) I drew this graph:

![[short_graph.png]](short_graph.png)

changing the the grad would change the output, example grad in node d if changed will change the value of L similarly changing grad in f will change the value of L. For full graph check the video or the notebook. So to find out how much would L change with change of d we need to find dL/dD. In our code L = d\*f so calculating the derivative
![[derivative.png]](derivative.png)

Understanding this with an example

![[graph.png]](graph.png)

Let us say the value of L is -8 as in the above graph. suppose I change the value of f now,
![[code.png]](code.png)
First the value of L1 is -8, then I changed the value of f by 1, adding h where h =1. So how should the value of L be changed? dl/df which is d (L = d\* f). So it means gradient of f is 4, so when I increased the value of f by 1 the value of L became -4 (L2 = L.data) the change here is 4. I have printed the values to make it easy. L1 was -8 then we changed f by 1 the gradient of f was 4 so it increased the value by 4 which made it -4.

### How node C impacts L?

To find how C impacts L we need to do dL/dC, we find this using chain rule.
![[graph.png]](graph.png)
#### Chain Rule

The chain rule states that knowing the instantaneous rate of change of _z_ relative to _y_ and that of _y_ relative to _x_ allows one to calculate the instantaneous rate of change of _z_ relative to _x_ as the product of the two rates of change.

As put by [George F. Simmons](https://en.wikipedia.org/wiki/George_F._Simmons "George F. Simmons"): "If a car travels twice as fast as a bicycle and the bicycle is four times as fast as a walking man, then the car travels 2 × 4 = 8 times as fast as the man."[[1]](https://en.wikipedia.org/wiki/Chain_rule#cite_note-1)

The relationship between this example and the chain rule is as follows. Let z, y and x be the (variable) positions of the car, the bicycle, and the walking man, respectively. The rate of change of relative positions of the car and the bicycle is dzdy=2.![{\textstyle {\frac {dz}{dy}}=2.}](https://wikimedia.org/api/rest_v1/media/math/render/svg/22c0a6d7a5d2d56a62796839af72d517b6023910) Similarly, dydx=4.![{\textstyle {\frac {dy}{dx}}=4.}](https://wikimedia.org/api/rest_v1/media/math/render/svg/fe3bcfe19742de78e60ec58cc82380def6132fc0) So, the rate of change of the relative positions of the car and the walking man isdzdx=dzdy⋅dydx=2⋅4=8.![{\displaystyle {\frac {dz}{dx}}={\frac {dz}{dy}}\cdot {\frac {dy}{dx}}=2\cdot 4=8.}](https://wikimedia.org/api/rest_v1/media/math/render/svg/370629d3d6db86ea972c602a1e96a024ba912389)

Using chain rule the gradient comes out to be -2

![[chainrule.png]](chainrule.png)

**Next we find dL/dA**

dL/dA = (dL/dE)*(dE/dA)

## Manual backpropagation example #2: a neuronn.
In this example in the video he implements back propagation in actual neuron using a hyper bolic function. We will do a more complex funciton with weights and take it to a activation function.

But before that I want to capture one thing
### What is the difference between cost function and activation function?
![[neuron.png]](neuron.png)

- A cost function is a measure of error between what value your model predicts and what the value actually is. For example, say we wish to predict the value yi for data point xi.
- An activation function is a very important feature of an artificial neural network , they basically decide whether the neuron should be activated or not.

In [artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network), the **activation function** defines the output of that node given an input or set of inputs.

![[activation_function.png]](
(activation_function.png)

In the example in the video we have taken TanhH function.
Check backpropagation file now for how it worked. 
 The end neuron looks like this
 ![[backpropagation_manual.png]](backpropagation_manual.png)

### Doing similar operations with PyTorch
https://pytorch.org/
I will keep on adding things here as needed

A [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor") is a multi-dimensional matrix containing elements of a single data type.

## Implementing Video 2 in the series

This video is 5 part video.

In this we will recommend a name from a list of names.
We will learn to use tensor to manipulate 2d arrays.


```
chars = sorted(list(set(''.join(words))))
stoi = {s: i +1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j,i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i,j].item(), ha="center", va="top", color='gray')
plt.axis("off")
```

Using this code we are able to create a matrix of how many times a combination of 2 letters occurs. For example 
![[First_Row.png]](First_Row.png)

This row represents how many times a name starts with a letter, we will measure probability now and use [Torch.Multinomial library](https://pytorch.org/docs/stable/generated/torch.multinomial.html)  
Additional reads is 

https://pytorch.org/docs/stable/notes/broadcasting.html
Before continuing the lecture will learn basics of pytorch as I feel I will not be able to understand the future videos.

When I run this code

```
g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p
```

The output is always

```
tensor([0.7081, 0.3542, 0.1054])
```

The reason seems to be due to seed

### What is a Seed?

A **seed** is an integer used to initialize a random number generator. It determines the starting point for the sequence of random numbers that will be generated. By setting a specific seed value, you make the sequence of random numbers **predictable** and **reproducible**.

### Why Does the Output Remain the Same?

When you set a seed in PyTorch, like this:

```
g = torch.Generator().manual_seed(2147483647)
```

it means you are telling the generator `g` to start from a fixed point in its internal sequence of random numbers. Every time you run `torch.rand(3, generator=g)`, it will generate the same sequence of values starting from that point, producing the same output.

For instance:

`p = torch.rand(3, generator=g) print(p)  # Always gives tensor([0.7081, 0.3542, 0.1054]) with seed 2147483647`

If you change the seed to something else, like `12345`, you’ll get a different sequence of values. However, each time you use that seed (`12345`), it will generate the same sequence of random numbers associated with it.

### Why Use a Seed?

Setting a seed is especially useful in tasks that require reproducibility, like machine learning experiments. By using the same seed, you ensure that your results can be repeated, which is crucial for debugging, testing, and sharing results with others.

Another Important thing to learn is  Broadcasting
### Broadcasting

**Broadcasting** is a method that allows you to perform operations on tensors of different shapes. It automatically expands smaller tensors to match the shape of larger ones so that mathematical operations can be performed on them without needing to explicitly reshape or copy data.

### How Broadcasting Works

When performing an operation (e.g., addition, multiplication) on two tensors with different shapes, PyTorch checks if broadcasting can make the shapes compatible by following a set of rules:

1. **Match from the Right**: Starting from the rightmost dimension, compare the shapes of each dimension.
2. **Rule of Size 1**: If a dimension of one tensor has size `1`, it can be stretched (broadcasted) to match the size of the other tensor in that dimension.
3. **Equal Dimensions**: If two dimensions are equal, they are compatible.
4. **Mismatched Sizes**: If neither of the dimensions is `1` or if the dimensions don’t match, broadcasting is not possible, and an error will be raised.
#### Example of Broadcasting

```
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # Shape: (2, 3)

B = torch.tensor([10, 20, 30])  # Shape: (3,)

```

In this case:

- `A` has shape `(2, 3)`, and `B` has shape `(3,)`.
- PyTorch will attempt to apply broadcasting so that `B` can match the shape of `A` along the leftmost dimension.
When we do `A + B`, the result will be:


```
A + B # Shape (2, 3) # tensor([[11, 22, 33], # [14, 25, 36]])
```
#### Steps in Broadcasting for This Example

1. **Expand B**: Since `B` has shape `(3,)`, it is automatically expanded to `(1, 3)`.
2. **Repeat Along Rows**: To match the shape of `A`, `B` is then repeated along the first dimension, resulting in an effective shape of `(2, 3)` for the operation.
3. **Perform the Element-wise Operation**: Now that `A` and the expanded `B` both have shape `(2, 3)`, PyTorch can add them element-wise.
#### Benefits of Broadcasting

- **Memory Efficiency**: Broadcasting avoids creating copies of data unnecessarily.
- **Ease of Use**: It simplifies operations on tensors of different shapes by automatically adjusting dimensions.
#### Summary

Broadcasting is a powerful technique that enables element-wise operations on tensors of different shapes by automatically adjusting dimensions, following specific rules. It’s widely used in PyTorch and other libraries for flexible and memory-efficient computations.

### Why  P /= P.sum(1) and P = N.float() P /= P.sum(1, keepdims= True) returns different values

I added this question in chatgpt and it gave a good example response
Sure! Let’s go through the process with an example using an actual \(2 \times 2\) tensor. We’ll examine the behavior of `P.sum(1)` with and without `keepdims=True` to see how it affects the results.


#### Example Tensor
Let's create a tensor `P`:
```python
P = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
```

#### Case 1: `P.sum(1)` without `keepdims=True`

```python
P_sum = P.sum(1)
```

Here’s what happens:
1. `P.sum(1)` calculates the sum across each row (dimension 1), so:
   - For the first row: \(1.0 + 2.0 = 3.0\)
   - For the second row: \(3.0 + 4.0 = 7.0\)
   
   This results in a tensor of shape `[2]` (1 value per row):
   ```python
   P_sum = torch.tensor([3.0, 7.0])  # Shape: [2]
   ```

2. **Dividing without `keepdims=True`**:
   When you do:
   ```python
   P /= P_sum
   ```
   Broadcasting applies, so `P_sum` (shape `[2]`) will be applied across each row in `P`. However, because it lacks a second dimension, the division operation will implicitly expand `P_sum` to match each row, which can lead to an unexpected result:

   ```python
   P = P / P_sum  # Results in:
   tensor([[0.3333, 0.6667],
           [0.4286, 0.5714]])
   ```
   This division works row-wise, but it divides each row by a single scalar, leading to a different normalization effect.

#### Case 2: `P.sum(1, keepdims=True)`

```python
P_sum_keepdims = P.sum(1, keepdims=True)
```

With `keepdims=True`, the sum retains the original shape along dimension `1`:
1. `P.sum(1, keepdims=True)` will result in:
   ```python
   P_sum_keepdims = torch.tensor([[3.0],
                                  [7.0]])  # Shape: [2, 1]
   ```

2. **Dividing with `keepdims=True`**:
   Now, when you divide by `P_sum_keepdims`:
   ```python
   P = P / P_sum_keepdims
   ```
   Broadcasting can align each row individually because of the additional singleton dimension in `P_sum_keepdims`, giving you:
   ```python
   tensor([[0.3333, 0.6667],
           [0.4286, 0.5714]])
   ```

In this case, each element is divided by the sum of its respective row, as expected, without any unintended broadcasting effects. 

#### Why `keepdims=True` Is Important
Using `keepdims=True` retains the original tensor structure, making sure that each row is normalized individually without mismatched dimensions.

#### Another question I asked is What is P.sum(1) what does 1 mean here?
This is the response

In `P.sum(1)`, the `1` is the **dimension along which the sum is computed**. In PyTorch (and many other tensor libraries), dimensions are indexed starting from 0, so:

- `0` refers to the **rows** (first dimension).
- `1` refers to the **columns** (second dimension).

#### Explanation with Example

Suppose we have this tensor `P`:

```python
P = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
```

Now, if we do:

1. **`P.sum(0)`**: Sums along **dimension 0** (across rows).
   - This will sum each column, resulting in a tensor with one element per column:
   ```python
   P.sum(0)  # Output: tensor([5.0, 7.0, 9.0])
   ```
   - Explanation:
     - Column 1: \(1.0 + 4.0 = 5.0\)
     - Column 2: \(2.0 + 5.0 = 7.0\)
     - Column 3: \(3.0 + 6.0 = 9.0\)

2. **`P.sum(1)`**: Sums along **dimension 1** (across columns).
   - This will sum each row, resulting in a tensor with one element per row:
   ```python
   P.sum(1)  # Output: tensor([6.0, 15.0])
   ```
   - Explanation:
     - Row 1: \(1.0 + 2.0 + 3.0 = 6.0\)
     - Row 2: \(4.0 + 5.0 + 6.0 = 15.0\)

#### Summary
- **`0`** in `sum(0)` means summing across rows (getting a result per column).
- **`1`** in `sum(1)` means summing across columns (getting a result per row).

### Training Loss
We now calculate the training loss of the function to know how good our model is. Likelihood is the probability of the entire dataset of entire model. Since the product of the probability was very low so for convenience we work with log likelihood. **

We changed this:
```
P = (N).float()
P /= P.sum(1, keepdims=True)
```

to

```
P = (N+1).float() # We did N+1 to do model smoothing
P /= P.sum(1, keepdims=True)
```

In the video its mentioned that 

"It does not make sense for the input layer to have integer values" because of that he did the encoding of integers using [One Hot Encoding](https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html)

**I also asked chatGpt why should we do ecoding this is the answer:**

In a neural network, it's generally not recommended for the input layer to take on raw integer values because neural networks typically perform better with continuous, normalized, or scaled data. Here’s why:

1. **Gradient-Based Learning:** Neural networks use gradient-based optimization (like gradient descent) to minimize error. Continuous, normalized inputs help ensure smooth gradients, making optimization more effective. Integer values, especially if large, can lead to uneven or steep gradients, which can hinder convergence.

2. **Range and Scale Variability:** Integer inputs can vary widely in scale. For example, if some features are in the range of 0-10 while others are in thousands, the network might struggle to learn meaningful patterns across these disparities. Normalization (e.g., scaling inputs to a 0–1 range) mitigates this problem, allowing each feature to contribute proportionately to learning.

3. **Activation Function Sensitivity:** Activation functions (like sigmoid, tanh, or ReLU) are sensitive to input ranges. Large integer values might push neurons to extreme activation (saturation), making it harder for the network to learn effectively. By using continuous, normalized inputs, the network can make better use of the full range of activation functions, facilitating more effective learning.

4. **Interpretability of Features:** Continuous data often represents meaningful relationships better in many real-world applications. If integer values encode discrete categories, they are often better represented through one-hot encoding, while numeric values (like age or salary) should be normalized.

In general, preprocessing inputs to be continuous, normalized values rather than raw integers helps neural networks learn more effectively and converge faster.

## Building a AI Agent
While I continue to learn the tutorials, I will start building an AI Agent.
I know prompts so I am good on that part, I need to learn building a RAG pipeline.



| Resources                                                  |                                                                                                                  |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Medium Blog                                                | [https://prvnk10.medium.com/sigmoid-neuron-ad0ec6f9a3e2](https://prvnk10.medium.com/sigmoid-neuron-ad0ec6f9a3e2) |
| Website                                                    | http://neuralnetworksanddeeplearning.com/                                                                        |
| Youtube                                                    | https://www.youtube.com/@3blue1brown                                                                             |
| [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) | [PlayList](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)                  |
| My Rag Application                                         | https://github.com/Mayank-Sharma-27/MySkinBuddy                                                                  |
| Book I am learning                                         | https://github.com/Mayank-Sharma-27/udlbook                                                                      |
