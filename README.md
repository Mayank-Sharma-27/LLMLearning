# Neural Networks Fundamentals

The most common neuron model userd today is one called the sigmoid neuron, but before understanding that. I will try to understand  what are perceptrons and what gaps that they had which lead to people learning sigmoid neurons?

Perceptrons were [developed](http://books.google.ca/books/about/Principles_of_neurodynamics.html?id=7FhRAAAAMAAJ) in the 1950s and 1960s by the scientist [Frank Rosenblatt](http://en.wikipedia.org/wiki/Frank_Rosenblatt), inspired by earlier [work](http://scholar.google.ca/scholar?cluster=4035975255085082870) by [Warren McCulloch](http://en.wikipedia.org/wiki/Warren_McCulloch) and [Walter Pitts](http://en.wikipedia.org/wiki/Walter_Pitts). T

A perceptron takes a binary inputs x1,x2,… and produces a single binary output.

![4 steps (1).png](Neural%20Networks%20Fundamentals%20116a06ba1bc880368378cc368c6f0eb7/4_steps_(1).png)

In the example shown the perceptron has three inputs, x1,x2,x3 In general it could have more or fewer inputs. The neuron’s output is determined by weather the sum of the inputs is greater then sun threshold value.

x1+x2+x3 < threshold then the output will be 0 otherwise it will be 1. But in this example the weight of each input is 1. 

### What do we mean by weight?

 A way you can think about the perceptron is that it's a device that makes decisions by weighing up evidence. Let us understand this with an example:

Suppose you are on a weight loss journey and someone just put a box of sweets in front of you. Now you may want to decide weather of not to eat the sweets based on the following 3 factors:

1. Did complete your calorie goals for the day?
2. Does your favourite sweet is part of the sweets which are offered?
3. Are the sweets present low in calorie?

We can represent these factors by corresponding binary variables x1,x2 and x3. For instance if you like the sweet we will have x1=1 if you don’t like it x1= . Similarly for others as well we will have x2=1 if your favourite sweet is present and x2=0 if its not there, similar with x3 as well.

Now suppose you are a person who absolutely loves sweets and you are happy to eat even if it has your favourite sweet present even if its not low calorie or you have completed the calorie limit for the day. One way to do this is choose a weight W2 as 6 and W1 and W3 as 2 respectively. The larger value of W2 indicates that the type of sweet matters a lot to you, much more than the other 2 weights. Finally suppose you choose the threshold as 5 for the perceptron network. 

With these choices, the perceptron implements the desired decision-making model, outputting 1 whenever the sweet present is favourite, and 0 whenever its not. It makes no difference to the output whether you have hit the calorie limit or the sweets present are low calorie.

From this example we understand that the neuron’s output is determined by weighted sum 

∑j wjxj is less than or greater than some *threshold value*. Just like the weights, the threshold is a real number which is a parameter of the neuron. To put it in more precise algebraic terms:

![Screenshot 2024-10-05 at 11.03.53 AM.png](Neural%20Networks%20Fundamentals%20116a06ba1bc880368378cc368c6f0eb7/Screenshot_2024-10-05_at_11.03.53_AM.png)
