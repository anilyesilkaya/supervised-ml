
<a id="T_93F7051D"></a>

# Supervised Learning \- Week 2 / Part 1
<!-- Begin Toc -->

## Table of Contents
&emsp;[Vectorization](#H_9A2868AA)
 
&emsp;[Linear Regression with Multiple Variables](#H_764ED9B1)
 
&emsp;&emsp;[An Alternative Way to Gradient Descent](#H_B8D86F04)
 
&emsp;&emsp;[Implementation of the Gradient Descent with Multiple Variables](#H_70A466BC)
 
&emsp;&emsp;&emsp;[Problem Statement](#H_7E181C33)
 
&emsp;[Gradient Descent in Practice](#H_3DD18DCA)
 
&emsp;&emsp;[**Feature Scaling in Practice**](#H_B4802151)
 
&emsp;&emsp;[Learning Rate in Practice](#H_09652588)
 
&emsp;&emsp;&emsp;[Case 1: alpha = 9.9e-7](#H_5114F050)
 
&emsp;&emsp;&emsp;[Case 2: alpha = 9e-7](#H_A3D29B82)
 
&emsp;&emsp;&emsp;[Case 2: alpha = 1e-7](#H_CBCC986D)
 
&emsp;[Local Function Definitions](#H_B2D384DC)
 
<!-- End Toc -->

In this part we will learn how to make linear regression algorithm much faster and powerful.

-  Suppose we are interested in linear regression and we are looking for not only one feature but multiple features. 
-  In the previous part we saw that our dataset had one feature; $x$, size of the house, and we were predicting $y$; price of the house by using the linear regression formula: $y=wx+b$. 
-   What if we not only know the size of the house ( $x_1$ ) but also the number of bedrooms ( $x_2$ ), number of floors ( $x_3$ ) and age of the house in years ( $x_4$ )? 
```matlab
clear all;
close all;
clc;

% Table
hSize = [2104; 1416; 1534; 852];
hNumBedrooms = [5; 3; 3; 2];
hNumFloors = [1; 2; 2; 1];
hAge = [45; 40; 30; 36];
hPrice = [460; 232; 315; 178];
table(hSize,hNumBedrooms,hNumFloors,hAge,hPrice)
```


| |hSize|hNumBedrooms|hNumFloors|hAge|hPrice|
|:--:|:--:|:--:|:--:|:--:|:--:|
|1|2104|5|1|45|460|
|2|1416|3|2|40|232|
|3|1534|3|2|30|315|
|4|852|2|1|36|178|


-  Note that we represent each feature ( $j^{\textrm{th}}$ column in the table) with subscript $j$, where ${\mathbf{x}}_j =[x_j^{(1)} \,x_j^{(2)} \,\cdots \,x_j^{(m)} ]^{\textrm{T}}$ $j^{\textrm{th}}$ feature, for $j\in \lbrace 1,2,\cdots ,n\rbrace$. 
-  We represent the number of features by using $n$, where $n=4$ for the table above. 
-  Moreover, ${\mathbf{x}}^{(i)} =[x_1 \,x_2 \,\cdots \,x_n ]$ represents the features of $i^{\textrm{th}}$ training example ( $i^{\textrm{th}}$ row in the table), where $i\in \lbrace 1,2,\cdots ,m\rbrace$. The parameter $m$ represents the number of examples. 
-  Consequently, we can represent the value of $j^{\textrm{th}}$ feature (column) in $i^{\textrm{th}}$ training example (row) by using the notation $x_j^{(i)}$. 

Now, we have multiple features, let's look at our how our model look like

-  **Previously:** $f_{w,b} (x)=wx+b$ 
-  **If we generalize the one feature case:** $\begin{array}{l} f_{w,b} (\mathbf{x})=w_1 x_1 +w_2 x_2 +\cdots +w_n x_n +b\newline =\left(\sum_{i=1}^n w_i x_i \right)+b={\mathbf{w}}^{\textrm{T}} \mathbf{x}+b \end{array}$ 

where this expression is called multi parameter linear regression. More importantly, here, both $\mathbf{w}$ and $b$ are the parameters.


Example: $f_{w,b} (\mathbf{x})=\textrm{Price}=0.1\cdot \textrm{size}+4\cdot \textrm{numBedrooms}+10\cdot \textrm{numFloors}-2\cdot \textrm{years}+\textrm{basePrice}$ 

-  Base price of a house starts at \$80 000 assuming it has no size, no bedrooms, no floors and no age. 
-  For every addtiional squared foot, the price increases \$100. 
-  For every addition of bedrooms, the price increases \$4000. 
-  For every addition of floors, the price increases \$10 000. 
-  For every additional year in houses age, the price reduces \$2000. 
<a id="H_9A2868AA"></a>

# Vectorization

We can define a case where $n=3$ 

 $$ \mathbf{w}=[w_1 \,\,w_2 \,\,w_3 ]^{\textrm{T}} $$ 

 $$ \mathbf{x}=[x_1 \,\,x_2 \,\,x_3 ]^{\textrm{T}} $$ 

 $b$ is a number


We can calculate the linear regression expression without vectorization as follows:

```
f = w(1)*x(1) + w(2)*x(2) + w(3)*x(3) + b;
```

If the number of features is not 3 but 100 000, this would be inefficient to code as well as inefficient for the computation. An alternative approach without also vectorization becomes,

```
n = 3;
f = 0;
for i = 1:n
    f = F + w(i)*x(i)
end
f = f + b;
```

With vectorization,

```
f = dot(w,x) + b;
```

or

```
% If both w and x are row vectors
f = w*x' + b;

% If both w and x are column vectors
f = w'*x + b;
```
<a id="H_B0C8ACD4"></a>

**Let's evaluate the benefits of vectorization**

```matlab
w = randn(1e8,1);
x = randn(1e8,1);

% Non-vectoral implementation
tStart = tic;
f = 0;
for i = 1:length(x)
    f = f + w(i)*x(i);
end
tExec = toc(tStart);
disp(['Non-vectoral execution time: ', num2str(tExec),' seconds.'])
```

```matlabTextOutput
Non-vectoral execution time: 0.17716 seconds.
```

```matlab
% Vectoral implementation
tStart = tic;
f = w.'*x;
tExec = toc(tStart);
disp(['Vectoral implementation execution time: ', num2str(tExec),' seconds.'])
```

```matlabTextOutput
Vectoral implementation execution time: 0.023895 seconds.
```

```matlab
clear w x % release memory
```
<a id="H_764ED9B1"></a>

# Linear Regression with Multiple Variables

------------------------------


 **Single Varible (Scalar) Notation**

-  **Inputs:** $x^{(i)}$ 
-  **Labels:** $y^{(i)}$ 
-  **Parameters:** $w$ and $b$ 
-  **Model:** $f\_{w,b} (x^{(i)} )=wx^{(i)} +b$ 
-  **Cost Function:** $J(w,b)=\frac{1}{2m}\sum\_{i=1}^m {\left(f\_{w,b} (x^{(i)} )-y^{(i)} \right)}^2 =\frac{1}{2m}\sum\_{i=1}^m {\left(wx^{(i)} +b-y^{(i)} \right)}^2$ 
-  **Gradient Descent:** $w=w-\alpha \frac{\textrm{d}J(w,b)}{\textrm{d}w}$ and $b=b-\alpha \frac{\textrm{d}J(w,b)}{\textrm{d}b}$ 

------------------------------


 **Multiple Variables (Vector) Notation**

-  **Inputs:** ${\mathbf{x}}^{(i)} =\[x\_1 \,x\_2 \,\cdots \,x\_n \]$ represents the $i^{\textrm{th}}$ row of $\mathbf{X}$, where $\mathbf{X}$ is the $m\times n$ input data matrix. 
-  **Labels:** $y^{(i)}$ 
-  **Parameters:** $\mathbf{w}=\[w\_1 \,w\_2 \,\cdots w\_n \]^{\textrm{T}}$ and $b$ 
-  **Model:** $f\_{\mathbf{w},b} ({\mathbf{x}}^{(i)} )={\mathbf{x}}^{(i)} \mathbf{w}+b=\left(\sum\_{j=1}^n w\_j x\_j^{(i)} \right)+b$, for $i\in \lbrace 1,2,\cdots ,m\rbrace$ or $f\_{\mathbf{w},b} (\mathbf{X})=\mathbf{X}\mathbf{w}+b$ 
-  **Cost Function:** $J(\mathbf{w},b)=\frac{1}{2m}\sum\_{i=1}^m {\left(\left(\sum\_{j=1}^n w\_j x\_j^{(i)} \right)+b-y^{(i)} \right)}^2$ or $J(\mathbf{w},b)=\frac{1}{2m}\sum\_{i=1}^m {\left({\mathbf{x}}^{(i)} \mathbf{w}+b-y^{(i)} \right)}^2$ 
-  **Gradient Descent:** $w\_j =w\_j -\alpha \frac{\partial }{\partial w\_j }J(\mathbf{w},b)$, for $j\in \lbrace 1,2,\cdots ,n\rbrace$ and $b=b-\alpha \frac{\partial }{\partial b}J(\mathbf{w},b)$ 

------------------------------


How can we minimize the cost function in multiple variable case? 

-  We can calculate the gradient of the cost function w.r.t. $\mathbf{w}$ as follows: 

 $\nabla J(\mathbf{w},b)=\left\lbrack \begin{array}{c} \frac{\partial J(\mathbf{w},b)}{\partial w\_0 }\newline \frac{\partial J(\mathbf{w},b)}{\partial w\_1 }\newline \vdots \newline \frac{\partial J(\mathbf{w},b)}{\partial w\_n } \end{array}\right\rbrack$, where the parial derivative $\frac{\partial }{\partial w\_j }J(\mathbf{w},b)$ for $j\in \lbrace 0,1,2,\cdots ,n\rbrace$ could be calculated as follows:


1) $\frac{\partial }{\partial w\_j }J(\mathbf{w},b)=\frac{\partial }{\partial w\_j }\left(\frac{1}{2m}\sum\_{i=1}^m {\left({\mathbf{x}}^{(i)} \mathbf{w}+b-y^{(i)} \right)}^2 \right)=\frac{1}{2m}\sum\_{i=1}^m \frac{\partial }{\partial w\_j }{\left(\left(\sum\_{j=1}^n w\_j x\_j^{(i)} \right)+b-y^{(i)} \right)}^2$ 


2) By using the chain and sum rules,

 $$ \begin{array}{l} \frac{\partial }{\partial w_j }J(\mathbf{w},b)=\frac{1}{m}\sum_{i=1}^m \left(\left(\sum_{j=1}^n w_j x_j^{(i)} \right)+b-y^{(i)} \right)\left(\frac{\partial }{\partial w_j }\sum_{j=1}^n w_j x_j^{(i)} \right)\newline =\frac{1}{m}\sum_{i=1}^m \left(\left(\sum_{j=1}^n w_j x_j^{(i)} \right)+b-y^{(i)} \right)\left(\frac{\partial }{\partial w_j }\left(w_0 x_0^{(i)} +w_1 x_1^{(i)} +\cdots +w_n x_n^{(i)} \right)\right) \end{array} $$ 

3) We can see from the above expression that 


 $\frac{\partial }{\partial w\_j }\left(w\_0 x\_0^{(i)} +w\_1 x\_1^{(i)} +\cdots +w\_n x\_n^{(i)} \right)=x\_j^{(i)}$ for $j\in \lbrace 0,1,2,\cdots ,n\rbrace$.


4) Therefore,

 $$ \frac{\partial }{\partial w_j }J(\mathbf{w},b)=\frac{1}{m}\sum_{i=1}^m \left(w_1 x_1^{(i)} +w_w x_w^{(i)} +\cdots w_n x_n^{(i)} +b-y^{(i)} \right)\cdot x_j^{(i)} =\frac{1}{m}\sum_{i=1}^m \left({\mathbf{x}}^{(i)} \mathbf{w}+b-y^{(i)} \right)\cdot x_j^{(i)} $$ 

Similarly, the partial derivative of the cost function w.r.t. the bias becomes,


5) $\frac{\partial }{\partial b}J(\mathbf{w},b)=\frac{\partial }{\partial b}\left(\frac{1}{2m}\sum\_{i=1}^m {\left(f\_{\mathbf{w},b} ({\mathbf{x}}^{(i)} )-y^{(i)} \right)}^2 \right)=\frac{1}{2m}\sum\_{i=1}^m \frac{\partial }{\partial b}{\left(\left(\sum\_{j=1}^n w\_j x\_j^{(i)} \right)+b-y^{(i)} \right)}^2$ 


6) If we solve the above expression using (2)

 $$ \frac{\partial }{\partial b}J(\mathbf{w},b)=\frac{1}{m}\sum_{i=1}^m \left(w_1 x_1^{(i)} +w_2 x_2^{(i)} +\cdots w_n x_n^{(i)} +b-y^{(i)} \right)\cdot \frac{\partial }{\partial b}\left(w_1 x_1^{(i)} +w_2 x_2^{(i)} +\cdots w_n x_n^{(i)} +b-y^{(i)} \right) $$ 

7) As we know that $\frac{\partial }{\partial b}\left(w\_1 x\_1^{(i)} +w\_2 x\_2^{(i)} +\cdots w\_n x\_n^{(i)} +b-y^{(i)} \right)=1$. Hence,

 $$ \frac{\partial }{\partial b}J(\mathbf{w},b)=\frac{1}{m}\sum_{i=1}^m \left(w_1 x_1^{(i)} +w_2 x_2^{(i)} +\cdots w_n x_n^{(i)} +b-y^{(i)} \right)=\frac{1}{m}\sum_{i=1}^m \left({\mathbf{x}}^{(i)} \mathbf{w}+b-y^{(i)} \right) $$ 

------------------------------


**Consequently, the policy for updating the parameters in BGD becomes:**

 $$ w_j =w_j -\alpha \frac{\partial }{\partial w_j }J(\mathbf{w},b)=w_j -\alpha \left(\frac{1}{m}\sum_{i=1}^m \left({\mathbf{x}}^{(i)} \mathbf{w}+b-y^{(i)} \right)\cdot x_j^{(i)} \right)~~(1) $$ 

 $$ b=b-\alpha \frac{\partial }{\partial b}J(\mathbf{w},b)=b-\alpha \left(\frac{1}{m}\sum_{i=1}^m \left({\mathbf{x}}^{(i)} \mathbf{w}\right)+b-y^{(i)} \right)~~(2) $$ 

for $i\in \lbrace 1,2,\cdots ,m\rbrace$ and $j\in \lbrace 1,2,\cdots ,n\rbrace$.

<a id="H_B8D86F04"></a>

## An Alternative Way to Gradient Descent

This method is called "Normal Equation".


**Advantages**

-  It solves the linear regression problem for the parameters $w$ and $b$ without iterations. 

**Disadvantages**

-  Only works for linear regression and can't be generalized to other learning algortihms logistic regression, neural networks etc. 
-  Slow when number of features is large ( > 10, 000 features) 

**What you need to know**

-  Normal equation method may be used in machine learning libraries that implement linear regression. 
-  Gradient descent is the recommended method for finding parameters $w$ and $b$. 
<a id="H_70A466BC"></a>

## Implementation of the Gradient Descent with Multiple Variables

Similar to the gradient descent implementation with single feature we will have two steps:

-  Step 1: Initialize the $\mathbf{w}=\[w\_1 \,w\_2 \,\cdots \,w\_n \]^{\textrm{T}}$ and $b$. 
-  Step 2: Repeat (1) and (2) simultaneously until the convergence (or the stopping criteria is met). 

We can also represent the above steps mathematically as follows:


1) Randomly initialize $\mathbf{w}$ and $b$.


2) Repeat until the stopping criteria is met:

-  $\displaystyle w\_j =w\_j -\frac{\alpha }{m}\sum\_{i=1}^m \left({\mathbf{x}}^{(i)} \mathbf{w}+b-y^{(i)} \right)\cdot x\_j^{(i)}$ 
-  $\displaystyle b=b-\frac{\alpha }{m}\sum\_{i=1}^m \left({\mathbf{x}}^{(i)} \mathbf{w}+b-y^{(i)} \right)$ 

where $i\in \lbrace 1,2,\cdots ,m\rbrace$ and $j\in \lbrace 1,2,\cdots ,n\rbrace$. Note that $\mathbf{w}$ and $b$  must be updated simultaneously. Note that

-  ${\mathbf{x}}^{(i)} =\[x\_1 \,x\_2 \,\cdots \,x\_n \]$ represents $1\times n$ vector, the $i^{\textrm{th}}$ row of $\mathbf{X}$, where $\mathbf{X}$ is the $m\times n$ input data matrix. 
-  $\mathbf{w}=\[w\_1 \,w\_2 \,\cdots w\_n \]^{\textrm{T}}$ is the $n\times 1$ weights vector. 
<a id="H_7E181C33"></a>

### Problem Statement

We will use the example of housing price prediction. The training dataset contains three examples with four features (size, bedrooms, floors and age). Please note that the size is given in sqft rather than 1000 sqft, which will cause issues and we will solve this in the following section!


Let's present the data in a table format

```matlab
% Housing Price Dataset
houseSize = [2104; 1416; 852];
numBedrooms = [5; 3; 2];
numFloors = [1; 2; 1];
houseAge = [45; 40; 35];
housePrice = [460; 232; 178];
table(houseSize,numBedrooms,numFloors,houseAge,housePrice)
```


| |houseSize|numBedrooms|numFloors|houseAge|housePrice|
|:--:|:--:|:--:|:--:|:--:|:--:|
|1|2104|5|1|45|460|
|2|1416|3|2|40|232|
|3|852|2|1|35|178|



**Objective:** You will build a linear regression algorithm so that you can predict the price for other houses. For instance, a house with 1200 sqft, 3 bedrooms, 1 floor and 40 years old.


Let's convert the information contained in the table into a matrix. Note that the parameter "price" is what we are trying to estimate/predict. Therefore,

-  X_train: Training example matrix ( $m\times n$ ), where the matrix contains $n$ features and $m$ examples. 
-  y_train: Training example targets ( $m\times 1$ ) 
```matlab
X_train = [houseSize numBedrooms numFloors houseAge]
```

```matlabTextOutput
X_train = 3x4
        2104           5           1          45
        1416           3           2          40
         852           2           1          35

```

```matlab
y_train = housePrice
```

```matlabTextOutput
y_train = 3x1
   460
   232
   178

```

```matlab
m = size(X_train,1)
```

```matlabTextOutput
m = 3
```

```matlab
n = size(X_train,2)
```

```matlabTextOutput
n = 4
```


Since our goal is to find the linear relationship ( $\mathbf{w}$ and $b$ ), the linear regression model becomes,


 $f\_{\mathbf{x},b} ({\mathbf{x}}^{(i)} )={\mathbf{x}}^{(i)} \mathbf{w}+b$ for $i\in \lbrace 1,2,\cdots ,m\rbrace$, where $\mathbf{w}$ and ${\mathbf{x}}^{(i)}$ are the $n\times 1$ column vector that consists of the linear weights and $1\times n$ $i^{\textrm{th}}$ row vector of the training matrix (X_train), respectively. The training matrix $\mathbf{X}$ is also given as follows.

 $$ \mathbf{X}=\left\lbrack \begin{array}{cccc} x_1^{(1)}  & x_2^{(1)}  & \cdots  & x_n^{(1)} \newline x_1^{(2)}  & x_2^{(2)}  & \cdots  & x_n^{(2)} \newline  &  & \ddots  & \newline x_1^{(m)}  & x_2^{(m)}  & \cdots  & x_n^{(m)}  \end{array}\right\rbrack $$ 

Let's implement the batch gradient descent (BGD) algorithm for multiple variables.


**Notation**

||||
| :-- | :-- | :-- |
| **General Notation**  | **Description/Style**  | **Representaion in MATLAB**   |
| $\displaystyle a$  | scalar - non bold  |   |
| $\displaystyle \mathbf{a}$  | vector - bold  |   |
| $\displaystyle \mathbf{A}$  | matrix - bold capital  |   |
| Regression Specific  |  |   |
| $\displaystyle m$  | number of examples/datapoints  | `m`   |
| $\displaystyle n$  | number of features  | `n`   |
| $\displaystyle \mathbf{X}$  | $m\times n$ training example matrix  | `X_train`   |
| $\displaystyle \mathbf{y}$  | training example targets  | y_train   |
| $\displaystyle {\mathbf{x}}^{(i)}$  | $i^{\textrm{th}}$ row vector of $\mathbf{X}$, which corresponds <br> to a data point with $j$ features, where $j\in \lbrace 1,2,\cdots ,n\rbrace$  | `X_train(i,:)`   |
| $\displaystyle {\mathbf{x}}\_j$  | $j^{\textrm{th}}$ column vector of $\mathbf{X}$, which corresponds to all data points for a given <br> feature  | `X_train(:,j)`   |
| $\displaystyle y^{(i)}$  | $i^{\textrm{th}}$ element of the targets vector  | `y_train(i)`   |
| $\displaystyle \mathbf{w}$  | parameter - $n\times 1$ weights vector  | `w`   |
| $\displaystyle b$  | parameter - bias  | `b`   |
| $\displaystyle f\_{\mathbf{w},b} ({\mathbf{x}}^{(i)} )$  | Model evaluation at the point ${\mathbf{x}}^{(i)}$ parameterized by $\mathbf{w}$ and $b$. <br> $\displaystyle f\_{\mathbf{w},b} ({\mathbf{x}}^{(i)} )={\mathbf{x}}^{(i)} \mathbf{w}+b$  | `X_train(i,:)*w+b`   |
| $\displaystyle \frac{\partial J(\mathbf{w},b)}{\partial w\_j }$  | Gradient/partial derivative of the cost function w.r.t the $j^{\textrm{th}}$ weight $w\_j$   | `dJdw`   |
| $\displaystyle \frac{\partial J(\mathbf{w},b)}{\partial b}$  | Gradient/partial derivative of the cost function w.r.t the bias $b$  | `dJdb`   |


The steps for BGD for multiple varibles are given by,

1.  Initialize the parameters; n-by-1 $\mathbf{w}$ vector and scalar $b$ (zero initialization is generally simple and efficient enough at this stage).
2. Update both $\mathbf{w}$ and $b$ simultaneously.
3. Repeat (2) until the BGD algorithm until either maximum number of iterations is reached or the algorithm yields a cost value, which is smaller than a user selected threshold and doesn't change anymore (algorithm converges).
```matlab
% BGD settings
alpha = 5.0e-7; % Learning rate
condStop = 1e-3; % Stopping condition
maxIter = 1000; % Maximum number of iterations
verbose = true;
verboseFreq = 100;

% Run BGD algoritm and print the results
 
tStart = tic;
[w,b,J_hist] = hBatchGradientDescentMV(X_train,y_train,alpha,condStop,maxIter,verbose,verboseFreq);
```

```matlabTextOutput
Iteration #0        J = 2529.46    w_1 = 2.41e-01    w_2 = 5.59e-04    w_3 = 1.84e-04    w_4 = 6.03e-03    b = 1.45e-04
Iteration #100        J = 695.99    w_1 = 2.02e-01    w_2 = 7.98e-04    w_3 = -9.97e-04    w_4 = -2.20e-03    b = -1.20e-04
Iteration #200        J = 694.92    w_1 = 2.03e-01    w_2 = 1.13e-03    w_3 = -2.14e-03    w_4 = -9.41e-03    b = -3.60e-04
Iteration #300        J = 693.86    w_1 = 2.03e-01    w_2 = 1.46e-03    w_3 = -3.29e-03    w_4 = -1.66e-02    b = -5.98e-04
Iteration #400        J = 692.81    w_1 = 2.03e-01    w_2 = 1.78e-03    w_3 = -4.43e-03    w_4 = -2.37e-02    b = -8.36e-04
Iteration #500        J = 691.77    w_1 = 2.03e-01    w_2 = 2.11e-03    w_3 = -5.57e-03    w_4 = -3.08e-02    b = -1.07e-03
Iteration #600        J = 690.73    w_1 = 2.03e-01    w_2 = 2.44e-03    w_3 = -6.71e-03    w_4 = -3.79e-02    b = -1.31e-03
Iteration #700        J = 689.71    w_1 = 2.03e-01    w_2 = 2.77e-03    w_3 = -7.85e-03    w_4 = -4.50e-02    b = -1.54e-03
Iteration #800        J = 688.70    w_1 = 2.04e-01    w_2 = 3.10e-03    w_3 = -8.99e-03    w_4 = -5.20e-02    b = -1.77e-03
Iteration #900        J = 687.69    w_1 = 2.04e-01    w_2 = 3.43e-03    w_3 = -1.01e-02    w_4 = -5.90e-02    b = -2.01e-03
==================================
BGD Stopped: max number of iterations (999)
```

```matlab
tElapsed1 = toc(tStart);
```

Let's plot the cost function against the iteration number

```matlab
figure;
plot(1:length(J_hist),J_hist);
xlabel('# Iterations');
ylabel('Cost function, J(w,b)');
```

![figure_0.png](./supervised_learning_week2_part1_media/figure_0.png)

```matlab
disp(['The parameters found by gradient descent: w = [',num2str(w(:,end).'),'], b = ',num2str(b(end))])
```

```matlabTextOutput
The parameters found by gradient descent: w = [0.20397   0.0037492   -0.011249   -0.065861], b = -0.0022354
```

```matlab
for i = 1:m
    fprintf('Example #%d - Estimated value: %.2f | Target value: %.2f\n',i, w.'*X_train(i,:).'+b, y_train(i))
end
```

```matlabTextOutput
Example #1 - Estimated value: 508.04 | Target value: 409.62
Example #4.286872e+02 - Estimated value: 424.99 | Target value: 425.71
Example #4.255719e+02 - Estimated value: 425.60 | Target value: 425.59
Example #4.255964e+02 - Estimated value: 425.60 | Target value: 425.60
Example #4.255980e+02 - Estimated value: 425.60 | Target value: 425.60
Example #4.255998e+02 - Estimated value: 425.60 | Target value: 425.60
Example #4.256017e+02 - Estimated value: 425.60 | Target value: 425.60
Example #4.256035e+02 - Estimated value: 425.60 | Target value: 425.60
Example #4.256053e+02 - Estimated value: 425.61 | Target value: 425.61
Example #4.256071e+02 - Estimated value: 425.61 | Target value: 425.61
Example #4.256090e+02 - Estimated value: 425.61 | Target value: 425.61
Example #4.256108e+02 - Estimated value: 425.61 | Target value: 425.61
Example #4.256126e+02 - Estimated value: 425.61 | Target value: 425.61
Example #4.256144e+02 - Estimated value: 425.62 | Target value: 425.62
Example #4.256162e+02 - Estimated value: 425.62 | Target value: 425.62
Example #4.256181e+02 - Estimated value: 425.62 | Target value: 425.62
Example #4.256199e+02 - Estimated value: 425.62 | Target value: 425.62
Example #4.256217e+02 - Estimated value: 425.62 | Target value: 425.62
Example #4.256235e+02 - Estimated value: 425.62 | Target value: 425.62
Example #4.256253e+02 - Estimated value: 425.63 | Target value: 425.63
Example #4.256271e+02 - Estimated value: 425.63 | Target value: 425.63
Example #4.256290e+02 - Estimated value: 425.63 | Target value: 425.63
Example #4.256308e+02 - Estimated value: 425.63 | Target value: 425.63
Example #4.256326e+02 - Estimated value: 425.63 | Target value: 425.63
Example #4.256344e+02 - Estimated value: 425.64 | Target value: 425.64
Example #4.256362e+02 - Estimated value: 425.64 | Target value: 425.64
Example #4.256381e+02 - Estimated value: 425.64 | Target value: 425.64
Example #4.256399e+02 - Estimated value: 425.64 | Target value: 425.64
Example #4.256417e+02 - Estimated value: 425.64 | Target value: 425.64
Example #4.256435e+02 - Estimated value: 425.64 | Target value: 425.64
Example #4.256453e+02 - Estimated value: 425.65 | Target value: 425.65
Example #4.256471e+02 - Estimated value: 425.65 | Target value: 425.65
Example #4.256490e+02 - Estimated value: 425.65 | Target value: 425.65
Example #4.256508e+02 - Estimated value: 425.65 | Target value: 425.65
Example #4.256526e+02 - Estimated value: 425.65 | Target value: 425.65
Example #4.256544e+02 - Estimated value: 425.66 | Target value: 425.66
Example #4.256562e+02 - Estimated value: 425.66 | Target value: 425.66
Example #4.256580e+02 - Estimated value: 425.66 | Target value: 425.66
Example #4.256598e+02 - Estimated value: 425.66 | Target value: 425.66
Example #4.256617e+02 - Estimated value: 425.66 | Target value: 425.66
Example #4.256635e+02 - Estimated value: 425.66 | Target value: 425.66
Example #4.256653e+02 - Estimated value: 425.67 | Target value: 425.67
Example #4.256671e+02 - Estimated value: 425.67 | Target value: 425.67
Example #4.256689e+02 - Estimated value: 425.67 | Target value: 425.67
Example #4.256707e+02 - Estimated value: 425.67 | Target value: 425.67
Example #4.256725e+02 - Estimated value: 425.67 | Target value: 425.67
Example #4.256744e+02 - Estimated value: 425.67 | Target value: 425.68
Example #4.256762e+02 - Estimated value: 425.68 | Target value: 425.68
Example #4.256780e+02 - Estimated value: 425.68 | Target value: 425.68
Example #4.256798e+02 - Estimated value: 425.68 | Target value: 425.68
Example #4.256816e+02 - Estimated value: 425.68 | Target value: 425.68
Example #4.256834e+02 - Estimated value: 425.68 | Target value: 425.68
Example #4.256852e+02 - Estimated value: 425.69 | Target value: 425.69
Example #4.256870e+02 - Estimated value: 425.69 | Target value: 425.69
Example #4.256888e+02 - Estimated value: 425.69 | Target value: 425.69
Example #4.256907e+02 - Estimated value: 425.69 | Target value: 425.69
Example #4.256925e+02 - Estimated value: 425.69 | Target value: 425.69
Example #4.256943e+02 - Estimated value: 425.69 | Target value: 425.70
Example #4.256961e+02 - Estimated value: 425.70 | Target value: 425.70
Example #4.256979e+02 - Estimated value: 425.70 | Target value: 425.70
Example #4.256997e+02 - Estimated value: 425.70 | Target value: 425.70
Example #4.257015e+02 - Estimated value: 425.70 | Target value: 425.70
Example #4.257033e+02 - Estimated value: 425.70 | Target value: 425.70
Example #4.257051e+02 - Estimated value: 425.71 | Target value: 425.71
Example #4.257069e+02 - Estimated value: 425.71 | Target value: 425.71
Example #4.257087e+02 - Estimated value: 425.71 | Target value: 425.71
Example #4.257106e+02 - Estimated value: 425.71 | Target value: 425.71
Example #4.257124e+02 - Estimated value: 425.71 | Target value: 425.71
Example #4.257142e+02 - Estimated value: 425.71 | Target value: 425.72
Example #4.257160e+02 - Estimated value: 425.72 | Target value: 425.72
Example #4.257178e+02 - Estimated value: 425.72 | Target value: 425.72
Example #4.257196e+02 - Estimated value: 425.72 | Target value: 425.72
Example #4.257214e+02 - Estimated value: 425.72 | Target value: 425.72
Example #4.257232e+02 - Estimated value: 425.72 | Target value: 425.72
Example #4.257250e+02 - Estimated value: 425.73 | Target value: 425.73
Example #4.257268e+02 - Estimated value: 425.73 | Target value: 425.73
Example #4.257286e+02 - Estimated value: 425.73 | Target value: 425.73
Example #4.257304e+02 - Estimated value: 425.73 | Target value: 425.73
Example #4.257322e+02 - Estimated value: 425.73 | Target value: 425.73
Example #4.257340e+02 - Estimated value: 425.73 | Target value: 425.74
Example #4.257358e+02 - Estimated value: 425.74 | Target value: 425.74
Example #4.257376e+02 - Estimated value: 425.74 | Target value: 425.74
Example #4.257394e+02 - Estimated value: 425.74 | Target value: 425.74
Example #4.257412e+02 - Estimated value: 425.74 | Target value: 425.74
Example #4.257431e+02 - Estimated value: 425.74 | Target value: 425.74
Example #4.257449e+02 - Estimated value: 425.75 | Target value: 425.75
Example #4.257467e+02 - Estimated value: 425.75 | Target value: 425.75
Example #4.257485e+02 - Estimated value: 425.75 | Target value: 425.75
Example #4.257503e+02 - Estimated value: 425.75 | Target value: 425.75
Example #4.257521e+02 - Estimated value: 425.75 | Target value: 425.75
Example #4.257539e+02 - Estimated value: 425.75 | Target value: 425.76
Example #4.257557e+02 - Estimated value: 425.76 | Target value: 425.76
Example #4.257575e+02 - Estimated value: 425.76 | Target value: 425.76
Example #4.257593e+02 - Estimated value: 425.76 | Target value: 425.76
Example #4.257611e+02 - Estimated value: 425.76 | Target value: 425.76
Example #4.257629e+02 - Estimated value: 425.76 | Target value: 425.76
Example #4.257647e+02 - Estimated value: 425.77 | Target value: 425.77
Example #4.257665e+02 - Estimated value: 425.77 | Target value: 425.77
Example #4.257683e+02 - Estimated value: 425.77 | Target value: 425.77
Example #4.257701e+02 - Estimated value: 425.77 | Target value: 425.77
Example #4.257719e+02 - Estimated value: 425.77 | Target value: 425.77
Example #4.257737e+02 - Estimated value: 425.77 | Target value: 425.77
Example #4.257755e+02 - Estimated value: 425.78 | Target value: 425.78
Example #4.257773e+02 - Estimated value: 425.78 | Target value: 425.78
Example #4.257791e+02 - Estimated value: 425.78 | Target value: 425.78
Example #4.257809e+02 - Estimated value: 425.78 | Target value: 425.78
Example #4.257827e+02 - Estimated value: 425.78 | Target value: 425.78
Example #4.257845e+02 - Estimated value: 425.79 | Target value: 425.79
Example #4.257863e+02 - Estimated value: 425.79 | Target value: 425.79
Example #4.257881e+02 - Estimated value: 425.79 | Target value: 425.79
Example #4.257899e+02 - Estimated value: 425.79 | Target value: 425.79
Example #4.257917e+02 - Estimated value: 425.79 | Target value: 425.79
Example #4.257935e+02 - Estimated value: 425.79 | Target value: 425.79
Example #4.257952e+02 - Estimated value: 425.80 | Target value: 425.80
Example #4.257970e+02 - Estimated value: 425.80 | Target value: 425.80
Example #4.257988e+02 - Estimated value: 425.80 | Target value: 425.80
Example #4.258006e+02 - Estimated value: 425.80 | Target value: 425.80
Example #4.258024e+02 - Estimated value: 425.80 | Target value: 425.80
Example #4.258042e+02 - Estimated value: 425.80 | Target value: 425.81
Example #4.258060e+02 - Estimated value: 425.81 | Target value: 425.81
Example #4.258078e+02 - Estimated value: 425.81 | Target value: 425.81
Example #4.258096e+02 - Estimated value: 425.81 | Target value: 425.81
Example #4.258114e+02 - Estimated value: 425.81 | Target value: 425.81
Example #4.258132e+02 - Estimated value: 425.81 | Target value: 425.81
Example #4.258150e+02 - Estimated value: 425.82 | Target value: 425.82
Example #4.258168e+02 - Estimated value: 425.82 | Target value: 425.82
Example #4.258186e+02 - Estimated value: 425.82 | Target value: 425.82
Example #4.258204e+02 - Estimated value: 425.82 | Target value: 425.82
Example #4.258222e+02 - Estimated value: 425.82 | Target value: 425.82
Example #4.258240e+02 - Estimated value: 425.82 | Target value: 425.83
Example #4.258257e+02 - Estimated value: 425.83 | Target value: 425.83
Example #4.258275e+02 - Estimated value: 425.83 | Target value: 425.83
Example #4.258293e+02 - Estimated value: 425.83 | Target value: 425.83
Example #4.258311e+02 - Estimated value: 425.83 | Target value: 425.83
Example #4.258329e+02 - Estimated value: 425.83 | Target value: 425.83
Example #4.258347e+02 - Estimated value: 425.84 | Target value: 425.84
Example #4.258365e+02 - Estimated value: 425.84 | Target value: 425.84
Example #4.258383e+02 - Estimated value: 425.84 | Target value: 425.84
Example #4.258401e+02 - Estimated value: 425.84 | Target value: 425.84
Example #4.258419e+02 - Estimated value: 425.84 | Target value: 425.84
Example #4.258437e+02 - Estimated value: 425.84 | Target value: 425.84
Example #4.258455e+02 - Estimated value: 425.85 | Target value: 425.85
Example #4.258472e+02 - Estimated value: 425.85 | Target value: 425.85
Example #4.258490e+02 - Estimated value: 425.85 | Target value: 425.85
Example #4.258508e+02 - Estimated value: 425.85 | Target value: 425.85
Example #4.258526e+02 - Estimated value: 425.85 | Target value: 425.85
Example #4.258544e+02 - Estimated value: 425.85 | Target value: 425.86
Example #4.258562e+02 - Estimated value: 425.86 | Target value: 425.86
Example #4.258580e+02 - Estimated value: 425.86 | Target value: 425.86
Example #4.258598e+02 - Estimated value: 425.86 | Target value: 425.86
Example #4.258615e+02 - Estimated value: 425.86 | Target value: 425.86
Example #4.258633e+02 - Estimated value: 425.86 | Target value: 425.86
Example #4.258651e+02 - Estimated value: 425.87 | Target value: 425.87
Example #4.258669e+02 - Estimated value: 425.87 | Target value: 425.87
Example #4.258687e+02 - Estimated value: 425.87 | Target value: 425.87
Example #4.258705e+02 - Estimated value: 425.87 | Target value: 425.87
Example #4.258723e+02 - Estimated value: 425.87 | Target value: 425.87
Example #4.258741e+02 - Estimated value: 425.87 | Target value: 425.88
Example #4.258758e+02 - Estimated value: 425.88 | Target value: 425.88
Example #4.258776e+02 - Estimated value: 425.88 | Target value: 425.88
Example #4.258794e+02 - Estimated value: 425.88 | Target value: 425.88
Example #4.258812e+02 - Estimated value: 425.88 | Target value: 425.88
Example #4.258830e+02 - Estimated value: 425.88 | Target value: 425.88
Example #4.258848e+02 - Estimated value: 425.89 | Target value: 425.89
Example #4.258866e+02 - Estimated value: 425.89 | Target value: 425.89
Example #4.258883e+02 - Estimated value: 425.89 | Target value: 425.89
Example #4.258901e+02 - Estimated value: 425.89 | Target value: 425.89
Example #4.258919e+02 - Estimated value: 425.89 | Target value: 425.89
Example #4.258937e+02 - Estimated value: 425.89 | Target value: 425.89
Example #4.258955e+02 - Estimated value: 425.90 | Target value: 425.90
Example #4.258973e+02 - Estimated value: 425.90 | Target value: 425.90
Example #4.258990e+02 - Estimated value: 425.90 | Target value: 425.90
Example #4.259008e+02 - Estimated value: 425.90 | Target value: 425.90
Example #4.259026e+02 - Estimated value: 425.90 | Target value: 425.90
Example #4.259044e+02 - Estimated value: 425.90 | Target value: 425.91
Example #4.259062e+02 - Estimated value: 425.91 | Target value: 425.91
Example #4.259079e+02 - Estimated value: 425.91 | Target value: 425.91
Example #4.259097e+02 - Estimated value: 425.91 | Target value: 425.91
Example #4.259115e+02 - Estimated value: 425.91 | Target value: 425.91
Example #4.259133e+02 - Estimated value: 425.91 | Target value: 425.91
Example #4.259151e+02 - Estimated value: 425.92 | Target value: 425.92
Example #4.259168e+02 - Estimated value: 425.92 | Target value: 425.92
Example #4.259186e+02 - Estimated value: 425.92 | Target value: 425.92
Example #4.259204e+02 - Estimated value: 425.92 | Target value: 425.92
Example #4.259222e+02 - Estimated value: 425.92 | Target value: 425.92
Example #4.259240e+02 - Estimated value: 425.92 | Target value: 425.93
Example #4.259257e+02 - Estimated value: 425.93 | Target value: 425.93
Example #4.259275e+02 - Estimated value: 425.93 | Target value: 425.93
Example #4.259293e+02 - Estimated value: 425.93 | Target value: 425.93
Example #4.259311e+02 - Estimated value: 425.93 | Target value: 425.93
Example #4.259329e+02 - Estimated value: 425.93 | Target value: 425.93
Example #4.259346e+02 - Estimated value: 425.94 | Target value: 425.94
Example #4.259364e+02 - Estimated value: 425.94 | Target value: 425.94
Example #4.259382e+02 - Estimated value: 425.94 | Target value: 425.94
Example #4.259400e+02 - Estimated value: 425.94 | Target value: 425.94
Example #4.259417e+02 - Estimated value: 425.94 | Target value: 425.94
Example #4.259435e+02 - Estimated value: 425.94 | Target value: 425.94
Example #4.259453e+02 - Estimated value: 425.95 | Target value: 425.95
Example #4.259471e+02 - Estimated value: 425.95 | Target value: 425.95
Example #4.259489e+02 - Estimated value: 425.95 | Target value: 425.95
Example #4.259506e+02 - Estimated value: 425.95 | Target value: 425.95
Example #4.259524e+02 - Estimated value: 425.95 | Target value: 425.95
Example #4.259542e+02 - Estimated value: 425.95 | Target value: 425.96
Example #4.259560e+02 - Estimated value: 425.96 | Target value: 425.96
Example #4.259577e+02 - Estimated value: 425.96 | Target value: 425.96
Example #4.259595e+02 - Estimated value: 425.96 | Target value: 425.96
Example #4.259613e+02 - Estimated value: 425.96 | Target value: 425.96
Example #4.259631e+02 - Estimated value: 425.96 | Target value: 425.96
Example #4.259648e+02 - Estimated value: 425.97 | Target value: 425.97
Example #4.259666e+02 - Estimated value: 425.97 | Target value: 425.97
Example #4.259684e+02 - Estimated value: 425.97 | Target value: 425.97
Example #4.259701e+02 - Estimated value: 425.97 | Target value: 425.97
Example #4.259719e+02 - Estimated value: 425.97 | Target value: 425.97
Example #4.259737e+02 - Estimated value: 425.97 | Target value: 425.97
Example #4.259755e+02 - Estimated value: 425.98 | Target value: 425.98
Example #4.259772e+02 - Estimated value: 425.98 | Target value: 425.98
Example #4.259790e+02 - Estimated value: 425.98 | Target value: 425.98
Example #4.259808e+02 - Estimated value: 425.98 | Target value: 425.98
Example #4.259826e+02 - Estimated value: 425.98 | Target value: 425.98
Example #4.259843e+02 - Estimated value: 425.98 | Target value: 425.99
Example #4.259861e+02 - Estimated value: 425.99 | Target value: 425.99
Example #4.259879e+02 - Estimated value: 425.99 | Target value: 425.99
Example #4.259896e+02 - Estimated value: 425.99 | Target value: 425.99
Example #4.259914e+02 - Estimated value: 425.99 | Target value: 425.99
Example #4.259932e+02 - Estimated value: 425.99 | Target value: 425.99
Example #4.259949e+02 - Estimated value: 426.00 | Target value: 426.00
Example #4.259967e+02 - Estimated value: 426.00 | Target value: 426.00
Example #4.259985e+02 - Estimated value: 426.00 | Target value: 426.00
Example #4.260003e+02 - Estimated value: 426.00 | Target value: 426.00
Example #4.260020e+02 - Estimated value: 426.00 | Target value: 426.00
Example #4.260038e+02 - Estimated value: 426.00 | Target value: 426.00
Example #4.260056e+02 - Estimated value: 426.01 | Target value: 426.01
Example #4.260073e+02 - Estimated value: 426.01 | Target value: 426.01
Example #4.260091e+02 - Estimated value: 426.01 | Target value: 426.01
Example #4.260109e+02 - Estimated value: 426.01 | Target value: 426.01
Example #4.260126e+02 - Estimated value: 426.01 | Target value: 426.01
Example #4.260144e+02 - Estimated value: 426.01 | Target value: 426.02
Example #4.260162e+02 - Estimated value: 426.02 | Target value: 426.02
Example #4.260179e+02 - Estimated value: 426.02 | Target value: 426.02
Example #4.260197e+02 - Estimated value: 426.02 | Target value: 426.02
Example #4.260215e+02 - Estimated value: 426.02 | Target value: 426.02
Example #4.260232e+02 - Estimated value: 426.02 | Target value: 426.02
Example #4.260250e+02 - Estimated value: 426.03 | Target value: 426.03
Example #4.260268e+02 - Estimated value: 426.03 | Target value: 426.03
Example #4.260285e+02 - Estimated value: 426.03 | Target value: 426.03
Example #4.260303e+02 - Estimated value: 426.03 | Target value: 426.03
Example #4.260321e+02 - Estimated value: 426.03 | Target value: 426.03
Example #4.260338e+02 - Estimated value: 426.03 | Target value: 426.04
Example #4.260356e+02 - Estimated value: 426.04 | Target value: 426.04
Example #4.260374e+02 - Estimated value: 426.04 | Target value: 426.04
Example #4.260391e+02 - Estimated value: 426.04 | Target value: 426.04
Example #4.260409e+02 - Estimated value: 426.04 | Target value: 426.04
Example #4.260426e+02 - Estimated value: 426.04 | Target value: 426.04
Example #4.260444e+02 - Estimated value: 426.04 | Target value: 426.05
Example #4.260462e+02 - Estimated value: 426.05 | Target value: 426.05
Example #4.260479e+02 - Estimated value: 426.05 | Target value: 426.05
Example #4.260497e+02 - Estimated value: 426.05 | Target value: 426.05
Example #4.260515e+02 - Estimated value: 426.05 | Target value: 426.05
Example #4.260532e+02 - Estimated value: 426.05 | Target value: 426.05
Example #4.260550e+02 - Estimated value: 426.06 | Target value: 426.06
Example #4.260567e+02 - Estimated value: 426.06 | Target value: 426.06
Example #4.260585e+02 - Estimated value: 426.06 | Target value: 426.06
Example #4.260603e+02 - Estimated value: 426.06 | Target value: 426.06
Example #4.260620e+02 - Estimated value: 426.06 | Target value: 426.06
Example #4.260638e+02 - Estimated value: 426.06 | Target value: 426.06
Example #4.260656e+02 - Estimated value: 426.07 | Target value: 426.07
Example #4.260673e+02 - Estimated value: 426.07 | Target value: 426.07
Example #4.260691e+02 - Estimated value: 426.07 | Target value: 426.07
Example #4.260708e+02 - Estimated value: 426.07 | Target value: 426.07
Example #4.260726e+02 - Estimated value: 426.07 | Target value: 426.07
Example #4.260744e+02 - Estimated value: 426.07 | Target value: 426.08
Example #4.260761e+02 - Estimated value: 426.08 | Target value: 426.08
Example #4.260779e+02 - Estimated value: 426.08 | Target value: 426.08
Example #4.260796e+02 - Estimated value: 426.08 | Target value: 426.08
Example #4.260814e+02 - Estimated value: 426.08 | Target value: 426.08
Example #4.260831e+02 - Estimated value: 426.08 | Target value: 426.08
Example #4.260849e+02 - Estimated value: 426.09 | Target value: 426.09
Example #4.260867e+02 - Estimated value: 426.09 | Target value: 426.09
Example #4.260884e+02 - Estimated value: 426.09 | Target value: 426.09
Example #4.260902e+02 - Estimated value: 426.09 | Target value: 426.09
Example #4.260919e+02 - Estimated value: 426.09 | Target value: 426.09
Example #4.260937e+02 - Estimated value: 426.09 | Target value: 426.09
Example #4.260954e+02 - Estimated value: 426.10 | Target value: 426.10
Example #4.260972e+02 - Estimated value: 426.10 | Target value: 426.10
Example #4.260990e+02 - Estimated value: 426.10 | Target value: 426.10
Example #4.261007e+02 - Estimated value: 426.10 | Target value: 426.10
Example #4.261025e+02 - Estimated value: 426.10 | Target value: 426.10
Example #4.261042e+02 - Estimated value: 426.10 | Target value: 426.11
Example #4.261060e+02 - Estimated value: 426.11 | Target value: 426.11
Example #4.261077e+02 - Estimated value: 426.11 | Target value: 426.11
Example #4.261095e+02 - Estimated value: 426.11 | Target value: 426.11
Example #4.261112e+02 - Estimated value: 426.11 | Target value: 426.11
Example #4.261130e+02 - Estimated value: 426.11 | Target value: 426.11
Example #4.261148e+02 - Estimated value: 426.12 | Target value: 426.12
Example #4.261165e+02 - Estimated value: 426.12 | Target value: 426.12
Example #4.261183e+02 - Estimated value: 426.12 | Target value: 426.12
Example #4.261200e+02 - Estimated value: 426.12 | Target value: 426.12
Example #4.261218e+02 - Estimated value: 426.12 | Target value: 426.12
Example #4.261235e+02 - Estimated value: 426.12 | Target value: 426.12
Example #4.261253e+02 - Estimated value: 426.13 | Target value: 426.13
Example #4.261270e+02 - Estimated value: 426.13 | Target value: 426.13
Example #4.261288e+02 - Estimated value: 426.13 | Target value: 426.13
Example #4.261305e+02 - Estimated value: 426.13 | Target value: 426.13
Example #4.261323e+02 - Estimated value: 426.13 | Target value: 426.13
Example #4.261340e+02 - Estimated value: 426.13 | Target value: 426.14
Example #4.261358e+02 - Estimated value: 426.14 | Target value: 426.14
Example #4.261375e+02 - Estimated value: 426.14 | Target value: 426.14
Example #4.261393e+02 - Estimated value: 426.14 | Target value: 426.14
Example #4.261410e+02 - Estimated value: 426.14 | Target value: 426.14
Example #4.261428e+02 - Estimated value: 426.14 | Target value: 426.14
Example #4.261445e+02 - Estimated value: 426.15 | Target value: 426.15
Example #4.261463e+02 - Estimated value: 426.15 | Target value: 426.15
Example #4.261480e+02 - Estimated value: 426.15 | Target value: 426.15
Example #4.261498e+02 - Estimated value: 426.15 | Target value: 426.15
Example #4.261515e+02 - Estimated value: 426.15 | Target value: 426.15
Example #4.261533e+02 - Estimated value: 426.15 | Target value: 426.15
Example #4.261550e+02 - Estimated value: 426.16 | Target value: 426.16
Example #4.261568e+02 - Estimated value: 426.16 | Target value: 426.16
Example #4.261585e+02 - Estimated value: 426.16 | Target value: 426.16
Example #4.261603e+02 - Estimated value: 426.16 | Target value: 426.16
Example #4.261620e+02 - Estimated value: 426.16 | Target value: 426.16
Example #4.261638e+02 - Estimated value: 426.16 | Target value: 426.16
Example #4.261655e+02 - Estimated value: 426.17 | Target value: 426.17
Example #4.261673e+02 - Estimated value: 426.17 | Target value: 426.17
Example #4.261690e+02 - Estimated value: 426.17 | Target value: 426.17
Example #4.261708e+02 - Estimated value: 426.17 | Target value: 426.17
Example #4.261725e+02 - Estimated value: 426.17 | Target value: 426.17
Example #4.261743e+02 - Estimated value: 426.17 | Target value: 426.18
Example #4.261760e+02 - Estimated value: 426.18 | Target value: 426.18
Example #4.261777e+02 - Estimated value: 426.18 | Target value: 426.18
Example #4.261795e+02 - Estimated value: 426.18 | Target value: 426.18
Example #4.261812e+02 - Estimated value: 426.18 | Target value: 426.18
Example #4.261830e+02 - Estimated value: 426.18 | Target value: 426.18
Example #4.261847e+02 - Estimated value: 426.19 | Target value: 460.00
Example #2 - Estimated value: 341.97 | Target value: 275.72
Example #2.885561e+02 - Estimated value: 286.07 | Target value: 286.55
Example #2.864569e+02 - Estimated value: 286.47 | Target value: 286.47
Example #2.864713e+02 - Estimated value: 286.47 | Target value: 286.47
Example #2.864702e+02 - Estimated value: 286.47 | Target value: 286.47
Example #2.864693e+02 - Estimated value: 286.47 | Target value: 286.47
Example #2.864683e+02 - Estimated value: 286.47 | Target value: 286.47
Example #2.864674e+02 - Estimated value: 286.47 | Target value: 286.47
Example #2.864665e+02 - Estimated value: 286.47 | Target value: 286.47
Example #2.864655e+02 - Estimated value: 286.47 | Target value: 286.46
Example #2.864646e+02 - Estimated value: 286.46 | Target value: 286.46
Example #2.864636e+02 - Estimated value: 286.46 | Target value: 286.46
Example #2.864627e+02 - Estimated value: 286.46 | Target value: 286.46
Example #2.864618e+02 - Estimated value: 286.46 | Target value: 286.46
Example #2.864608e+02 - Estimated value: 286.46 | Target value: 286.46
Example #2.864599e+02 - Estimated value: 286.46 | Target value: 286.46
Example #2.864590e+02 - Estimated value: 286.46 | Target value: 286.46
Example #2.864580e+02 - Estimated value: 286.46 | Target value: 286.46
Example #2.864571e+02 - Estimated value: 286.46 | Target value: 286.46
Example #2.864561e+02 - Estimated value: 286.46 | Target value: 286.46
Example #2.864552e+02 - Estimated value: 286.45 | Target value: 286.45
Example #2.864543e+02 - Estimated value: 286.45 | Target value: 286.45
Example #2.864533e+02 - Estimated value: 286.45 | Target value: 286.45
Example #2.864524e+02 - Estimated value: 286.45 | Target value: 286.45
Example #2.864515e+02 - Estimated value: 286.45 | Target value: 286.45
Example #2.864505e+02 - Estimated value: 286.45 | Target value: 286.45
Example #2.864496e+02 - Estimated value: 286.45 | Target value: 286.45
Example #2.864486e+02 - Estimated value: 286.45 | Target value: 286.45
Example #2.864477e+02 - Estimated value: 286.45 | Target value: 286.45
Example #2.864468e+02 - Estimated value: 286.45 | Target value: 286.45
Example #2.864458e+02 - Estimated value: 286.45 | Target value: 286.45
Example #2.864449e+02 - Estimated value: 286.44 | Target value: 286.44
Example #2.864440e+02 - Estimated value: 286.44 | Target value: 286.44
Example #2.864430e+02 - Estimated value: 286.44 | Target value: 286.44
Example #2.864421e+02 - Estimated value: 286.44 | Target value: 286.44
Example #2.864412e+02 - Estimated value: 286.44 | Target value: 286.44
Example #2.864402e+02 - Estimated value: 286.44 | Target value: 286.44
Example #2.864393e+02 - Estimated value: 286.44 | Target value: 286.44
Example #2.864383e+02 - Estimated value: 286.44 | Target value: 286.44
Example #2.864374e+02 - Estimated value: 286.44 | Target value: 286.44
Example #2.864365e+02 - Estimated value: 286.44 | Target value: 286.44
Example #2.864355e+02 - Estimated value: 286.44 | Target value: 286.43
Example #2.864346e+02 - Estimated value: 286.43 | Target value: 286.43
Example #2.864337e+02 - Estimated value: 286.43 | Target value: 286.43
Example #2.864327e+02 - Estimated value: 286.43 | Target value: 286.43
Example #2.864318e+02 - Estimated value: 286.43 | Target value: 286.43
Example #2.864309e+02 - Estimated value: 286.43 | Target value: 286.43
Example #2.864299e+02 - Estimated value: 286.43 | Target value: 286.43
Example #2.864290e+02 - Estimated value: 286.43 | Target value: 286.43
Example #2.864281e+02 - Estimated value: 286.43 | Target value: 286.43
Example #2.864271e+02 - Estimated value: 286.43 | Target value: 286.43
Example #2.864262e+02 - Estimated value: 286.43 | Target value: 286.43
Example #2.864253e+02 - Estimated value: 286.42 | Target value: 286.42
Example #2.864243e+02 - Estimated value: 286.42 | Target value: 286.42
Example #2.864234e+02 - Estimated value: 286.42 | Target value: 286.42
Example #2.864225e+02 - Estimated value: 286.42 | Target value: 286.42
Example #2.864215e+02 - Estimated value: 286.42 | Target value: 286.42
Example #2.864206e+02 - Estimated value: 286.42 | Target value: 286.42
Example #2.864197e+02 - Estimated value: 286.42 | Target value: 286.42
Example #2.864187e+02 - Estimated value: 286.42 | Target value: 286.42
Example #2.864178e+02 - Estimated value: 286.42 | Target value: 286.42
Example #2.864169e+02 - Estimated value: 286.42 | Target value: 286.42
Example #2.864159e+02 - Estimated value: 286.42 | Target value: 286.42
Example #2.864150e+02 - Estimated value: 286.41 | Target value: 286.41
Example #2.864141e+02 - Estimated value: 286.41 | Target value: 286.41
Example #2.864131e+02 - Estimated value: 286.41 | Target value: 286.41
Example #2.864122e+02 - Estimated value: 286.41 | Target value: 286.41
Example #2.864113e+02 - Estimated value: 286.41 | Target value: 286.41
Example #2.864104e+02 - Estimated value: 286.41 | Target value: 286.41
Example #2.864094e+02 - Estimated value: 286.41 | Target value: 286.41
Example #2.864085e+02 - Estimated value: 286.41 | Target value: 286.41
Example #2.864076e+02 - Estimated value: 286.41 | Target value: 286.41
Example #2.864066e+02 - Estimated value: 286.41 | Target value: 286.41
Example #2.864057e+02 - Estimated value: 286.41 | Target value: 286.41
Example #2.864048e+02 - Estimated value: 286.40 | Target value: 286.40
Example #2.864038e+02 - Estimated value: 286.40 | Target value: 286.40
Example #2.864029e+02 - Estimated value: 286.40 | Target value: 286.40
Example #2.864020e+02 - Estimated value: 286.40 | Target value: 286.40
Example #2.864010e+02 - Estimated value: 286.40 | Target value: 286.40
Example #2.864001e+02 - Estimated value: 286.40 | Target value: 286.40
Example #2.863992e+02 - Estimated value: 286.40 | Target value: 286.40
Example #2.863983e+02 - Estimated value: 286.40 | Target value: 286.40
Example #2.863973e+02 - Estimated value: 286.40 | Target value: 286.40
Example #2.863964e+02 - Estimated value: 286.40 | Target value: 286.40
Example #2.863955e+02 - Estimated value: 286.40 | Target value: 286.39
Example #2.863945e+02 - Estimated value: 286.39 | Target value: 286.39
Example #2.863936e+02 - Estimated value: 286.39 | Target value: 286.39
Example #2.863927e+02 - Estimated value: 286.39 | Target value: 286.39
Example #2.863918e+02 - Estimated value: 286.39 | Target value: 286.39
Example #2.863908e+02 - Estimated value: 286.39 | Target value: 286.39
Example #2.863899e+02 - Estimated value: 286.39 | Target value: 286.39
Example #2.863890e+02 - Estimated value: 286.39 | Target value: 286.39
Example #2.863880e+02 - Estimated value: 286.39 | Target value: 286.39
Example #2.863871e+02 - Estimated value: 286.39 | Target value: 286.39
Example #2.863862e+02 - Estimated value: 286.39 | Target value: 286.39
Example #2.863853e+02 - Estimated value: 286.38 | Target value: 286.38
Example #2.863843e+02 - Estimated value: 286.38 | Target value: 286.38
Example #2.863834e+02 - Estimated value: 286.38 | Target value: 286.38
Example #2.863825e+02 - Estimated value: 286.38 | Target value: 286.38
Example #2.863815e+02 - Estimated value: 286.38 | Target value: 286.38
Example #2.863806e+02 - Estimated value: 286.38 | Target value: 286.38
Example #2.863797e+02 - Estimated value: 286.38 | Target value: 286.38
Example #2.863788e+02 - Estimated value: 286.38 | Target value: 286.38
Example #2.863778e+02 - Estimated value: 286.38 | Target value: 286.38
Example #2.863769e+02 - Estimated value: 286.38 | Target value: 286.38
Example #2.863760e+02 - Estimated value: 286.38 | Target value: 286.38
Example #2.863750e+02 - Estimated value: 286.37 | Target value: 286.37
Example #2.863741e+02 - Estimated value: 286.37 | Target value: 286.37
Example #2.863732e+02 - Estimated value: 286.37 | Target value: 286.37
Example #2.863723e+02 - Estimated value: 286.37 | Target value: 286.37
Example #2.863713e+02 - Estimated value: 286.37 | Target value: 286.37
Example #2.863704e+02 - Estimated value: 286.37 | Target value: 286.37
Example #2.863695e+02 - Estimated value: 286.37 | Target value: 286.37
Example #2.863686e+02 - Estimated value: 286.37 | Target value: 286.37
Example #2.863676e+02 - Estimated value: 286.37 | Target value: 286.37
Example #2.863667e+02 - Estimated value: 286.37 | Target value: 286.37
Example #2.863658e+02 - Estimated value: 286.37 | Target value: 286.37
Example #2.863649e+02 - Estimated value: 286.36 | Target value: 286.36
Example #2.863639e+02 - Estimated value: 286.36 | Target value: 286.36
Example #2.863630e+02 - Estimated value: 286.36 | Target value: 286.36
Example #2.863621e+02 - Estimated value: 286.36 | Target value: 286.36
Example #2.863612e+02 - Estimated value: 286.36 | Target value: 286.36
Example #2.863602e+02 - Estimated value: 286.36 | Target value: 286.36
Example #2.863593e+02 - Estimated value: 286.36 | Target value: 286.36
Example #2.863584e+02 - Estimated value: 286.36 | Target value: 286.36
Example #2.863575e+02 - Estimated value: 286.36 | Target value: 286.36
Example #2.863565e+02 - Estimated value: 286.36 | Target value: 286.36
Example #2.863556e+02 - Estimated value: 286.36 | Target value: 286.35
Example #2.863547e+02 - Estimated value: 286.35 | Target value: 286.35
Example #2.863538e+02 - Estimated value: 286.35 | Target value: 286.35
Example #2.863528e+02 - Estimated value: 286.35 | Target value: 286.35
Example #2.863519e+02 - Estimated value: 286.35 | Target value: 286.35
Example #2.863510e+02 - Estimated value: 286.35 | Target value: 286.35
Example #2.863501e+02 - Estimated value: 286.35 | Target value: 286.35
Example #2.863491e+02 - Estimated value: 286.35 | Target value: 286.35
Example #2.863482e+02 - Estimated value: 286.35 | Target value: 286.35
Example #2.863473e+02 - Estimated value: 286.35 | Target value: 286.35
Example #2.863464e+02 - Estimated value: 286.35 | Target value: 286.35
Example #2.863455e+02 - Estimated value: 286.35 | Target value: 286.34
Example #2.863445e+02 - Estimated value: 286.34 | Target value: 286.34
Example #2.863436e+02 - Estimated value: 286.34 | Target value: 286.34
Example #2.863427e+02 - Estimated value: 286.34 | Target value: 286.34
Example #2.863418e+02 - Estimated value: 286.34 | Target value: 286.34
Example #2.863408e+02 - Estimated value: 286.34 | Target value: 286.34
Example #2.863399e+02 - Estimated value: 286.34 | Target value: 286.34
Example #2.863390e+02 - Estimated value: 286.34 | Target value: 286.34
Example #2.863381e+02 - Estimated value: 286.34 | Target value: 286.34
Example #2.863372e+02 - Estimated value: 286.34 | Target value: 286.34
Example #2.863362e+02 - Estimated value: 286.34 | Target value: 286.34
Example #2.863353e+02 - Estimated value: 286.34 | Target value: 286.33
Example #2.863344e+02 - Estimated value: 286.33 | Target value: 286.33
Example #2.863335e+02 - Estimated value: 286.33 | Target value: 286.33
Example #2.863325e+02 - Estimated value: 286.33 | Target value: 286.33
Example #2.863316e+02 - Estimated value: 286.33 | Target value: 286.33
Example #2.863307e+02 - Estimated value: 286.33 | Target value: 286.33
Example #2.863298e+02 - Estimated value: 286.33 | Target value: 286.33
Example #2.863289e+02 - Estimated value: 286.33 | Target value: 286.33
Example #2.863279e+02 - Estimated value: 286.33 | Target value: 286.33
Example #2.863270e+02 - Estimated value: 286.33 | Target value: 286.33
Example #2.863261e+02 - Estimated value: 286.33 | Target value: 286.33
Example #2.863252e+02 - Estimated value: 286.32 | Target value: 286.32
Example #2.863243e+02 - Estimated value: 286.32 | Target value: 286.32
Example #2.863233e+02 - Estimated value: 286.32 | Target value: 286.32
Example #2.863224e+02 - Estimated value: 286.32 | Target value: 286.32
Example #2.863215e+02 - Estimated value: 286.32 | Target value: 286.32
Example #2.863206e+02 - Estimated value: 286.32 | Target value: 286.32
Example #2.863197e+02 - Estimated value: 286.32 | Target value: 286.32
Example #2.863187e+02 - Estimated value: 286.32 | Target value: 286.32
Example #2.863178e+02 - Estimated value: 286.32 | Target value: 286.32
Example #2.863169e+02 - Estimated value: 286.32 | Target value: 286.32
Example #2.863160e+02 - Estimated value: 286.32 | Target value: 286.32
Example #2.863151e+02 - Estimated value: 286.31 | Target value: 286.31
Example #2.863141e+02 - Estimated value: 286.31 | Target value: 286.31
Example #2.863132e+02 - Estimated value: 286.31 | Target value: 286.31
Example #2.863123e+02 - Estimated value: 286.31 | Target value: 286.31
Example #2.863114e+02 - Estimated value: 286.31 | Target value: 286.31
Example #2.863105e+02 - Estimated value: 286.31 | Target value: 286.31
Example #2.863096e+02 - Estimated value: 286.31 | Target value: 286.31
Example #2.863086e+02 - Estimated value: 286.31 | Target value: 286.31
Example #2.863077e+02 - Estimated value: 286.31 | Target value: 286.31
Example #2.863068e+02 - Estimated value: 286.31 | Target value: 286.31
Example #2.863059e+02 - Estimated value: 286.31 | Target value: 286.31
Example #2.863050e+02 - Estimated value: 286.30 | Target value: 286.30
Example #2.863040e+02 - Estimated value: 286.30 | Target value: 286.30
Example #2.863031e+02 - Estimated value: 286.30 | Target value: 286.30
Example #2.863022e+02 - Estimated value: 286.30 | Target value: 286.30
Example #2.863013e+02 - Estimated value: 286.30 | Target value: 286.30
Example #2.863004e+02 - Estimated value: 286.30 | Target value: 286.30
Example #2.862995e+02 - Estimated value: 286.30 | Target value: 286.30
Example #2.862985e+02 - Estimated value: 286.30 | Target value: 286.30
Example #2.862976e+02 - Estimated value: 286.30 | Target value: 286.30
Example #2.862967e+02 - Estimated value: 286.30 | Target value: 286.30
Example #2.862958e+02 - Estimated value: 286.30 | Target value: 286.30
Example #2.862949e+02 - Estimated value: 286.29 | Target value: 286.29
Example #2.862940e+02 - Estimated value: 286.29 | Target value: 286.29
Example #2.862930e+02 - Estimated value: 286.29 | Target value: 286.29
Example #2.862921e+02 - Estimated value: 286.29 | Target value: 286.29
Example #2.862912e+02 - Estimated value: 286.29 | Target value: 286.29
Example #2.862903e+02 - Estimated value: 286.29 | Target value: 286.29
Example #2.862894e+02 - Estimated value: 286.29 | Target value: 286.29
Example #2.862885e+02 - Estimated value: 286.29 | Target value: 286.29
Example #2.862876e+02 - Estimated value: 286.29 | Target value: 286.29
Example #2.862866e+02 - Estimated value: 286.29 | Target value: 286.29
Example #2.862857e+02 - Estimated value: 286.29 | Target value: 286.29
Example #2.862848e+02 - Estimated value: 286.28 | Target value: 286.28
Example #2.862839e+02 - Estimated value: 286.28 | Target value: 286.28
Example #2.862830e+02 - Estimated value: 286.28 | Target value: 286.28
Example #2.862821e+02 - Estimated value: 286.28 | Target value: 286.28
Example #2.862811e+02 - Estimated value: 286.28 | Target value: 286.28
Example #2.862802e+02 - Estimated value: 286.28 | Target value: 286.28
Example #2.862793e+02 - Estimated value: 286.28 | Target value: 286.28
Example #2.862784e+02 - Estimated value: 286.28 | Target value: 286.28
Example #2.862775e+02 - Estimated value: 286.28 | Target value: 286.28
Example #2.862766e+02 - Estimated value: 286.28 | Target value: 286.28
Example #2.862757e+02 - Estimated value: 286.28 | Target value: 286.28
Example #2.862748e+02 - Estimated value: 286.27 | Target value: 286.27
Example #2.862738e+02 - Estimated value: 286.27 | Target value: 286.27
Example #2.862729e+02 - Estimated value: 286.27 | Target value: 286.27
Example #2.862720e+02 - Estimated value: 286.27 | Target value: 286.27
Example #2.862711e+02 - Estimated value: 286.27 | Target value: 286.27
Example #2.862702e+02 - Estimated value: 286.27 | Target value: 286.27
Example #2.862693e+02 - Estimated value: 286.27 | Target value: 286.27
Example #2.862684e+02 - Estimated value: 286.27 | Target value: 286.27
Example #2.862674e+02 - Estimated value: 286.27 | Target value: 286.27
Example #2.862665e+02 - Estimated value: 286.27 | Target value: 286.27
Example #2.862656e+02 - Estimated value: 286.27 | Target value: 286.27
Example #2.862647e+02 - Estimated value: 286.26 | Target value: 286.26
Example #2.862638e+02 - Estimated value: 286.26 | Target value: 286.26
Example #2.862629e+02 - Estimated value: 286.26 | Target value: 286.26
Example #2.862620e+02 - Estimated value: 286.26 | Target value: 286.26
Example #2.862611e+02 - Estimated value: 286.26 | Target value: 286.26
Example #2.862601e+02 - Estimated value: 286.26 | Target value: 286.26
Example #2.862592e+02 - Estimated value: 286.26 | Target value: 286.26
Example #2.862583e+02 - Estimated value: 286.26 | Target value: 286.26
Example #2.862574e+02 - Estimated value: 286.26 | Target value: 286.26
Example #2.862565e+02 - Estimated value: 286.26 | Target value: 286.26
Example #2.862556e+02 - Estimated value: 286.26 | Target value: 286.25
Example #2.862547e+02 - Estimated value: 286.25 | Target value: 286.25
Example #2.862538e+02 - Estimated value: 286.25 | Target value: 286.25
Example #2.862529e+02 - Estimated value: 286.25 | Target value: 286.25
Example #2.862519e+02 - Estimated value: 286.25 | Target value: 286.25
Example #2.862510e+02 - Estimated value: 286.25 | Target value: 286.25
Example #2.862501e+02 - Estimated value: 286.25 | Target value: 286.25
Example #2.862492e+02 - Estimated value: 286.25 | Target value: 286.25
Example #2.862483e+02 - Estimated value: 286.25 | Target value: 286.25
Example #2.862474e+02 - Estimated value: 286.25 | Target value: 286.25
Example #2.862465e+02 - Estimated value: 286.25 | Target value: 286.25
Example #2.862456e+02 - Estimated value: 286.25 | Target value: 286.24
Example #2.862447e+02 - Estimated value: 286.24 | Target value: 286.24
Example #2.862438e+02 - Estimated value: 286.24 | Target value: 286.24
Example #2.862428e+02 - Estimated value: 286.24 | Target value: 286.24
Example #2.862419e+02 - Estimated value: 286.24 | Target value: 286.24
Example #2.862410e+02 - Estimated value: 286.24 | Target value: 286.24
Example #2.862401e+02 - Estimated value: 286.24 | Target value: 286.24
Example #2.862392e+02 - Estimated value: 286.24 | Target value: 286.24
Example #2.862383e+02 - Estimated value: 286.24 | Target value: 286.24
Example #2.862374e+02 - Estimated value: 286.24 | Target value: 286.24
Example #2.862365e+02 - Estimated value: 286.24 | Target value: 286.24
Example #2.862356e+02 - Estimated value: 286.24 | Target value: 286.23
Example #2.862347e+02 - Estimated value: 286.23 | Target value: 286.23
Example #2.862338e+02 - Estimated value: 286.23 | Target value: 286.23
Example #2.862329e+02 - Estimated value: 286.23 | Target value: 286.23
Example #2.862319e+02 - Estimated value: 286.23 | Target value: 286.23
Example #2.862310e+02 - Estimated value: 286.23 | Target value: 286.23
Example #2.862301e+02 - Estimated value: 286.23 | Target value: 286.23
Example #2.862292e+02 - Estimated value: 286.23 | Target value: 286.23
Example #2.862283e+02 - Estimated value: 286.23 | Target value: 286.23
Example #2.862274e+02 - Estimated value: 286.23 | Target value: 286.23
Example #2.862265e+02 - Estimated value: 286.23 | Target value: 286.23
Example #2.862256e+02 - Estimated value: 286.23 | Target value: 286.22
Example #2.862247e+02 - Estimated value: 286.22 | Target value: 286.22
Example #2.862238e+02 - Estimated value: 286.22 | Target value: 286.22
Example #2.862229e+02 - Estimated value: 286.22 | Target value: 286.22
Example #2.862220e+02 - Estimated value: 286.22 | Target value: 286.22
Example #2.862211e+02 - Estimated value: 286.22 | Target value: 286.22
Example #2.862201e+02 - Estimated value: 286.22 | Target value: 286.22
Example #2.862192e+02 - Estimated value: 286.22 | Target value: 286.22
Example #2.862183e+02 - Estimated value: 286.22 | Target value: 286.22
Example #2.862174e+02 - Estimated value: 286.22 | Target value: 286.22
Example #2.862165e+02 - Estimated value: 286.22 | Target value: 286.22
Example #2.862156e+02 - Estimated value: 286.22 | Target value: 286.22
Example #2.862147e+02 - Estimated value: 286.21 | Target value: 286.21
Example #2.862138e+02 - Estimated value: 286.21 | Target value: 286.21
Example #2.862129e+02 - Estimated value: 286.21 | Target value: 286.21
Example #2.862120e+02 - Estimated value: 286.21 | Target value: 286.21
Example #2.862111e+02 - Estimated value: 286.21 | Target value: 286.21
Example #2.862102e+02 - Estimated value: 286.21 | Target value: 286.21
Example #2.862093e+02 - Estimated value: 286.21 | Target value: 286.21
Example #2.862084e+02 - Estimated value: 286.21 | Target value: 286.21
Example #2.862075e+02 - Estimated value: 286.21 | Target value: 286.21
Example #2.862066e+02 - Estimated value: 286.21 | Target value: 286.21
Example #2.862057e+02 - Estimated value: 286.21 | Target value: 286.21
Example #2.862048e+02 - Estimated value: 286.20 | Target value: 286.20
Example #2.862039e+02 - Estimated value: 286.20 | Target value: 286.20
Example #2.862029e+02 - Estimated value: 286.20 | Target value: 286.20
Example #2.862020e+02 - Estimated value: 286.20 | Target value: 286.20
Example #2.862011e+02 - Estimated value: 286.20 | Target value: 286.20
Example #2.862002e+02 - Estimated value: 286.20 | Target value: 286.20
Example #2.861993e+02 - Estimated value: 286.20 | Target value: 286.20
Example #2.861984e+02 - Estimated value: 286.20 | Target value: 286.20
Example #2.861975e+02 - Estimated value: 286.20 | Target value: 286.20
Example #2.861966e+02 - Estimated value: 286.20 | Target value: 286.20
Example #2.861957e+02 - Estimated value: 286.20 | Target value: 286.20
Example #2.861948e+02 - Estimated value: 286.19 | Target value: 286.19
Example #2.861939e+02 - Estimated value: 286.19 | Target value: 286.19
Example #2.861930e+02 - Estimated value: 286.19 | Target value: 286.19
Example #2.861921e+02 - Estimated value: 286.19 | Target value: 286.19
Example #2.861912e+02 - Estimated value: 286.19 | Target value: 286.19
Example #2.861903e+02 - Estimated value: 286.19 | Target value: 286.19
Example #2.861894e+02 - Estimated value: 286.19 | Target value: 286.19
Example #2.861885e+02 - Estimated value: 286.19 | Target value: 286.19
Example #2.861876e+02 - Estimated value: 286.19 | Target value: 286.19
Example #2.861867e+02 - Estimated value: 286.19 | Target value: 286.19
Example #2.861858e+02 - Estimated value: 286.19 | Target value: 286.19
Example #2.861849e+02 - Estimated value: 286.18 | Target value: 286.18
Example #2.861840e+02 - Estimated value: 286.18 | Target value: 286.18
Example #2.861831e+02 - Estimated value: 286.18 | Target value: 286.18
Example #2.861822e+02 - Estimated value: 286.18 | Target value: 286.18
Example #2.861813e+02 - Estimated value: 286.18 | Target value: 286.18
Example #2.861804e+02 - Estimated value: 286.18 | Target value: 286.18
Example #2.861795e+02 - Estimated value: 286.18 | Target value: 286.18
Example #2.861786e+02 - Estimated value: 286.18 | Target value: 286.18
Example #2.861777e+02 - Estimated value: 286.18 | Target value: 286.18
Example #2.861768e+02 - Estimated value: 286.18 | Target value: 286.18
Example #2.861759e+02 - Estimated value: 286.18 | Target value: 286.18
Example #2.861750e+02 - Estimated value: 286.17 | Target value: 286.17
Example #2.861741e+02 - Estimated value: 286.17 | Target value: 286.17
Example #2.861732e+02 - Estimated value: 286.17 | Target value: 286.17
Example #2.861723e+02 - Estimated value: 286.17 | Target value: 286.17
Example #2.861714e+02 - Estimated value: 286.17 | Target value: 286.17
Example #2.861705e+02 - Estimated value: 286.17 | Target value: 286.17
Example #2.861696e+02 - Estimated value: 286.17 | Target value: 286.17
Example #2.861687e+02 - Estimated value: 286.17 | Target value: 286.17
Example #2.861678e+02 - Estimated value: 286.17 | Target value: 232.00
Example #3 - Estimated value: 205.83 | Target value: 165.95
Example #1.736768e+02 - Estimated value: 172.18 | Target value: 172.47
Example #1.724110e+02 - Estimated value: 172.42 | Target value: 172.42
Example #1.724172e+02 - Estimated value: 172.42 | Target value: 172.42
Example #1.724142e+02 - Estimated value: 172.41 | Target value: 172.41
Example #1.724113e+02 - Estimated value: 172.41 | Target value: 172.41
Example #1.724083e+02 - Estimated value: 172.41 | Target value: 172.41
Example #1.724054e+02 - Estimated value: 172.40 | Target value: 172.40
Example #1.724025e+02 - Estimated value: 172.40 | Target value: 172.40
Example #1.723995e+02 - Estimated value: 172.40 | Target value: 172.40
Example #1.723966e+02 - Estimated value: 172.40 | Target value: 172.39
Example #1.723937e+02 - Estimated value: 172.39 | Target value: 172.39
Example #1.723907e+02 - Estimated value: 172.39 | Target value: 172.39
Example #1.723878e+02 - Estimated value: 172.39 | Target value: 172.39
Example #1.723849e+02 - Estimated value: 172.38 | Target value: 172.38
Example #1.723819e+02 - Estimated value: 172.38 | Target value: 172.38
Example #1.723790e+02 - Estimated value: 172.38 | Target value: 172.38
Example #1.723761e+02 - Estimated value: 172.38 | Target value: 172.37
Example #1.723731e+02 - Estimated value: 172.37 | Target value: 172.37
Example #1.723702e+02 - Estimated value: 172.37 | Target value: 172.37
Example #1.723673e+02 - Estimated value: 172.37 | Target value: 172.37
Example #1.723643e+02 - Estimated value: 172.36 | Target value: 172.36
Example #1.723614e+02 - Estimated value: 172.36 | Target value: 172.36
Example #1.723585e+02 - Estimated value: 172.36 | Target value: 172.36
Example #1.723555e+02 - Estimated value: 172.35 | Target value: 172.35
Example #1.723526e+02 - Estimated value: 172.35 | Target value: 172.35
Example #1.723497e+02 - Estimated value: 172.35 | Target value: 172.35
Example #1.723467e+02 - Estimated value: 172.35 | Target value: 172.34
Example #1.723438e+02 - Estimated value: 172.34 | Target value: 172.34
Example #1.723409e+02 - Estimated value: 172.34 | Target value: 172.34
Example #1.723380e+02 - Estimated value: 172.34 | Target value: 172.34
Example #1.723350e+02 - Estimated value: 172.33 | Target value: 172.33
Example #1.723321e+02 - Estimated value: 172.33 | Target value: 172.33
Example #1.723292e+02 - Estimated value: 172.33 | Target value: 172.33
Example #1.723263e+02 - Estimated value: 172.33 | Target value: 172.32
Example #1.723233e+02 - Estimated value: 172.32 | Target value: 172.32
Example #1.723204e+02 - Estimated value: 172.32 | Target value: 172.32
Example #1.723175e+02 - Estimated value: 172.32 | Target value: 172.32
Example #1.723145e+02 - Estimated value: 172.31 | Target value: 172.31
Example #1.723116e+02 - Estimated value: 172.31 | Target value: 172.31
Example #1.723087e+02 - Estimated value: 172.31 | Target value: 172.31
Example #1.723058e+02 - Estimated value: 172.30 | Target value: 172.30
Example #1.723029e+02 - Estimated value: 172.30 | Target value: 172.30
Example #1.722999e+02 - Estimated value: 172.30 | Target value: 172.30
Example #1.722970e+02 - Estimated value: 172.30 | Target value: 172.30
Example #1.722941e+02 - Estimated value: 172.29 | Target value: 172.29
Example #1.722912e+02 - Estimated value: 172.29 | Target value: 172.29
Example #1.722882e+02 - Estimated value: 172.29 | Target value: 172.29
Example #1.722853e+02 - Estimated value: 172.28 | Target value: 172.28
Example #1.722824e+02 - Estimated value: 172.28 | Target value: 172.28
Example #1.722795e+02 - Estimated value: 172.28 | Target value: 172.28
Example #1.722766e+02 - Estimated value: 172.28 | Target value: 172.27
Example #1.722736e+02 - Estimated value: 172.27 | Target value: 172.27
Example #1.722707e+02 - Estimated value: 172.27 | Target value: 172.27
Example #1.722678e+02 - Estimated value: 172.27 | Target value: 172.27
Example #1.722649e+02 - Estimated value: 172.26 | Target value: 172.26
Example #1.722620e+02 - Estimated value: 172.26 | Target value: 172.26
Example #1.722590e+02 - Estimated value: 172.26 | Target value: 172.26
Example #1.722561e+02 - Estimated value: 172.26 | Target value: 172.25
Example #1.722532e+02 - Estimated value: 172.25 | Target value: 172.25
Example #1.722503e+02 - Estimated value: 172.25 | Target value: 172.25
Example #1.722474e+02 - Estimated value: 172.25 | Target value: 172.25
Example #1.722445e+02 - Estimated value: 172.24 | Target value: 172.24
Example #1.722416e+02 - Estimated value: 172.24 | Target value: 172.24
Example #1.722386e+02 - Estimated value: 172.24 | Target value: 172.24
Example #1.722357e+02 - Estimated value: 172.23 | Target value: 172.23
Example #1.722328e+02 - Estimated value: 172.23 | Target value: 172.23
Example #1.722299e+02 - Estimated value: 172.23 | Target value: 172.23
Example #1.722270e+02 - Estimated value: 172.23 | Target value: 172.23
Example #1.722241e+02 - Estimated value: 172.22 | Target value: 172.22
Example #1.722212e+02 - Estimated value: 172.22 | Target value: 172.22
Example #1.722182e+02 - Estimated value: 172.22 | Target value: 172.22
Example #1.722153e+02 - Estimated value: 172.21 | Target value: 172.21
Example #1.722124e+02 - Estimated value: 172.21 | Target value: 172.21
Example #1.722095e+02 - Estimated value: 172.21 | Target value: 172.21
Example #1.722066e+02 - Estimated value: 172.21 | Target value: 172.20
Example #1.722037e+02 - Estimated value: 172.20 | Target value: 172.20
Example #1.722008e+02 - Estimated value: 172.20 | Target value: 172.20
Example #1.721979e+02 - Estimated value: 172.20 | Target value: 172.20
Example #1.721950e+02 - Estimated value: 172.19 | Target value: 172.19
Example #1.721921e+02 - Estimated value: 172.19 | Target value: 172.19
Example #1.721891e+02 - Estimated value: 172.19 | Target value: 172.19
Example #1.721862e+02 - Estimated value: 172.19 | Target value: 172.18
Example #1.721833e+02 - Estimated value: 172.18 | Target value: 172.18
Example #1.721804e+02 - Estimated value: 172.18 | Target value: 172.18
Example #1.721775e+02 - Estimated value: 172.18 | Target value: 172.18
Example #1.721746e+02 - Estimated value: 172.17 | Target value: 172.17
Example #1.721717e+02 - Estimated value: 172.17 | Target value: 172.17
Example #1.721688e+02 - Estimated value: 172.17 | Target value: 172.17
Example #1.721659e+02 - Estimated value: 172.16 | Target value: 172.16
Example #1.721630e+02 - Estimated value: 172.16 | Target value: 172.16
Example #1.721601e+02 - Estimated value: 172.16 | Target value: 172.16
Example #1.721572e+02 - Estimated value: 172.16 | Target value: 172.16
Example #1.721543e+02 - Estimated value: 172.15 | Target value: 172.15
Example #1.721514e+02 - Estimated value: 172.15 | Target value: 172.15
Example #1.721485e+02 - Estimated value: 172.15 | Target value: 172.15
Example #1.721456e+02 - Estimated value: 172.14 | Target value: 172.14
Example #1.721427e+02 - Estimated value: 172.14 | Target value: 172.14
Example #1.721398e+02 - Estimated value: 172.14 | Target value: 172.14
Example #1.721369e+02 - Estimated value: 172.14 | Target value: 172.13
Example #1.721340e+02 - Estimated value: 172.13 | Target value: 172.13
Example #1.721311e+02 - Estimated value: 172.13 | Target value: 172.13
Example #1.721282e+02 - Estimated value: 172.13 | Target value: 172.13
Example #1.721253e+02 - Estimated value: 172.12 | Target value: 172.12
Example #1.721224e+02 - Estimated value: 172.12 | Target value: 172.12
Example #1.721195e+02 - Estimated value: 172.12 | Target value: 172.12
Example #1.721166e+02 - Estimated value: 172.12 | Target value: 172.11
Example #1.721137e+02 - Estimated value: 172.11 | Target value: 172.11
Example #1.721108e+02 - Estimated value: 172.11 | Target value: 172.11
Example #1.721079e+02 - Estimated value: 172.11 | Target value: 172.11
Example #1.721050e+02 - Estimated value: 172.10 | Target value: 172.10
Example #1.721021e+02 - Estimated value: 172.10 | Target value: 172.10
Example #1.720992e+02 - Estimated value: 172.10 | Target value: 172.10
Example #1.720963e+02 - Estimated value: 172.10 | Target value: 172.09
Example #1.720934e+02 - Estimated value: 172.09 | Target value: 172.09
Example #1.720905e+02 - Estimated value: 172.09 | Target value: 172.09
Example #1.720876e+02 - Estimated value: 172.09 | Target value: 172.09
Example #1.720847e+02 - Estimated value: 172.08 | Target value: 172.08
Example #1.720818e+02 - Estimated value: 172.08 | Target value: 172.08
Example #1.720789e+02 - Estimated value: 172.08 | Target value: 172.08
Example #1.720760e+02 - Estimated value: 172.08 | Target value: 172.07
Example #1.720731e+02 - Estimated value: 172.07 | Target value: 172.07
Example #1.720702e+02 - Estimated value: 172.07 | Target value: 172.07
Example #1.720674e+02 - Estimated value: 172.07 | Target value: 172.07
Example #1.720645e+02 - Estimated value: 172.06 | Target value: 172.06
Example #1.720616e+02 - Estimated value: 172.06 | Target value: 172.06
Example #1.720587e+02 - Estimated value: 172.06 | Target value: 172.06
Example #1.720558e+02 - Estimated value: 172.05 | Target value: 172.05
Example #1.720529e+02 - Estimated value: 172.05 | Target value: 172.05
Example #1.720500e+02 - Estimated value: 172.05 | Target value: 172.05
Example #1.720471e+02 - Estimated value: 172.05 | Target value: 172.05
Example #1.720442e+02 - Estimated value: 172.04 | Target value: 172.04
Example #1.720413e+02 - Estimated value: 172.04 | Target value: 172.04
Example #1.720385e+02 - Estimated value: 172.04 | Target value: 172.04
Example #1.720356e+02 - Estimated value: 172.03 | Target value: 172.03
Example #1.720327e+02 - Estimated value: 172.03 | Target value: 172.03
Example #1.720298e+02 - Estimated value: 172.03 | Target value: 172.03
Example #1.720269e+02 - Estimated value: 172.03 | Target value: 172.02
Example #1.720240e+02 - Estimated value: 172.02 | Target value: 172.02
Example #1.720211e+02 - Estimated value: 172.02 | Target value: 172.02
Example #1.720183e+02 - Estimated value: 172.02 | Target value: 172.02
Example #1.720154e+02 - Estimated value: 172.01 | Target value: 172.01
Example #1.720125e+02 - Estimated value: 172.01 | Target value: 172.01
Example #1.720096e+02 - Estimated value: 172.01 | Target value: 172.01
Example #1.720067e+02 - Estimated value: 172.01 | Target value: 172.00
Example #1.720038e+02 - Estimated value: 172.00 | Target value: 172.00
Example #1.720010e+02 - Estimated value: 172.00 | Target value: 172.00
Example #1.719981e+02 - Estimated value: 172.00 | Target value: 172.00
Example #1.719952e+02 - Estimated value: 171.99 | Target value: 171.99
Example #1.719923e+02 - Estimated value: 171.99 | Target value: 171.99
Example #1.719894e+02 - Estimated value: 171.99 | Target value: 171.99
Example #1.719865e+02 - Estimated value: 171.99 | Target value: 171.98
Example #1.719837e+02 - Estimated value: 171.98 | Target value: 171.98
Example #1.719808e+02 - Estimated value: 171.98 | Target value: 171.98
Example #1.719779e+02 - Estimated value: 171.98 | Target value: 171.98
Example #1.719750e+02 - Estimated value: 171.97 | Target value: 171.97
Example #1.719721e+02 - Estimated value: 171.97 | Target value: 171.97
Example #1.719693e+02 - Estimated value: 171.97 | Target value: 171.97
Example #1.719664e+02 - Estimated value: 171.97 | Target value: 171.96
Example #1.719635e+02 - Estimated value: 171.96 | Target value: 171.96
Example #1.719606e+02 - Estimated value: 171.96 | Target value: 171.96
Example #1.719578e+02 - Estimated value: 171.96 | Target value: 171.96
Example #1.719549e+02 - Estimated value: 171.95 | Target value: 171.95
Example #1.719520e+02 - Estimated value: 171.95 | Target value: 171.95
Example #1.719491e+02 - Estimated value: 171.95 | Target value: 171.95
Example #1.719462e+02 - Estimated value: 171.95 | Target value: 171.94
Example #1.719434e+02 - Estimated value: 171.94 | Target value: 171.94
Example #1.719405e+02 - Estimated value: 171.94 | Target value: 171.94
Example #1.719376e+02 - Estimated value: 171.94 | Target value: 171.94
Example #1.719347e+02 - Estimated value: 171.93 | Target value: 171.93
Example #1.719319e+02 - Estimated value: 171.93 | Target value: 171.93
Example #1.719290e+02 - Estimated value: 171.93 | Target value: 171.93
Example #1.719261e+02 - Estimated value: 171.93 | Target value: 171.92
Example #1.719233e+02 - Estimated value: 171.92 | Target value: 171.92
Example #1.719204e+02 - Estimated value: 171.92 | Target value: 171.92
Example #1.719175e+02 - Estimated value: 171.92 | Target value: 171.92
Example #1.719146e+02 - Estimated value: 171.91 | Target value: 171.91
Example #1.719118e+02 - Estimated value: 171.91 | Target value: 171.91
Example #1.719089e+02 - Estimated value: 171.91 | Target value: 171.91
Example #1.719060e+02 - Estimated value: 171.91 | Target value: 171.90
Example #1.719032e+02 - Estimated value: 171.90 | Target value: 171.90
Example #1.719003e+02 - Estimated value: 171.90 | Target value: 171.90
Example #1.718974e+02 - Estimated value: 171.90 | Target value: 171.90
Example #1.718946e+02 - Estimated value: 171.89 | Target value: 171.89
Example #1.718917e+02 - Estimated value: 171.89 | Target value: 171.89
Example #1.718888e+02 - Estimated value: 171.89 | Target value: 171.89
Example #1.718859e+02 - Estimated value: 171.88 | Target value: 171.88
Example #1.718831e+02 - Estimated value: 171.88 | Target value: 171.88
Example #1.718802e+02 - Estimated value: 171.88 | Target value: 171.88
Example #1.718773e+02 - Estimated value: 171.88 | Target value: 171.88
Example #1.718745e+02 - Estimated value: 171.87 | Target value: 171.87
Example #1.718716e+02 - Estimated value: 171.87 | Target value: 171.87
Example #1.718688e+02 - Estimated value: 171.87 | Target value: 171.87
Example #1.718659e+02 - Estimated value: 171.86 | Target value: 171.86
Example #1.718630e+02 - Estimated value: 171.86 | Target value: 171.86
Example #1.718602e+02 - Estimated value: 171.86 | Target value: 171.86
Example #1.718573e+02 - Estimated value: 171.86 | Target value: 171.86
Example #1.718544e+02 - Estimated value: 171.85 | Target value: 171.85
Example #1.718516e+02 - Estimated value: 171.85 | Target value: 171.85
Example #1.718487e+02 - Estimated value: 171.85 | Target value: 171.85
Example #1.718458e+02 - Estimated value: 171.84 | Target value: 171.84
Example #1.718430e+02 - Estimated value: 171.84 | Target value: 171.84
Example #1.718401e+02 - Estimated value: 171.84 | Target value: 171.84
Example #1.718373e+02 - Estimated value: 171.84 | Target value: 171.84
Example #1.718344e+02 - Estimated value: 171.83 | Target value: 171.83
Example #1.718315e+02 - Estimated value: 171.83 | Target value: 171.83
Example #1.718287e+02 - Estimated value: 171.83 | Target value: 171.83
Example #1.718258e+02 - Estimated value: 171.82 | Target value: 171.82
Example #1.718230e+02 - Estimated value: 171.82 | Target value: 171.82
Example #1.718201e+02 - Estimated value: 171.82 | Target value: 171.82
Example #1.718172e+02 - Estimated value: 171.82 | Target value: 171.82
Example #1.718144e+02 - Estimated value: 171.81 | Target value: 171.81
Example #1.718115e+02 - Estimated value: 171.81 | Target value: 171.81
Example #1.718087e+02 - Estimated value: 171.81 | Target value: 171.81
Example #1.718058e+02 - Estimated value: 171.80 | Target value: 171.80
Example #1.718030e+02 - Estimated value: 171.80 | Target value: 171.80
Example #1.718001e+02 - Estimated value: 171.80 | Target value: 171.80
Example #1.717972e+02 - Estimated value: 171.80 | Target value: 171.80
Example #1.717944e+02 - Estimated value: 171.79 | Target value: 171.79
Example #1.717915e+02 - Estimated value: 171.79 | Target value: 171.79
Example #1.717887e+02 - Estimated value: 171.79 | Target value: 171.79
Example #1.717858e+02 - Estimated value: 171.78 | Target value: 171.78
Example #1.717830e+02 - Estimated value: 171.78 | Target value: 171.78
Example #1.717801e+02 - Estimated value: 171.78 | Target value: 171.78
Example #1.717773e+02 - Estimated value: 171.78 | Target value: 171.78
Example #1.717744e+02 - Estimated value: 171.77 | Target value: 171.77
Example #1.717716e+02 - Estimated value: 171.77 | Target value: 171.77
Example #1.717687e+02 - Estimated value: 171.77 | Target value: 171.77
Example #1.717659e+02 - Estimated value: 171.76 | Target value: 171.76
Example #1.717630e+02 - Estimated value: 171.76 | Target value: 171.76
Example #1.717602e+02 - Estimated value: 171.76 | Target value: 171.76
Example #1.717573e+02 - Estimated value: 171.76 | Target value: 171.76
Example #1.717545e+02 - Estimated value: 171.75 | Target value: 171.75
Example #1.717516e+02 - Estimated value: 171.75 | Target value: 171.75
Example #1.717488e+02 - Estimated value: 171.75 | Target value: 171.75
Example #1.717459e+02 - Estimated value: 171.74 | Target value: 171.74
Example #1.717431e+02 - Estimated value: 171.74 | Target value: 171.74
Example #1.717402e+02 - Estimated value: 171.74 | Target value: 171.74
Example #1.717374e+02 - Estimated value: 171.74 | Target value: 171.74
Example #1.717345e+02 - Estimated value: 171.73 | Target value: 171.73
Example #1.717317e+02 - Estimated value: 171.73 | Target value: 171.73
Example #1.717288e+02 - Estimated value: 171.73 | Target value: 171.73
Example #1.717260e+02 - Estimated value: 171.73 | Target value: 171.72
Example #1.717231e+02 - Estimated value: 171.72 | Target value: 171.72
Example #1.717203e+02 - Estimated value: 171.72 | Target value: 171.72
Example #1.717174e+02 - Estimated value: 171.72 | Target value: 171.72
Example #1.717146e+02 - Estimated value: 171.71 | Target value: 171.71
Example #1.717118e+02 - Estimated value: 171.71 | Target value: 171.71
Example #1.717089e+02 - Estimated value: 171.71 | Target value: 171.71
Example #1.717061e+02 - Estimated value: 171.71 | Target value: 171.70
Example #1.717032e+02 - Estimated value: 171.70 | Target value: 171.70
Example #1.717004e+02 - Estimated value: 171.70 | Target value: 171.70
Example #1.716975e+02 - Estimated value: 171.70 | Target value: 171.70
Example #1.716947e+02 - Estimated value: 171.69 | Target value: 171.69
Example #1.716919e+02 - Estimated value: 171.69 | Target value: 171.69
Example #1.716890e+02 - Estimated value: 171.69 | Target value: 171.69
Example #1.716862e+02 - Estimated value: 171.69 | Target value: 171.68
Example #1.716833e+02 - Estimated value: 171.68 | Target value: 171.68
Example #1.716805e+02 - Estimated value: 171.68 | Target value: 171.68
Example #1.716777e+02 - Estimated value: 171.68 | Target value: 171.68
Example #1.716748e+02 - Estimated value: 171.67 | Target value: 171.67
Example #1.716720e+02 - Estimated value: 171.67 | Target value: 171.67
Example #1.716691e+02 - Estimated value: 171.67 | Target value: 171.67
Example #1.716663e+02 - Estimated value: 171.67 | Target value: 171.66
Example #1.716635e+02 - Estimated value: 171.66 | Target value: 171.66
Example #1.716606e+02 - Estimated value: 171.66 | Target value: 171.66
Example #1.716578e+02 - Estimated value: 171.66 | Target value: 171.66
Example #1.716549e+02 - Estimated value: 171.65 | Target value: 171.65
Example #1.716521e+02 - Estimated value: 171.65 | Target value: 171.65
Example #1.716493e+02 - Estimated value: 171.65 | Target value: 171.65
Example #1.716464e+02 - Estimated value: 171.65 | Target value: 171.64
Example #1.716436e+02 - Estimated value: 171.64 | Target value: 171.64
Example #1.716408e+02 - Estimated value: 171.64 | Target value: 171.64
Example #1.716379e+02 - Estimated value: 171.64 | Target value: 171.64
Example #1.716351e+02 - Estimated value: 171.63 | Target value: 171.63
Example #1.716323e+02 - Estimated value: 171.63 | Target value: 171.63
Example #1.716294e+02 - Estimated value: 171.63 | Target value: 171.63
Example #1.716266e+02 - Estimated value: 171.63 | Target value: 171.62
Example #1.716238e+02 - Estimated value: 171.62 | Target value: 171.62
Example #1.716209e+02 - Estimated value: 171.62 | Target value: 171.62
Example #1.716181e+02 - Estimated value: 171.62 | Target value: 171.62
Example #1.716153e+02 - Estimated value: 171.61 | Target value: 171.61
Example #1.716124e+02 - Estimated value: 171.61 | Target value: 171.61
Example #1.716096e+02 - Estimated value: 171.61 | Target value: 171.61
Example #1.716068e+02 - Estimated value: 171.61 | Target value: 171.60
Example #1.716040e+02 - Estimated value: 171.60 | Target value: 171.60
Example #1.716011e+02 - Estimated value: 171.60 | Target value: 171.60
Example #1.715983e+02 - Estimated value: 171.60 | Target value: 171.60
Example #1.715955e+02 - Estimated value: 171.59 | Target value: 171.59
Example #1.715926e+02 - Estimated value: 171.59 | Target value: 171.59
Example #1.715898e+02 - Estimated value: 171.59 | Target value: 171.59
Example #1.715870e+02 - Estimated value: 171.59 | Target value: 171.59
Example #1.715842e+02 - Estimated value: 171.58 | Target value: 171.58
Example #1.715813e+02 - Estimated value: 171.58 | Target value: 171.58
Example #1.715785e+02 - Estimated value: 171.58 | Target value: 171.58
Example #1.715757e+02 - Estimated value: 171.57 | Target value: 171.57
Example #1.715729e+02 - Estimated value: 171.57 | Target value: 171.57
Example #1.715700e+02 - Estimated value: 171.57 | Target value: 171.57
Example #1.715672e+02 - Estimated value: 171.57 | Target value: 171.57
Example #1.715644e+02 - Estimated value: 171.56 | Target value: 171.56
Example #1.715616e+02 - Estimated value: 171.56 | Target value: 171.56
Example #1.715587e+02 - Estimated value: 171.56 | Target value: 171.56
Example #1.715559e+02 - Estimated value: 171.55 | Target value: 171.55
Example #1.715531e+02 - Estimated value: 171.55 | Target value: 171.55
Example #1.715503e+02 - Estimated value: 171.55 | Target value: 171.55
Example #1.715474e+02 - Estimated value: 171.55 | Target value: 171.55
Example #1.715446e+02 - Estimated value: 171.54 | Target value: 171.54
Example #1.715418e+02 - Estimated value: 171.54 | Target value: 171.54
Example #1.715390e+02 - Estimated value: 171.54 | Target value: 171.54
Example #1.715362e+02 - Estimated value: 171.54 | Target value: 171.53
Example #1.715333e+02 - Estimated value: 171.53 | Target value: 171.53
Example #1.715305e+02 - Estimated value: 171.53 | Target value: 171.53
Example #1.715277e+02 - Estimated value: 171.53 | Target value: 171.53
Example #1.715249e+02 - Estimated value: 171.52 | Target value: 171.52
Example #1.715221e+02 - Estimated value: 171.52 | Target value: 171.52
Example #1.715192e+02 - Estimated value: 171.52 | Target value: 171.52
Example #1.715164e+02 - Estimated value: 171.52 | Target value: 171.51
Example #1.715136e+02 - Estimated value: 171.51 | Target value: 171.51
Example #1.715108e+02 - Estimated value: 171.51 | Target value: 171.51
Example #1.715080e+02 - Estimated value: 171.51 | Target value: 171.51
Example #1.715051e+02 - Estimated value: 171.50 | Target value: 171.50
Example #1.715023e+02 - Estimated value: 171.50 | Target value: 171.50
Example #1.714995e+02 - Estimated value: 171.50 | Target value: 171.50
Example #1.714967e+02 - Estimated value: 171.50 | Target value: 171.49
Example #1.714939e+02 - Estimated value: 171.49 | Target value: 171.49
Example #1.714911e+02 - Estimated value: 171.49 | Target value: 171.49
Example #1.714883e+02 - Estimated value: 171.49 | Target value: 171.49
Example #1.714854e+02 - Estimated value: 171.48 | Target value: 171.48
Example #1.714826e+02 - Estimated value: 171.48 | Target value: 171.48
Example #1.714798e+02 - Estimated value: 171.48 | Target value: 171.48
Example #1.714770e+02 - Estimated value: 171.48 | Target value: 171.48
Example #1.714742e+02 - Estimated value: 171.47 | Target value: 171.47
Example #1.714714e+02 - Estimated value: 171.47 | Target value: 171.47
Example #1.714686e+02 - Estimated value: 171.47 | Target value: 178.00
```

<a id="H_3DD18DCA"></a>

# Gradient Descent in Practice

Let's now take a look at some techniques that make gradient descent (GD) work much better (more efficient). First, let's now investigate a technique called f**eature scaling**, which will enable gradient descent to run faster.

-  If we take look at the relationship between the range/magnitude/size of a feature (how big are the numbers in a feature) and range/magnitude/size of its associated parameter. 
-  Let's take the "prediction of house prices" problem as our example, where we will assume that we have only two features 

The linear regression problem could be formulated as follows:


 $\hat{p} =w\_1 x\_1 +w\_2 x\_2 +b={\mathbf{w}}^{\textrm{T}} \mathbf{x}+b$, where the parameter $\hat{p}$ represents the predicted price. The parameters $\mathbf{w}$, $\mathbf{x}$ and $b$ are the linear regression weights parameter, features vector and bias parameter, respectively. Note that we represent each feature corresponds to the $j^{\textrm{th}}$ column of the training matrix $\mathbf{X}$, where ${\mathbf{x}}_j =[x_j^{(1)} \,x_j^{(2)} \,\cdots \,x_j^{(m)} ]^{\textrm{T}}$ and $j\in \lbrace 1,2,\cdots ,n\rbrace$. 


Spefically, 

-  $x\_1$ represents the size of a house (square feet, range: [300-2000]) 
-  $x\_2$ is the number of bedrooms (range: [0 5]). 

For this example, we can say that $x\_1$ takes realtively large range of values and $x\_2$ takes relatively small range of values. Now, let's take one training example:


House: $x\_1 =2000$, $x\_2 =5$, price = \$500K for this training example what do you think the reasonable values for the size of the parameters $w\_1$ and $w\_2$? 


&nbsp;&nbsp;&nbsp;&nbsp; 1) Let's look at one set of parameters:


 $w_1 =50$, $w_2 =0.1$, $b=50$. In this case, the estimated price becomes, $\hat{p} =w_1 x_1 +w_2 x_2 +b=50*2000+0.1*5+50=$ 100,050.5\textrm{K}$. It is clear that the estimated price is quite far from the actual price which was \$500K. This is not a good set of parameter choices for $w\_1$ and $w\_2$.&nbsp;&nbsp;&nbsp;&nbsp; 2) Let's take a look at another possibilityHouse: $w_1 =0.1 $, $ w_2 =50$, $b=50$, where the values of the weights are flipped. In this case, the estimated price becomes, $\hat{p} =w_1 x_1 +w_2 x_2 +b=0.1*2000+50*5+50= $500\textrm{K}$, which is a more reasonable estimate.

-  It is important to note that when a range of a feature is large ( $x\_1$ in this case) it is more likely that a "good model" will choose a smaller value (such as $w\_1 =0.1$ ). Likewise, when the range of a feature is small then a "good model" will choose a larger weight value (such as $w\_2 =50$ ). 
<a id="H_B4802151"></a>

## **Feature Scaling in Practice**

**Question: How does this relate to gradient descent?**

-  Let's take a look at the scatter plot of the features, $x\_1$ and $x\_2$ and contour plot. 

![image_0.png](./supervised_learning_week2_part1_media/image_0.png)


If we plot the training data, we can see that the $x\_1$ is in the much larger scale/range compared to $x\_2$. 

-  Next, let's look how the cost function might look in the contour plot. 

![image_1.png](./supervised_learning_week2_part1_media/image_1.png)


We can see that in the contour plot $w\_1$ has much narrower range (say between 0 and 1) compared to $w\_2$ (say between 10 and 100). Thus, the contours form ovals/ellipses as they are in shorter one side and longer in the other.


**This is because very small change in** $w\_1$ **can have a very large impact on the estimated price (** $\hat{p}$ **), and that corresponds to a very large impact on the cost** $J$ **as** $w\_1$ **tends to be multiplied with a very large number** $x\_1$ **(size in sq. feet). On the contrary, it takes significantly larger change in** $w\_2$ **to have a very large impact on the** $\hat{p}$ **and** $J$ **as it tends to be multiplied with a small number** $x\_2$ **(# of bedrooms).**

||||
| :-- | :-- | :-- |
|  | size of the feature $x\_j$  | size of the parameter $w\_j$   |
| size in sq. feet  | Large  | Small   |
| # of bedrooms  | Small  | Large   |


The contour plot above is where what we will be dealing with if we decide to run the gradient descent algorithm on the example as is. As the cost function, $J(\mathbf{w},b)$, w.r.t parameters, $w\_1$ and $w\_2$, is tall and slim, gradient descent algorithm might be bouncing back and forth until it finds the solution, which makes the searching for the optimal point much longer and increases the possibility of divergence.


![image_2.png](./supervised_learning_week2_part1_media/image_2.png)![image_3.png](./supervised_learning_week2_part1_media/image_3.png)


In situations like this, useful thing to do is to scale the features, this means that some transformation in the input training data is employed. Consequently, the range for the parameters $x\_1$ and $x\_2$ will be limited to be in 0 and 1. The corresponding data points and cost function plot after the feature scaling becomes,


![image_4.png](./supervised_learning_week2_part1_media/image_4.png)![image_5.png](./supervised_learning_week2_part1_media/image_5.png)


Please note from the transformed data yields a contour plot that is more uniform and symmetrical (shapes more like circles). As a result, gradient descent can find a much more direct path to the global minimum compared to the tall and slim cost function in untransformed training data.


Recap: When you have different features with different range of values, it can cause the gradient descent algorithm to operate slowly. Rescaling the features into a comparable scale makes gradient descent run much faster. 


Question: Why do we need feature scaling?

-  It makes gradient descent algorithm operates faster 
-  It prevents the optimization from getting stuck in local optima 
-  It gives a better error/cost surface shape 
-  Machine learning algorithms just see numbers and doesn't know what the numbers represent — if there is a vast difference in the range say few ranging in thousands and few ranging in the tens, and the model might make the underlying assumption that higher ranging numbers have superiority of some sort. Thus, these more significant numbers start playing a more decisive role while training the model.  

For instance,

||||
| :-- | :-- | :-- |
| Item  | Weight (g)  | Price (\$)   |
| Orange  | 15  | 1   |
| Apple  | 18  | 3   |
| Banana  | 12  | 2   |
| Grapes  | 10  | 5   |


Normally, weight cannot be compared with price as it won't yield any meaningful result, however, because of the large numbers in the weight column algorithm might assume that the weight is more important feature than the price. However, if we convert the weights to kg, then price becomes the more important feature.


To avoid that, we can employ feature scaling, where one significant number doesn’t impact the model just because of their large magnitude.


**Question: But how do we actually rescale features into a comparable scale?**


If we look at the data below, where the features $x_1$ (room size in sq. feet) and $x_2$ (number of bedrooms) is in the range of $[300\,\,\,2000] $ and $ [0\,\,\,5]$, respectively.


![image_6.png](./supervised_learning_week2_part1_media/image_6.png)


**1) Maximum value scaling:** One way to scale is to take each $x\_1$ and $x\_2$ value is to divide them into the maximum value that each feature could reach,

 $$ {\hat{\mathbf{x}} }_j =\frac{{\mathbf{x}}_j }{\max \lbrace {\mathbf{x}}_j \rbrace } $$ 

where $i\in \lbrace 1,2,\cdots ,m\rbrace$ and $j\in \lbrace 1,2,\cdots ,n\rbrace$. Thus, the scaled $x\_1$ and $x\_2$ will be in the range of $[0.15\,\,\,1] $ and $ [0\,\,\,1]$, respectively.


![image_7.png](./supervised_learning_week2_part1_media/image_7.png)![image_8.png](./supervised_learning_week2_part1_media/image_8.png)


**2) Min\-max scaling:** Min\-max scaling is one of the simplest methods to rescaling features into the interval $[a\,\,\,b]$ in its generalized form. To implement the min\-max scaling we need the following operation.

 $$ {\hat{\mathbf{x}} }_j =\frac{{\mathbf{x}}_j -\min \lbrace {\mathbf{x}}_j \rbrace }{\max \lbrace {\mathbf{x}}_j \rbrace -\min \lbrace {\mathbf{x}}_j \rbrace }(b-a)+a $$ 

![image_9.png](./supervised_learning_week2_part1_media/image_9.png)![image_10.png](./supervised_learning_week2_part1_media/image_10.png)


**3) Mean value scaling:** Similar to the min\-max scaling, you can also use the mean value for scaling the feature vector, which is called "mean normalization". Mean normalization rescales the datapoints in a way that they are symmetrically scattered around the origin,

 $$ {\hat{\mathbf{x}} }_j =\frac{{\mathbf{x}}_j -\mu_{{\mathbf{x}}_j } }{\max \lbrace {\mathbf{x}}_j \rbrace -\min \lbrace {\mathbf{x}}_j \rbrace } $$ 

where $i\in \lbrace 1,2,\cdots ,m\rbrace$ and $j\in \lbrace 1,2,\cdots ,n\rbrace$. The mean value of the each feature could be calculated as $\mu_j =\textrm{E}\lbrace x_j \rbrace$. If we assume that the mean value $x_1$ is $\mu_1 =600$, then the $x_1$ will be in the range of $[-0.18\,\,\,0.82]$. Similarly, if the mean value of the feature $x_2$ is $\mu_2 =2.3$, the $x_2$ will be in the range of $[-0.46\,\,\,0.54]$.


![image_11.png](./supervised_learning_week2_part1_media/image_11.png)![image_12.png](./supervised_learning_week2_part1_media/image_12.png)


**4) Z\-score normalization:** There is one last normalization method called z\-score normalization, which utilizes mean and standard deviation, to standardize each feature (with the assumption of each feature is Gaussian distributed), ${\hat{\mathbf{x}} }_j =\frac{{\mathbf{x}}_j -\mu_{{\mathbf{x}}_j } }{\sigma_{{\mathbf{x}}_j } }$, where $\mu_{{\mathbf{x}}_j } =\textrm{E}\lbrace {\mathbf{x}}_j \rbrace $, $ \sigma_j^2 =\textrm{E}\lbrace ({\mathbf{x}}_j -\mu_{{\mathbf{x}}_j } )^2 \rbrace$ and $i\in \lbrace 1,2,\cdots ,m\rbrace$ and $j\in \lbrace 1,2,\cdots ,n\rbrace$. For the feature $x_1$ with the mean and stadard deviation of $\mu_1 =600 $ and $ \sigma_1 =450$, the range of the $x_1$ becomes $[-0,67\,\,\,3.1]$. Similarly, for the feature $x_2$ with the mean and stadard deviation of $\mu_2 =2.3 $ and $ \sigma_2 =1.4$, the range of the $x_2$ becomes $[-1.6\,\,\,1.9]$.


![image_13.png](./supervised_learning_week2_part1_media/image_13.png)![image_14.png](./supervised_learning_week2_part1_media/image_14.png)


As a rule of thumb in feature scaling, aim for about $-1\le x_j \le 1$ for each feature $x_j$. These \-1 and 1 values could be a little bit loose depending on the dataset and the application. For example,

- $-1\le x_j \le 1 $, $ -3\le x_j \le 3 $, $ -0.3\le x_j \le 0.3$ $\Longrightarrow$ Acceptable range 
-  $0\le x_j \le 3$ $\Longrightarrow$ OK, no scaling needed 
-  $-2\le x_j \le 0.5$ $\Longrightarrow$ OK, no scaling needed 
-  $-100\le x_j \le 100$ $\Longrightarrow$ Too large, scaling needed 
-  $-0.001\le x_j \le 0.001$ $\Longrightarrow$ Too small, scaling needed 
-  $98.6\le x_j \le 105$ $\Longrightarrow$ Too large, scaling needed 

**Question: With feature scaling, how are we going to know that the gradient descent algorithm finds the global minimum (converging)?**


When running gradient descent, how can we be sure that it is converging (the algortihm is getting closer and closer to the global minimum of the cost function)? One of the key choices in gradient descent algorithm,

-  $\displaystyle w_j =w_j -\alpha \frac{1}{m}\sum_{i=1}^m \left({\mathbf{w}}^{\textrm{T}} {\mathbf{x}}^{(i)} +b-y^{(i)} \right)\cdot x_j^{(i)}$ 
-  $b=b-\alpha \frac{1}{m}\sum_{i=1}^m \left({\mathbf{w}}^{\textrm{T}} {\mathbf{x}}^{(i)} \right)+b-y^{(i)}$, where $i\in \lbrace 1,2,\cdots ,m\rbrace$ and $j\in \lbrace 1,2,\cdots ,n\rbrace$ 

is the correct selection of the learning rate ( $\alpha$ ). Recall that the main objective of the gradient algorithm is: $\min_{\mathbf{w},b} J(\mathbf{w},b)$. Therefore, if we plot the cost function ( $J$ ) w.r.t the number of iterations, where each iteration means after the simultaneous updates of the parameters $\mathbf{w}$ and $b$.


![image_15.png](./supervised_learning_week2_part1_media/image_15.png)


This figure is called the **learning curve**. If the gradient descent is working properly, the cost ( $J$ ) should **decrease** after every iteration. If $J$ **ever** increases from one iteration to other, that might mean that either $\alpha$ is chosen poorly (usually means that $\alpha$ is too large) or there might be a potential bug in the code. It is also important to note that the cost ( $J$ ) is not decreasing much between the iterations 400 and 500, this means that the gradient descent is more or less **converged**.


Another way to decide whether your model is done training or not is the **automatic convergence test**,

-  Let $\epsilon$ be $10^{-3}$ 
-  If $J(\mathbf{w},b)$ decreases by $\le \epsilon$ in one iteration, **declare convergence** (hopefully found parameters $\mathbf{w}$ and $b$ to be close to the global minimum). 

**Let's now learn about how to choose an appropriate learning rate.**


The gradient descent based learning algorithm will run much better with the selection of the appropriate learning rate. If it is too small, it might converge but run very slowly. However, if it is too large it might not even converge. It is important to note that with a small enough $\alpha$, the cost function $J(\mathbf{w},b)$ should **decrease** on every iteration. If we set learning rate to be a very small value and still the cost function increases over number of iterations, it is a strong indicator of an implementation bug. 

-  As a rule of thumb. values of learning rate to try: $\alpha \in \lbrace 0.001,0.003,0.01,0.03,0.1,0.3,1\rbrace$, where each item is roughly 3 times larger than the previous one. 
<a id="H_522CB13A"></a>

Let's have a practice to understand the concepts explanied above!


The goal of the practice is as follows:

1.  Explore the impact of learning rate on the gradient descent (GD) performance.
2. Explore the impact of feature scaling on the GD performance.

Similar to the previous practices, we will use the housing price prediction dataset.

```matlab
% Load House Prices Dataset
dataset = importdata('house_prices_dataset.txt');
X_train = dataset(:,1:4);
y_train = dataset(:,end);
disp(['Size X_train: [' num2str(size(X_train)) ']'])
```

```matlabTextOutput
Size X_train: [99   4]
```

```matlab
disp(['Size y_train: [' num2str(size(y_train)) ']'])
```

```matlabTextOutput
Size y_train: [99   1]
```

```matlab
X_train
```

```matlabTextOutput
X_train = 99x4
        1244           3           1          64
        1947           3           2          17
        1725           3           2          42
        1959           3           2          15
        1314           2           1          14
         864           2           1          66
        1836           3           1          17
        1026           3           1          43
        3194           4           2          87
         788           2           1          80
        1200           2           2          17
        1557           2           1          18
        1430           3           1          20
        1220           2           1          15
        1092           2           1          64

```


Let's visualize the dataset and its features by plotting each feature against the price

```matlab
plotDatasetAgainstFeatures(X_train,y_train,["size (sqft)","num bedrooms","num floors","age"],"Price ($1000s)")
```

![figure_1.png](./supervised_learning_week2_part1_media/figure_1.png)

Plotting the dataset features w.r.t. the targets gives us some hint of which features play a significant role on price. For instance, number of bedrooms and floors don't have a strong impact on price. However, we can infer that larger and newer houses have higher prices compared to smaller and older houses.


Let's display and compare the peak-to-peak range between the original and z-score normalized data

```matlab
mu = mean(X_train);
sigma = std(X_train);
X_mean = X_train - mu;
X_norm = (X_train-mu)./sigma;

figure;
tiledlayout(3,1,"TileSpacing","loose")

nexttile
scatter(X_train(:,1),X_train(:,4),'MarkerFaceColor','flat')
title("Unnormalized Input")
xlabel("size (sqft)")
ylabel("age")
axis equal

nexttile
scatter(X_mean(:,1),X_mean(:,4),'MarkerFaceColor','flat')
title("$X-\mu$","Interpreter","latex")
xlabel("size (sqft)")
ylabel("age")
axis equal

nexttile
scatter(X_norm(:,1),X_norm(:,4),'MarkerFaceColor','flat')
title("$(X-\mu)/\sigma$","Interpreter","latex")
xlabel("size (sqft)")
ylabel("age")
axis equal
```

![figure_2.png](./supervised_learning_week2_part1_media/figure_2.png)

```matlab

disp(['Peak-to-peak range of each column (original): [' num2str(max(X_train) - min(X_train)) ']'])
```

```matlabTextOutput
Peak-to-peak range of each column (original): [2406     4     1    95]
```

```matlab
disp(['Peak-to-peak range of each column (normalized): [' num2str(max(X_norm) - min(X_norm)) ']'])
```

```matlabTextOutput
Peak-to-peak range of each column (normalized): [5.8157      6.1042      2.0459      3.6667]
```


 Let's visualize the distribution of some features before and after the normalization

```matlab
featureLabels = ["size (sqft)","num bedrooms","num floors","age"];
f = figure;
T = tiledlayout(1,length(featureLabels));
for i = 1:length(featureLabels)
    nexttile
    h = histfit(X_train(:,i));
    xlabel(featureLabels(i))
    ylabel("Relative frequency")
end
title(T,"Feature distribution before the input normalization")
f.Position = [0 0 700 300];
```

![figure_3.png](./supervised_learning_week2_part1_media/figure_3.png)

```matlab
f = figure;
T = tiledlayout(1,length(featureLabels));
for i = 1:length(featureLabels)
    nexttile
    h = histfit(X_norm(:,i));
    xlabel(featureLabels(i))
    ylabel("Relative frequency")
end
title(T,"Feature distribution after the input normalization")
f.Position = [0 0 700 300];
```

![figure_4.png](./supervised_learning_week2_part1_media/figure_4.png)

It is important to note from above figures that the all the features are centered around 0 and they also share the same range which is roughly [-2 2]. Let's run our gradient descent algorithm with the normalized input and significantly larger lerning rate to see the input normalization effect.

```matlab
% BGD settings
alpha = 1.0e-1; % Learning rate
condStop = 1e-3; % Stopping condition
maxIter = 1000; % Maximum number of iterations
verbose = false;
verboseFreq = 100;

% Run BGD algoritm and print the results
 
tStart = tic;
[w_vec,b_vec,J_hist] = hBatchGradientDescentMV(X_norm, y_train, alpha, condStop, ...
                                                maxIter, verbose, verboseFreq);
```

```matlabTextOutput
==================================
BGD Stopped: max number of iterations (999)
```

```matlab
tElapsed2 = toc(tStart);
```

As can be seen from the results the normalized inputs yield a faster execution time

```matlab
disp(['Elapsed time unnormalized input: ' num2str(tElapsed1)])
```

```matlabTextOutput
Elapsed time unnormalized input: 0.029855
```

```matlab
disp(['Elapsed time normalized input: ' num2str(tElapsed2)])
```

```matlabTextOutput
Elapsed time normalized input: 0.059167
```


Let's plot the cost function $J(\mathbf{w},b)=\frac{1}{2m}\sum_{i=1}^m {\left({\mathbf{w}}^{\textrm{T}} {\mathbf{x}}^{(i)} +b-y^{(i)} \right)}^2 =\frac{1}{2m}\sum_{i=1}^m {\left({\mathbf{x}}^{(i)} \mathbf{w}+b-y^{(i)} \right)}^2$ contour plot for the unnormalized and normalized features.

```matlab
f = figure;
tiledlayout(3,2,"TileSpacing","tight");
X = X_train(:,4); % Input data
y = y_train;

% Plot the cost function without input normalization
w_vec = -1e4:50:1e4; % weights vector
b_vec = -1e4:50:1e4; % bias vector

% Plot the cost function before the input normalization
J = scanCostFunction(w_vec,b_vec,X,y);

nexttile([1 2]);
contour(b_vec,w_vec,J,20);
xlabel('b')
ylabel('w')
title("Cost w/o norm.")
axis equal

%------------------------------------------------------------
% Max-value normalize the input feature
X_normalize = normalizeInput(X,"Max Value");

% Plot the cost function after input normalization
J_maxVal = scanCostFunction(w_vec,b_vec,X_normalize,y);

nexttile;
contour(b_vec,w_vec,J_maxVal,20);
xlabel('b')
ylabel('w')
title("max-val norm.")
axis equal

%------------------------------------------------------------
% Mean-value normalize the input feature
X_normalize = normalizeInput(X,"Mean Value");

% Plot the cost function after input normalization
J_meanVal = scanCostFunction(w_vec,b_vec,X_normalize,y);

nexttile;
contour(b_vec,w_vec,J_meanVal,20);
xlabel('b')
ylabel('w')
title("mean-val norm.")
axis equal

%------------------------------------------------------------
% Min-max normalize the input feature
X_normalize = normalizeInput(X,"Min-max");

% Plot the cost function after input normalization
J_minMax = scanCostFunction(w_vec,b_vec,X_normalize,y);

nexttile;
contour(b_vec,w_vec,J_minMax,20);
xlabel('b')
ylabel('w')
title("min-max norm.")
axis equal

%------------------------------------------------------------
% z-score normalize the input feature
X_normalize = normalizeInput(X,"z-score");

% Plot the cost function after input normalization
J_zScore = scanCostFunction(w_vec,b_vec,X_normalize,y);

nexttile;
contour(b_vec,w_vec,J_zScore,20);
xlabel('b')
ylabel('w')
title("z-score norm.")
axis equal
f.Position = [0 0 750 750];
```

![figure_5.png](./supervised_learning_week2_part1_media/figure_5.png)

Let's plot two features $w_1 $ and $ w_2$ w.r.t the cost function ( $J(\mathbf{w},b)=\frac{1}{2m}\sum_{i=1}^m {\left({\mathbf{x}}^{(i)} \mathbf{w}+b-y^{(i)} \right)}^2$ )

```matlab
% Run BGD algoritm and print the results
 

[w_vec,b_vec,J_hist] = hBatchGradientDescentMV(X_norm,y_train,alpha,condStop,maxIter,verbose,verboseFreq);
```

```matlabTextOutput
==================================
BGD Stopped: max number of iterations (999)
```

```matlab
f = figure;
T = tiledlayout(1,2,"TileSpacing","loose");

% Cost function without normalization
w1_vec = -1e4:10:1e4; % w1 vector
w2_vec = -1e4:10:1e4; % w2 vector
X_trainSub = [X_train(:,1) X_train(:,2)]; % two out of four features (subset)
J = [];
for i = 1:length(w1_vec)
     w1 = w1_vec(i);
    for j = 1:length(w2_vec)
        w2 = w2_vec(j);
        w_subVec = [w1 w2];

        J(i,j) = ( 1/(2*length(X_trainSub)) )*sum( ((w_subVec*X_trainSub.')+b_vec(end)-y_train.').^2 );
    end
end

nexttile
contour(w1_vec,w2_vec,J.',"ShowText",true,"LabelFormat","%0.1e m")
xlabel("w1")
ylabel("w2")
title("J(w,b) w/o norm.")

% Cost function with normalization
w1_vec = -1e4:10:1e4; % w1 vector
w2_vec = -1e4:10:1e4; % w2 vector
X_norm = [normalizeInput(X_train(:,1),"Mean Value") normalizeInput(X_train(:,2),"Mean Value")];
J = [];
for i = 1:length(w1_vec)
     w1 = w1_vec(i);
    for j = 1:length(w2_vec)
        w2 = w2_vec(j);
        w_subVec = [w1 w2];

        J(i,j) = ( 1/(2*length(X_norm)) )*sum( ((w_subVec*X_norm.')+b_vec(end)-y_train.').^2 );
    end
end

nexttile
contour(w1_vec,w2_vec,J.',"ShowText",true,"LabelFormat","%0.1e m")
xlabel("w1")
ylabel("w2")
title("J(w,b) w/ mean val. norm.")
f.Position = [0 0 700 400];
```

![figure_6.png](./supervised_learning_week2_part1_media/figure_6.png)
<a id="H_09652588"></a>

## Learning Rate in Practice

We know that the learning rate ( $\alpha$ ) controls the size of the update in each iteration and it is shared by all the parameters. Let's run gradient descent and try a few settings.

<a id="H_5114F050"></a>

### Case 1: alpha = 9.9e-7
```matlab
% BGD settings
alpha = 9.9e-7; % Learning rate
condStop = 1e-3; % Stopping condition
maxIter = 1e4; % Maximum number of iterations
verbose = false;
verboseFreq = 100;

% Run BGD algoritm and print the results
 
[w_hist,b_hist,J_hist] = hBatchGradientDescentMV(X_train,y_train,alpha,condStop,maxIter,verbose,verboseFreq);
```

```matlabTextOutput
==================================
BGD Stopped: max number of iterations (9999)
```

```matlab
w_vec = -3:0.001:3; % weights vector
b_vec = 0; % bias vector

% Plot the cost function
J = scanCostFunction(w_vec,b_vec,X_train(:,1),y_train);

figure;
tiledlayout(1,2,"TileSpacing","loose")

nexttile
plot(1:length(J_hist),J_hist,'b','LineWidth',3);
xlabel("# Iterations");
ylabel("Cost function, J(w,b)");
title("Cost vs Iteration")
xlim([50 120])

nexttile
plot(w_hist(1,:),J_hist,'mo-','LineWidth',2)
hold on
plot(w_vec,J,'b','LineWidth',3)
hold off
title("Cost vs w(1,:)")
xlabel("w(1,:)");
ylabel("Cost function, J(w,b)");
ylim([min(J) max(J)])
```

![figure_7.png](./supervised_learning_week2_part1_media/figure_7.png)

It appears the learning rate is too high. The solution does not converge. Cost is *increasing* rather than decreasing. 

<a id="H_A3D29B82"></a>

### Case 2: alpha = 9e-7
```matlab
% BGD settings
alpha = 9e-7; % Learning rate
condStop = 1e-3; % Stopping condition
maxIter = 1e4; % Maximum number of iterations
verbose = false;
verboseFreq = 100;

% Run BGD algoritm and print the results
 
[w_hist,b_hist,J_hist] = hBatchGradientDescentMV(X_train,y_train,alpha,condStop,maxIter,verbose,verboseFreq);
```

```matlabTextOutput
==================================
BGD Stopped: max number of iterations (9999)
```

```matlab
w_vec = -3:0.001:3; % weights vector
b_vec = 0; % bias vector

% Plot the cost function
J = scanCostFunction(w_vec,b_vec,X_train(:,1),y_train);

figure;
tiledlayout(1,2,"TileSpacing","loose")

nexttile
plot(1:length(J_hist),J_hist,'b','LineWidth',3);
xlabel("# Iterations");
ylabel("Cost function, J(w,b)");
title("Cost vs Iteration")
xlim([1 100])

nexttile
plot(w_hist(1,:),J_hist,'mo-','LineWidth',2)
hold on
plot(w_vec,J,'b','LineWidth',3)
hold off
title("Cost vs w(1,:)")
xlabel("w(1,:)");
ylabel("Cost function, J(w,b)");
ylim([min(J) max(J)/50])
```

![figure_8.png](./supervised_learning_week2_part1_media/figure_8.png)

Cost is decreasing throughout the run showing that alpha is not too large.

<a id="H_CBCC986D"></a>

### Case 2: alpha = 1e-7
```matlab
% BGD settings
alpha = 1e-7; % Learning rate
condStop = 1e-3; % Stopping condition
maxIter = 1e4; % Maximum number of iterations
verbose = false;
verboseFreq = 100;

% Run BGD algoritm and print the results
 
[w_hist,b_hist,J_hist] = hBatchGradientDescentMV(X_train,y_train,alpha,condStop,maxIter,verbose,verboseFreq);
```

```matlabTextOutput
==================================
BGD Stopped: max number of iterations (9999)
```

```matlab
w_vec = -3:0.001:3; % weights vector
b_vec = 0; % bias vector

% Plot the cost function
J = scanCostFunction(w_vec,b_vec,X_train(:,1),y_train);

figure;
tiledlayout(1,2,"TileSpacing","loose")

nexttile
plot(1:length(J_hist),J_hist,'b','LineWidth',3);
xlabel("# Iterations");
ylabel("Cost function, J(w,b)");
title("Cost vs Iteration")
xlim([1 30])

nexttile
plot(w_hist(1,:),J_hist,'mo-','LineWidth',3)
hold on
plot(w_vec,J,'b','LineWidth',3)
hold off
title("Cost vs w(1,:)")
xlabel("w(1,:)");
ylabel("Cost function, J(w,b)");
ylim([min(J) max(J)/1000])
xlim([0.1 0.3])
```

![figure_9.png](./supervised_learning_week2_part1_media/figure_9.png)

You can see from the figures that $w_1$ is decreasing without crossing the minimum. This solution will also converge, though not quite as quickly as the previous example.

<a id="H_B2D384DC"></a>

# Local Function Definitions
```matlab
function plotDatasetAgainstFeatures(X_train,y_train,Xstring,ystring)

% Validity checks
if size(X_train,2) ~= numel(Xstring)
    error("Training data and respective string sizes are not matching.")
end
if size(y_train,2) ~= numel(ystring)
    error("Targets and respective strings vector sizes are not matching.")
end

% Create the feature plots
f = figure;
T = tiledlayout(1,size(X_train,2),"TileSpacing","loose");

for i = 1:size(X_train,2)
    nexttile
    plot(X_train(:,i),y_train,'o','MarkerFaceColor','b')
    xlabel(Xstring(i));
    ylabel(ystring)
end
f.Position = [0 0 900 300];
end
```

```matlab
function normInput = normalizeInput(input,normType)
if strcmpi(normType,"Max Value")
    normInput = input./max(input);
elseif strcmpi(normType,"Min-max")
    a = 0;
    b = 1;
    normInput = ( (input - min(input))/(max(input)-min(input)) )*(b-a) + a;
elseif strcmpi(normType,"Mean Value")
    meanInput = (1/length(input))*sum(input);
    normInput = ( (input - meanInput)/(max(input)-min(input)) );
elseif strcmpi(normType,"z-score")
    meanInput = (1/length(input))*sum(input);
    varInput = (1/(length(input)-1))*sum((input-meanInput).^2); % sample variance (N-1)
    normInput = (input - meanInput)/sqrt(varInput);
else
    error("Undefined input normalization type.")
end

end
```

```matlab
function J = scanCostFunction(w_vec,b_vec,X,y)
% Calculate the cost function values
J = [];

for i = 1:length(w_vec)
    w = w_vec(i);
    for j = 1:length(b_vec)
        b = b_vec(j);
        J(i,j) = (1/(2*length(X))).*sum((w.*X+b-y).^2);
    end
end

end
```
