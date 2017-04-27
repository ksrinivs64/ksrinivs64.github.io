---
layout: slide
title:  "Introduction to artificial neural networks"
description: Introduction to artificial neural networks"
date:   2017-03-11 13:34:08 -0500
theme: black
transition: slide
categories: introduction neural networks
---

<section style="text-align: left;">
<p>
Artificial neural networks (ANN) are biologically inspired learning systems.  They rely on large networks of units to learn complex behavior. </p>
<img class="plain"  src="/images/ExampleNN.png" width="30%" height="30%" style="float: right"/>
     <ul style="width: 60%;"> <small>
            <li><p>Input units represent text, image or any input. </p> </li>
            <li> <p>Output units represent discrete or real valued output. </p> </li> 
	    <li><p>The strength of connections between units is represented by <b> weights </b>.</p> </li> 
	    <li><p>Output units usually compute a weighted sum of inputs.</p> </li>  
 
	     </small>
        </ul>
</section>

<section style="text-align: left;">
<p>
The earliest definition of an  neural network dates back to 1943 (McCullogh and Pitts). </p>
<img class="plain"  src="/images/McCullogh.png" width="40%" height="40%" align="right"/>
     <ul style="width: 50%;"><small>
            <li><p>Each input unit (e.g., A, B, or C) is a binary unit [0, 1] </p> </li>
	    <li><p>Each output unit (O) has a fixed threshold $\theta$ </p> </li> 
	    <li><p>Excitatory units (A or B) all have the same positive weight to the output unit. </p> </li>
	    <li><p>Inhibitory units (C) have the same negative weight to the output unit. </p> </li>
	    <li><p>An output unit (C) emits 1 if the sum of excitatory inputs is greater than threshold AND there is no inhibition. </p></li></small>
        </ul>
</section>

<section style="text-align: left;">
<p>
McCullogh and Pitts used this simple network to model Boolean logic functions. </p>
<img class="plain"  src="/images/McCulloghLogic.png" width="15%" height="15%" align="right"/>
     <ul style="width: 80%; list-style: none;"><small>
            <li>AND can be modeled by setting $\theta$ to 2, as shown in figure</li>
<table style="width:60%">
  <tr>
    <th>Input 1</th>
    <th>Input 2</th>
    <th>Output</th>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>1</td>
  </tr>

</table>
	    <li>OR can be modeled by setting $\theta$ to 1</li> 
<table style="width:60%">
  <tr>
    <th>Input 1</th>
    <th>Input 2</th>
    <th>Output</th>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>1</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>1</td>
  </tr>
</table>

	    <li>NOT can be modeled by setting $\theta$ to 0, and by adding a single inhibitory connection from the input to the output </li>
<table style="width:60%">
  <tr>
    <th>Input 1</th>
    <th>Output</th>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
  </tr>
  <tr>
    <td>0</td>
    <td>1</td>
  </tr>
</table>
</small>
        </ul>
</section>


<section style="text-align: left;">
<p>
Not all Boolean logic functions can be modeled by a single layer. </p>
<img class="plain"  src="/images/XOR.png" width="30%" height="30%" align="right"/>
     <ul style="width: 60%; list-style: none;"><small>
            <li>XOR for instance needs at least 2 layers, where the final output neuron has $\theta$ set to 1, as shown in figure</li>
<table style="width:60%">
  <tr>
    <th>Input 1</th>
    <th>Input 2</th>
    <th>Output</th>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>1</td>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>0</td>
  </tr>

  <tr>
    <td>0</td>
    <td>1</td>
    <td>1</td>
  </tr>

  <tr>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>

</table>
</small>
</ul>
</section>

<section style="text-align: left;">
<p>
Problems with McCullogh and Pitts neurons </p>
     <ul>
            <li>No real learning in the system.  Weights are 'hard-wired' for each case. </li>
	    <li>No different from a digital logic circuit. </li>
     </ul>

Rosenblatt injected learning into these systems in his 1958 paper and called this learning system a <u>Perceptron</u>
</section>

<section style="text-align: left;">
<p>Perceptron</p>
<img class="plain"  src="/images/Perceptron.png" width="35%" height="35%" align="right"/>
     <ul style="width: 60%; list-style: none;">
      <li>The $\sum$ unit computed the sum as before. $\sum_{i=1}^{n}w_{i}x_{i}$</li>
      <li> Alternatively in vector notation, $\sum$ unit computes $w  \boldsymbol{\cdot} x$.</li>
      <p>
      <li> $o = \begin{cases} 1 & \text{if $\sum_{i=1}^{n}w_{i}x_{i}>0$} \\ -1 & \text{otherwise}\end{cases}$</li> </p>
      <li> Most importantly, weights were learned for different problems.</li>
</ul>
</section>

<section  style="text-align: left;">
<p>Perceptron training rule</p>
     <ul style="list-style: none;">
            <li>$w_{i} \leftarrow w_{i} + \Delta w_{i}$ </li>
	    <li>where</li>
	    <li>$\Delta w_{i} = \eta(t-o)x_{i}$ </li>
     </ul>
     <p> Where </p>
     <ul>
	<li> $t$ is the expected target value for $x_{i}$</li>
	<li> $o$ is the perceptron output for $x_{i}$. </li>
	<li> $\eta$ is a parameter called the learning rate. </li>
     </ul>
</section>

<section style="text-align: left;">
<p>Perceptron Example</p>
<img class="plain"  src="/images/PerceptronExample.png" width="30%" height="30%" align="right"/>
     For the example:
     <ul style="width: 60%; list-style: none;">
     <li> If $o=1$, then $\Delta w_{i} = \eta(t-o)x_{i}$</li>
     <li> $\Delta w_{i}=0.1(1 - (-1))0.8 = 0.16$ </li>
     <li> This increases the weight. </li>
     <p>
     <li> If the $t=-1$ and $o=1$ then the weight would be decreased. </li>
     </p>
     <p>
      <li>If expected output is -1, the weight remains unchanged.</li> </p>
</ul>
</section>

<section  style="text-align: left;">
<p>Perceptron training rule will converge if</p>
     <ul style="list-style: none;">
            <li>Training data are linearly separable. </li>
	    <li>$\eta$ is sufficiently small.</li>
     </ul>
<img class="plain"  src="/images/LinearSeparable.png" width="70%" height="70%"/>
     
</section>

<section  style="text-align: left;">
<p>Delta rule</p>

<div id="container" style="width:100%;">  
<div id="right" style="float:right; width:50%;text-align:center;">   
<figure>
<img class="plain"  src="/images/GradientDescent.png" width="80%" height="80%" align="right"/>
<figcaption> <small>The $x$ and $y$ plane represent all possible combinations of values for weight vectors $w_{0}$ and $w_{1}$ (aka the hypothesis space).  Error is plotted on $z$ at each choice of $w_{0}$, $w_{1}.$</small> </figcaption>
</figure>
</div>
<div id="left" style="float:left; width:50%;">
     <ul>
	    <li>Delta rule converges to a best-fit approximation if training is not linearly separable.</li>
	    <li>It searches for weights that minimize error by <u>gradient descent</u> on an error function over all possible weights. </li>
     </ul>
</div>
</div> 
</section>


<section  style="text-align: left;">
<p>The error function for the delta rule</p>
<div id="container" style="width:100%;">  
<div id="right" style="float:right; width:50%; text-align:center;">   
<figure>
<img class="plain"  src="/images/GradientDescent.png" width="80%" height="80%" align="right"/>
<figcaption> <small>Error function is a parabola with a single global minimum.</small> </figcaption>
</figure>
</div>

<div id="left" style="float:left; width:50%;">
     <ul>
	    <li>In the delta rule, training is done on the output of the linear unit <b>before</b> the application of any thresholds.</li>
	    <li>The error function for a dataset of D training examples is $$E(\vec{w})=1/2 \sum_{d \in D} (t_{d} - o_{d})^2$$</li>
     </ul> 
</div>
</div>
</section>

<section  style="text-align: left;">
<p>Gradient descent for the error function</p>
     <ul style="list-style: none;">
            <li>The direction of gradient descent is given by the derivative of $E$ with repect to $\vec{w}$. That is, $$\triangledown E (\vec{w}) \equiv [ \, \frac{\partial E}{\partial w_{0}},  \frac{\partial E}{\partial w_{1}}...,  \frac{\partial E}{\partial w_{n}} ] \,$$ </li>
	    <li><p>where $\triangledown E (\vec{w})$ is itself a vector, composed of partial derivatives of $E$ with respect to each weight $w_{i}$.</p> </li>
	    <p>
	    <li> $\triangledown E (\vec{w})$ points to the direction of steepest increase in $E$. </li>
	    <li> So, the change in weight $\Delta \vec{w} = - \eta \triangledown E (\vec{w})$ </li> </p>
     </ul>
</section>

<section  style="text-align: left;">
<p>Derivation of Gradient for a specific $w_{i}$</p>
     <ul style="list-style: none;">
            <li>$$\frac{\partial E}{\partial w_{i}} = \frac{\partial}{\partial w_{i}} \frac{1}{2}\sum_{d \in D} (t_{d} - o_{d})^2 $$ </li>
	    <li>$$=\frac{1}{2}\sum_{d \in D}\frac{\partial}{\partial w_{i}}(t_{d} - o_{d})^2$$</li>
	    <li>$$=\frac{1}{2}\sum_{d \in D}2(t_{d} - o_{d})\frac{\partial}{\partial w_{i}}(t_{d} - o_{d})$$ </li>
	    <li>$$=\sum_{d \in D}(t_{d} - o_{d})\frac{\partial}{\partial w_{i}}(t_{d} - \vec{w} \cdot \vec{x_{d}})$$ </li> </ul>

</section>

<section  style="text-align: left;">
<p> Delta rule </p>
     <ul>
	    <li>For a specific input component $x_{id}$ of a training example $d$, $$\frac{\partial E}{\partial w_{i}} = \sum_{d \in D} (t_{d} - o_{d})(-x_{id})$$  </li>
	    <li> This means that $$\Delta{w_{i}} = \eta \sum_{d \in D}(t_{d} - o_{d}) x_{id}$$ </li>

</ul>
</section>

<section  style="text-align: left;">
<p> Delta versus Perceptron rule </p>
The delta rule looks exactly like the perceptron rule, but there are actually some key differences:
<ul>
  <li> The delta rule is based on the error in the unthresholded linear combination of inputs.
</li>
 <li> In contrast, the perceptron rule is based on the error AFTER the thresholds have been applied. </li>
<li> The perceptron therefore will not converge if training data are not linearly separable. </li>
<li> The delta rule will converge to the minimum error hypothesis regardless of linearly separability. </li> 
     </ul>
</section>

<section  style="text-align: left;">
<p>Delta rule and Linear Regression</p>
<ul style="list-style: none;">
	    <li> What we have discussed so far is exactly the same as linear regression.  In linear regression, for a given example $d$ in $D$, $$ \hat{y}=\sum_{j=1}^{k} x_{dj} \theta_{j} $$ 
	    </li>
	    <li> $$ \hat{y} = x_{d1}\theta_{1} + x_{d2}\theta_{2} + ... x_{dk}\theta_{k}$$ </li> 
	    <li> $\theta_{1}$ is just the bias term with $x_{d1}$ set to 1. </li>
</ul>
</section>

<section  style="text-align: left;">
<p>Delta rule and Linear Regression</p>
<ul>
	    <li> $k$ is the number of dimensions which in a neural network maps to the number of hidden units. </li>
	    <li> $\theta_{1}...\theta_{k}$ are parameters which are weights for each hidden unit.</li>
	    <li> $\hat{y}$ corresponds to the output of a single output unit which was $o$ </li>
	    <li> In linear regression, as in neural nets, the error function that is minimized is $E = \frac{1}{2}\sum_{d \in D}(t - \hat{y})^2$. </li>
     </ul>
</section>

<section  style="text-align: left;">
<p>Adding nonlinearity</p>
<div id="container" style="width:100%;">  
<div id="right" style="float:right; width:40%; text-align:center;">   
<figure>
<img class="plain"  src="/images/Nonlinear.png" width="80%" height="80%" align="right"/>
<figcaption> <small>Green and red points here reflect two different classes of instances (e.g. cat vs. dog).</small> </figcaption>
</figure>
</div>
<div id="left" style="float:left; width:60%; text-align:left;">   
With linear output units the network can only learn linear functions. 
<ul style="list-style: none;">
	    <li> To be able to learn nonlinear functions, the output needs to be a nonlinear function of inputs.</li>
	    <li> The nonlinear function needs to be differentiable to use gradient descent.</li>
</ul>
</div>
</div>
</section>

<section  style="text-align: left;">
<p>Adding nonlinearity</p>
<ul style="list-style: none;">
	    <li> One popular function is the sigmoid function $\sigma$.  $o$ becomes $$o = \sigma(\vec{w} \cdot \vec{x})$$ </li>
	    <li> where $$\sigma = \frac{1}{1 + e^{-y}}$$ </li>
	    <li> The sigmoid function is basically a squashing function of the output - it squashes output to [0,1]. </li>
     </ul>
</section>

<section  style="text-align: left;">
<p>Differentiating the sigmoid function</p>
<ul style="list-style: none;">
              <li> $$ \frac{\partial \sigma}{\partial y} = \frac{e^{-y}}{(1+e^{-y})^2}$$ </li>
	    <li> $$ = \frac{e^{-y} + 1 - 1}{(1 + e^{-y})^2}$$ </li>
	    <li> $$ = \frac{1 + e^{-y}}{(1 + e^{-y})^2} - \frac{1}{(1 + e^{-y})^2}$$ </li>
	    <li> $$ = \frac{1}{1+ e^{-y}} - \left( \frac{1}{1 + e^{-y}} \right)^2 $$ </li>
	    <li> $$\frac{\partial \sigma}{\partial y} = \sigma(y) \cdot (1 - \sigma(y))$$ </li>
     </ul>
</section>


<section style="text-align: left;">
<p>Putting it together...</p>
<img class="plain"  src="/images/NonlinearSimple.png" width="30%" height="30%" align="right"/>
     In terms of our error term from before
     <ul style="width: 60%; list-style: none;">
	    <li>For a specific input component $x_{i}$ of a training example $d$, where $o = \sigma(y)$ $$\frac{\partial E}{\partial w_{i}} = \sum (t - o)o (1 - o) (-x_{i})$$  </li>
</ul>
</section>

<section style="text-align: left;">
<p>Extending it to multiple layers</p>
<img class="plain"  src="/images/BackProp.png" width="30%" height="30%" align="right"/>
     Forward pass:
     <ul style="width: 60%; list-style: none;"><small>
	    <li>$$net_{h1}=0.35 + .15 * .05 + .25 * .1$$
	    $$=.378$$ </li>
	    <li>$$o_{h1}=\frac{1}{1 + e^{-net_{h1}}}$$
	    $$=0.593$$</li>
	    <li>$$net_{y1}=0.6 + .4 * .593 + .45 * .598$$
	    $$o_{y1}=.751$$</li>
	    <li>Assuming expected output at $y_{1}$ is .01, $E_{y1}=\frac{1}{2}(t - o_{y1})^2$.</li>
	    <li> $=\frac{1}{2}(.01 - .751)^2 =.275$ </li>
	    <li>$E_{y2}=.023$</li>
</small>
</ul>
</section>


<section style="text-align: left;">
<p>Adjusting the weights</p>
<img class="plain"  src="/images/BackProp2.png" width="30%" height="30%" align="right"/>
     Backward pass to adjust weights for $w_{h1y1}$:
     <ul style="width: 60%; list-style: none;"><small>
	    <li>$$\frac{\partial E_{total}}{\partial w_{h1y1}}=(t - o_{y1})o_{y1} (1 - o_{y1}) (-h_{1})$$
	    $$= -(.01-.751) * .751 * (1- .751) * .593$$ 
	    $$= .082 $$</li>
	    $$ w_{h1y1} = w_{h1y1} - \eta * .082 $$ 
	    <p>
	    <li>Note that only $y_{1}$'s error is considered for $w_{h1y1}$ which makes sense because $w_{h1y1}$ only contributes to $y_{1}$ </li>
	    </p>

	    <p>
	    <li> To generalize, delta for $\delta o_{yj} = (t - o_{yj})o_{yj} (1 - o_{yj}) $ </li>
	    <li> $$\frac{\partial E_{total}}{\partial w_{ij}}= - \delta o_{yi} $$ </li>
	    </p>
</small>
</ul>
</section>

<section style="text-align: left;">
<p>Adjusting the weights</p>
<img class="plain"  src="/images/BackProp3.png" width="30%" height="30%" align="right"/>
     Backward pass to adjust weights for $w_{x1h1}$:
     <ul style="width: 60%; list-style: none;"><small>
	    <li>Intuitively, to determine how to change $w_{x1h1}$ we need to understand what is $\delta o_{h1}$.  </li>
	    <li> $h1$ contributes to error through $y1$ and $y2$, and the degree to which it contributes to error is governed by the weight. </li>           
<p>
<li> Mathematically, $$ \frac{\partial E_{total}}{\partial o_{h1}}= \frac{\partial E_{y1}}{\partial o_{h1}} + \frac{\partial E_{y2}}{\partial o_{h1}} $$ </li> </p>
<li> where $$\frac{\partial E_{y1}}{\partial o_{h1}} = \frac{\partial E_{y1}}{\partial o_{y1}} \frac{\partial o_{y1}}{\partial net_{y1}} \frac{\partial net_{y1}}{\partial o_{h1}} $$ </li>
</small>
</ul>
</section>


<section style="text-align: left;">
<p>Adjusting the weights for hidden layers</p>
<img class="plain"  src="/images/BackProp3.png" width="30%" height="30%" align="right"/>
     Backward pass to adjust weights for $w_{x1h1}$:
     <ul style="width: 60%; list-style: none;"><small>
<li> Note that $$\frac{\partial E_{y1}}{\partial o_{h1}} = \frac{\partial E_{y1}}{\partial o_{y1}} \frac{\partial o_{y1}}{\partial net_{y1}} \frac{\partial net_{y1}}{\partial o_{h1}} $$ can be rewritten as:
$$ = \delta y1 \frac{\partial net_{y1}}{\partial o_{h1}} $$ </li>

<li> $$\delta y1 = (.01 - .751)*.751*(1-.751) = .138 $$ </li>
<li> $$\frac{\partial net_{y1}}{\partial o_{h1}} = w_{h1y1}$$ because $$ net_{y1} = w_{h1y1} * o_{h1} + w_{h2y1} * o_{h2} $$ </li>
<li> Therefore $$\frac{\partial E_{y1}}{\partial o_{h1}} = .138 * .4 = .055$$ </li>
</small>
</ul>
</section>




<section style="text-align: left;">
<p>Adjusting the weights for hidden layers</p>
<img class="plain"  src="/images/BackProp3.png" width="30%" height="30%" align="right"/>
     Backward pass to adjust weights for $w_{x1h1}$:
     <ul style="width: 60%; list-style: none;"><small>
<li> Repeating this process for $y2$ yields: $\frac{\partial E_{y2}}{\partial o_{h1}} = -.019$ </li>
<li> And $\frac{\partial E_{total}}{\partial o_{h1}} = .055 - .019 = .036$ </li>
<li> Now that we have $\frac{\partial E_{total}}{\partial o_{h1}}$ we need to calculate $\frac{\partial o_{h1}}{\partial net_{h1}} \frac{\partial net_{h1}}{\partial w_{x1h1}}$ </li>
<li>$\frac{\partial o_{h1}}{\partial net_{h1}} = o_{h1}*(1- o_{h1}) = .593 * (1 - .593) = .241$ </li>
<li>Or $\delta h1 = .241 * .036 = .008 $ </li>
<li>$$\frac{\partial E_{total}}{\partial w_{x1h1}} = \delta h1 * x1 $$ </li>
<li>$$ = .008 * .05 = .0004$$ </li>
<li> $$w_{x1h1} = w_{x1h1} - (\eta * .004) $$ </li>
</small>
</ul>
</section>


<section style="text-align: left;">
<p>Layer wise back-propagation</p>
<img class="plain"  src="/images/BackProp3.png" width="30%" height="30%" align="right"/>
     Backward pass to adjust weights for $w_{x1h1}$:
     <ul style="width: 60%; list-style: none;"><small>
<li> Repeating this process for $y2$ yields: $\frac{\partial E_{y2}}{\partial o_{h1}} = -.019$ </li>
<li> And $\frac{\partial E_{total}}{\partial o_{h1}} = .055 - .019 = .036$ </li>
<li> Now that we have $\frac{\partial E_{total}}{\partial o_{h1}}$ we need to calculate $\frac{\partial o_{h1}}{\partial net_{h1}} \frac{\partial net_{h1}}{\partial w_{x1h1}}$ </li>
<li>$\frac{\partial o_{h1}}{\partial net_{h1}} = o_{h1}*(1- o_{h1}) = .593 * (1 - .593) = .241$ </li>
<li>Or $\delta h1 = .241 * .036 = .008 $ </li>
<li>$$\frac{\partial E_{total}}{\partial w_{x1h1}} = \delta h1 * x1 $$ </li>
<li>$$ = .008 * .05 = .0004$$ </li>
<li> $$w_{x1h1} = w_{x1h1} - (\eta * .004) $$ </li>
</small>
</ul>
</section>
