
<a id="T_60065D90"></a>

# Supervised Learning \- Week 1
<!-- Begin Toc -->

## Table of Contents
&emsp;[Linear Regression](#H_40B92E36)
 
&emsp;[Gradient Descent (GD) Algorithm}Please note that the **squared loss cost function** ensures that the 'error surface' is convex like a soup bowl.It will always have a minimum that can be reached by following the gradient in all dimensions.-  Why? The overview of what we are aiming with gradient descent;-  Have some function $J(w,b) $ \end{flushleft}}   \item{\begin{flushleft} We want to minimize this function such that $ \underset{w,b}{\mathrm{min}} \left\lbrace J(w,b)\right\rbrace =\underset{w,b}{\mathrm{min}} \left\lbrace \frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} +b-y^{(i)} \right)}^2 \right\rbrace $, where the $ (w,b)$ pair that minimizes the given function becomes our solution. It turns out that the gradient descent algorithm can minimize not only linear regression cost function but any function that is convex. <a id="H_5965BE43](#gradient-descent-gd-algorithmbeginparbeginflushleftplease-note-that-the-squared-loss-cost-function-ensures-that-the-error-surface-is-convex-like-a-soup-bowlit-will-always-have-a-minimum-that-can-be-reached-by-following-the-gradient-in-all-dimensionsendflushleftendparbeginitemizesetlengthitemsep1ex-itembeginflushleft-why-endflushleftenditemizebeginparbeginflushleftthe-overview-of-what-we-are-aiming-with-gradient-descentendflushleftendparbeginitemizesetlengthitemsep1ex-itembeginflushleft-have-some-function-jwb-undersetwbmathrmmin-leftlbrace-jwbrightrbrace-undersetwbmathrmmin-leftlbrace-frac12msumi1m-leftwxi-byi-right2-rightrbrace-wb-pair-that-minimizes-the-given-function-becomes-our-solution-it-turns-out-that-the-gradient-descent-algorithm-can-minimize-not-only-linear-regression-cost-function-but-any-function-that-is-convex-endflushleftenditemizelabelh5965be43)
 
&emsp;&emsp;&emsp;[Methodology"></a>1.  We start with some initial guesses of $w$ and $b$ (random initialization). For instance, in linear regression it won't matter what the initial values are. Hence, the common choice is setting them to zero $w=0$ and $b=0$. }   1.  Keep changing the parameters $w$ and $b$ a bit every time/iteration to reduce the cost $J(w,b)$. }   1.  Repeat step 2 until (hopefully) we settle at or near minimum. Note that for some cost functions that might not have a bowl shape (non\-convex) it is possible to have more than 1 minimum value. For instance,![figure_5.png](./supervised_learning_week1_media/figure_5.png)What gradient descent algorithm does is it sits on a location and looks around and asks; "if I wanted to take a baby step in one direction to go downhill as quickly as possible to one of these valleys what direction should I choose to take a step?"-  If you want to walk down the hill as efficiently as possible it turns out that the best direction is the direction towards the **steepest descent**. In the following figure we can see two different randomly initiated gradient descent algorithms, which converge to different local minimum points.![image_1.png](./supervised_learning_week1_media/image_1.png)<a id="H_81A0C8E0](#methodologybeginenumeratesetlengthitemsep1ex-itembeginflushleft-we-start-with-some-initial-guesses-of-and-random-initialization-for-instance-in-linear-regression-it-wont-matter-what-the-initial-values-are-hence-the-common-choice-is-setting-them-to-zero-w0-and-endflushleft-itembeginflushleft-keep-changing-the-parameters-and-a-bit-every-timeiteration-to-reduce-the-cost-jwb-endflushleft-itembeginflushleft-repeat-step-2-until-hopefully-we-settle-at-or-near-minimum-note-that-for-some-cost-functions-that-might-not-have-a-bowl-shape-nonconvex-it-is-possible-to-have-more-than-1-minimum-value-endflushleftendenumeratebeginparbeginflushleftfor-instanceendflushleftendparbegincenterincludegraphicswidthmaxwidth56196688409433015emfigure5pngendcenterbeginparbeginflushleftwhat-gradient-descent-algorithm-does-is-it-sits-on-a-location-and-looks-around-and-asks-if-i-wanted-to-take-a-baby-step-in-one-direction-to-go-downhill-as-quickly-as-possible-to-one-of-these-valleys-what-direction-should-i-choose-to-take-a-stependflushleftendparbeginitemizesetlengthitemsep1ex-itembeginflushleft-if-you-want-to-walk-down-the-hill-as-efficiently-as-possible-it-turns-out-that-the-best-direction-is-the-direction-towards-the-steepest-descent-endflushleftenditemizebeginparbeginflushleftin-the-following-figure-we-can-see-two-different-randomly-initiated-gradient-descent-algorithms-which-converge-to-different-local-minimum-pointsendflushleftendparbeginparbeginflushleftincludegraphicswidthmaxwidth8670346211741094emimage1endflushleftendparlabelh81a0c8e0)
 
&emsp;&emsp;[Implementation"></a>-  In each step, update the weight(s) as follows: $w=w-\alpha \frac{\partial }{\partial w}J(w,b)$. It means that update the $w$ such that take the current value of the $w$ and subtract the scaled derivative of the cost function with respect to $w$ (Here, the equal sign represents the assignment operation). The parameter $\alpha $ is the \textbf{learning rate}, where $ 0\le \alpha \le 1 $. The learning rate controls how big a step you take towards the steepest descent. \end{flushleft}}   \item{\begin{flushleft} In each step, also update the bias as follows: $ b=b-\alpha \frac{\partial }{\partial b}J(w,b)$. }   -  In gradient descent, we need to repeat the above steps until convergence. What is convergence? Convergence means that you reached the point at local/global minimum, where the parameters no longer change much in each step. }   -  There is one more subtle detail for correctly implementing the gradient descent; you will be updating the parameters $w$ and $b$, this update takes place for both parameters. In gradient descent, you want to simultaneously update the parameters $w$ and $b$. **Correct way (simultaneous update) of implementing gradient descent:**\begin{enumerate}\setlength{\itemsep}{-1ex}   \item{ $\displaystyle w_{\textrm{tmp}} =w-\alpha \frac{\partial }{\partial w}J(w,b) $ \end{flushleft}}   \item{\begin{flushleft} $ \displaystyle b_{\textrm{tmp}} =b-\alpha \frac{\partial }{\partial b}J(w,b) $ \end{flushleft}}   \item{\begin{flushleft} $ \displaystyle w=w_{\textrm{tmp}} $ \end{flushleft}}   \item{\begin{flushleft} $ \displaystyle b=b_{\textrm{tmp}} $ \end{flushleft}}\end{enumerate}\begin{flushleft}where $ w_{\textrm{tmp}} $ and $ b_{\textrm{tmp}}$ represents the temporary values for $w$ and $b$, respectively.**Incorrect way of implementing gradient descent:**\begin{enumerate}\setlength{\itemsep}{-1ex}   \item{ $\displaystyle w_{\textrm{tmp}} =w-\alpha \frac{\partial }{\partial w}J(w,b) $ \end{flushleft}}   \item{\begin{flushleft} $ \displaystyle w=w_{\textrm{tmp}} $ \end{flushleft}}   \item{\begin{flushleft} $ \displaystyle b_{\textrm{tmp}} =b-\alpha \frac{\partial }{\partial b}J(w,b) $ \end{flushleft}}   \item{\begin{flushleft} $ \displaystyle b=b_{\textrm{tmp}} $ \end{flushleft}}\end{enumerate}\begin{flushleft}It turns out that even if you implement the gradient descent incorrectly, it would still work more less the same as the correct implementation, however, it is important to implement the algorithm according to the correct theoretical insight, which will ensure our long term success and will avoid unexpected problems in the future.\end{flushleft}\label{H_99731A70}\matlabheadingtwo{Gradient Descent Intuition}\begin{flushleft}In order to understand why the weight and bias update formulas\end{flushleft}\begin{flushleft}$ w=w-\alpha \frac{\partial }{\partial w}J(w,b) $, \end{flushleft}\begin{flushleft}$ b=b-\alpha \frac{\partial }{\partial b}J(w,b)$ are used. Let's start with a simpler example, which contains one parameter. Thus, the cost function is in the form of $J(w)$ and the gradient descent based weight update formula becomes $w=w-\alpha \frac{\partial }{\partial w}J(w) $. Let's look at the previous simple single parameter cost function example and try to understand:\end{flushleft}\begin{itemize}\setlength{\itemsep}{-1ex}   \item{\begin{flushleft} What the learning rate ($ \alpha $) does \end{flushleft}}   \item{\begin{flushleft} What the derivative part ($ \frac{\partial }{\partial w}J(w,b) $ or $ \frac{\partial }{\partial b}J(w,b) $) does \end{flushleft}}   \item{\begin{flushleft} Why they are multiplied \end{flushleft}}   \item{\begin{flushleft} Why the multiplication is subtracted from the initial value and why it make sense \end{flushleft}}\end{itemize}\begin{flushleft}Since this example has only one parameter, the gradient descent algorithm looks like $ w=w-\alpha \frac{\partial }{\partial w}J(w)$. And we try to minimize the cost function by only setting $w$ such that $\underset{w}{\mathrm{min}} \left\lbrace J(w)\right\rbrace $.\end{flushleft}\begin{flushleft}Let's initialize the gradient descent in some random points similar to the previous example with two parameters. In order to plot the gradient of the cost function, we should first calculate it. The cost function is given by\end{flushleft}\begin{flushleft}$ J(w)=\frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} -y^{(i)} \right)}^2 $, where we need to calculate the $ \frac{\partial }{\partial w}J(w)=\frac{\partial }{\partial w}\left(\frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} -y^{(i)} \right)}^2 \right)$. Since we know that the derivative operation is distributive on the sum operation, $ \frac{\partial }{\partial w}J(w)=\frac{1}{2m}\sum_{i=1}^m \frac{\partial }{\partial w}{\left(wx^{(i)} -y^{(i)} \right)}^2 =\frac{1}{2m}\sum_{i=1}^m 2\left(wx^{(i)} -y^{(i)} \right)\cdot x^{(i)} =\frac{1}{m}\sum_{i=1}^m \left(wx^{(i)} -y^{(i)} \right)\cdot x^{(i)} ~~(3) $ Similarly, we can also obtain the rate of change of the cost function w.r.t. $b$ as follows: $ \frac{\partial }{\partial b}J(w)=\frac{1}{2m}\sum_{i=1}^m \frac{\partial }{\partial w}{\left(wx^{(i)} -y^{(i)} \right)}^2 =\frac{1}{2m}\sum_{i=1}^m 2\left(wx^{(i)} -y^{(i)} \right)=\frac{1}{m}\sum_{i=1}^m \left(wx^{(i)} -y^{(i)} \right)~~(4) $ -  Let's visualize the effect of the selected weight on the $J(w,b) $ and $ \frac{\partial }{\partial w}J(w)$: ![figure_6.png](./supervised_learning_week1_media/figure_6.png)<a id="H_53BAA125](#implementationbeginitemizesetlengthitemsep1ex-itembeginflushleft-in-each-step-update-the-weights-as-follows-wwalpha-fracpartial-partial-wjwb-it-means-that-update-the-such-that-take-the-current-value-of-the-and-subtract-the-scaled-derivative-of-the-cost-function-with-respect-to-here-the-equal-sign-represents-the-assignment-operation-the-parameter-alpha-0le-alpha-le-1-bbalpha-fracpartial-partial-bjwb-endflushleft-itembeginflushleft-in-gradient-descent-we-need-to-repeat-the-above-steps-until-convergence-what-is-convergence-convergence-means-that-you-reached-the-point-at-localglobal-minimum-where-the-parameters-no-longer-change-much-in-each-step-endflushleft-itembeginflushleft-there-is-one-more-subtle-detail-for-correctly-implementing-the-gradient-descent-you-will-be-updating-the-parameters-and-this-update-takes-place-for-both-parameters-in-gradient-descent-you-want-to-simultaneously-update-the-parameters-and-endflushleftenditemizebeginparbeginflushleftcorrect-way-simultaneous-update-of-implementing-gradient-descentendflushleftendparbeginenumeratesetlengthitemsep1ex-itembeginflushleft-displaystyle-wrmtmp-walpha-fracpartial-partial-wjwb-displaystyle-brmtmp-balpha-fracpartial-partial-bjwb-displaystyle-wwrmtmp-displaystyle-bbrmtmp-wrmtmp-brmtmp-represents-the-temporary-values-for-and-respectivelyendflushleftendparbeginparbeginflushleftincorrect-way-of-implementing-gradient-descentendflushleftendparbeginenumeratesetlengthitemsep1ex-itembeginflushleft-displaystyle-wrmtmp-walpha-fracpartial-partial-wjwb-displaystyle-wwrmtmp-displaystyle-brmtmp-balpha-fracpartial-partial-bjwb-displaystyle-bbrmtmp-wwalpha-fracpartial-partial-wjwb-bbalpha-fracpartial-partial-bjwb-are-used-lets-start-with-a-simpler-example-which-contains-one-parameter-thus-the-cost-function-is-in-the-form-of-and-the-gradient-descent-based-weight-update-formula-becomes-wwalpha-fracpartial-partial-wjw-alpha-fracpartial-partial-wjwb-fracpartial-partial-bjwb-wwalpha-fracpartial-partial-wjw-and-we-try-to-minimize-the-cost-function-by-only-setting-such-that-undersetwmathrmmin-leftlbrace-jwrightrbrace-jwfrac12msumi1m-leftwxi-yi-right2-fracpartial-partial-wjwfracpartial-partial-wleftfrac12msumi1m-leftwxi-yi-right2-right-endflushleftendparbeginparbeginflushleftsince-we-know-that-the-derivative-operation-is-distributive-on-the-sum-operationendflushleftendparbeginparfracpartial-partial-wjwfrac12msumi1m-fracpartial-partial-wleftwxi-yi-right2-frac12msumi1m-2leftwxi-yi-rightcdot-xi-frac1msumi1m-leftwxi-yi-rightcdot-xi-3endparbeginparbeginflushleftsimilarly-we-can-also-obtain-the-rate-of-change-of-the-cost-function-wrt-as-followsendflushleftendparbeginparfracpartial-partial-bjwfrac12msumi1m-fracpartial-partial-wleftwxi-yi-right2-frac12msumi1m-2leftwxi-yi-rightfrac1msumi1m-leftwxi-yi-right4endparbeginitemizesetlengthitemsep1ex-itembeginflushleft-lets-visualize-the-effect-of-the-selected-weight-on-the-jwb-fracpartial-partial-wjw-endflushleftenditemizebegincenterincludegraphicswidthmaxwidth56196688409433015emfigure6pngendcenterlabelh53baa125)
 
&emsp;&emsp;[Selection of the Learning Rate"></a>Selection of the learning rate, $\alpha $, is also really important in gradient descent implementation.\end{flushleft}\begin{flushleft}To understand what $ \alpha $ is doing in gradient descent algorithm, let's explore the case where $ \alpha $ is either too small or too large.\end{flushleft}\begin{itemize}\setlength{\itemsep}{-1ex}   \item{\begin{flushleft} If $ \alpha$ is too small: The gradient descent will work but you end up increasing/decreasing $w$ but **very slowly**.It means that you will need a lot of steps (it will take a very long time) to reach to the minimum point. }   \item{ If $\alpha$ is too large: You update $w$ with a giant step and the cost might even increase over iterations (means you might go further from the minimum point). In this case gradient descent will overshoot and never reach the minimum, which means that it fails to **converge** to a solution but also **diverge**. }\end{itemize}![image_2.png](./supervised_learning_week1_media/image_2.png)-  If you are at a local minimum already, further gradient descent steps won't change anything since the derivative of the cost function equals to zero, which means that $w$ won't be updated. It is important note that as we approach the local minimum, gradient descent will automatically take smaller steps due to the smaller derivative of the cost function. Therefore, gradient descent can reach local minimum without decreasing the learning rate. **In other words, gradient descent can reach local minimum with a fixed learning rate.**We can use gradient descent algorithm to minimize any cost function $J$, not just the mean squared error cost function that we are using for linear regression.<a id="H_3D466374](#selection-of-the-learning-ratebeginparbeginflushleftselection-of-the-learning-rate-alpha-alpha-alpha-alpha-is-too-small-the-gradient-descent-will-work-but-you-end-up-increasingdecreasing-but-very-slowlyit-means-that-you-will-need-a-lot-of-steps-it-will-take-a-very-long-time-to-reach-to-the-minimum-point-endflushleft-itembeginflushleft-if-alpha-is-too-large-you-update-with-a-giant-step-and-the-cost-might-even-increase-over-iterations-means-you-might-go-further-from-the-minimum-point-in-this-case-gradient-descent-will-overshoot-and-never-reach-the-minimum-which-means-that-it-fails-to-converge-to-a-solution-but-also-diverge-endflushleftenditemizebeginparbeginflushleftincludegraphicswidthmaxwidth6281986954340191emimage2endflushleftendparbeginitemizesetlengthitemsep1ex-itembeginflushleft-if-you-are-at-a-local-minimum-already-further-gradient-descent-steps-wont-change-anything-since-the-derivative-of-the-cost-function-equals-to-zero-which-means-that-wont-be-updated-endflushleftenditemizebeginparbeginflushleftit-is-important-note-that-as-we-approach-the-local-minimum-gradient-descent-will-automatically-take-smaller-steps-due-to-the-smaller-derivative-of-the-cost-function-therefore-gradient-descent-can-reach-local-minimum-without-decreasing-the-learning-rate-in-other-words-gradient-descent-can-reach-local-minimum-with-a-fixed-learning-rateendflushleftendparbeginparbeginflushleftwe-can-use-gradient-descent-algorithm-to-minimize-any-cost-function-j-not-just-the-mean-squared-error-cost-function-that-we-are-using-for-linear-regressionendflushleftendparlabelh3d466374)
 
&emsp;&emsp;[Putting It Altogether"></a>So far we had the following:-  **Linear regression model:** $f_{w,b} (x)=wx+b$ }   -  **Cost function (squared error):** $J(w,b)=\frac{1}{2m}\sum_{i=1}^m {\left(f_{w,b} (x^{(i)} )-y^{(i)} \right)}^2 =\frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} +b-y^{(i)} \right)}^2 $ \end{flushleft}}   \item{\begin{flushleft} \textbf{Gradient descent algorithm:} Repeat until convergence $ w=w-\alpha \frac{\partial }{\partial w}J(w,b) $ and $ b=b-\alpha \frac{\partial }{\partial b}J(w,b) $ \end{flushleft}}\end{itemize}\begin{flushleft}The natural question is how to calculate $ w=w-\alpha \frac{\partial }{\partial w}J(w,b) $ and $ b=b-\alpha \frac{\partial }{\partial b}J(w,b) $?\end{flushleft}\begin{flushleft}Let's start with $ w=w-\alpha \frac{\partial }{\partial w}J(w,b) $. By using sum and chain rules we can find the partial derivative as follows:\end{flushleft}\begin{itemize}\setlength{\itemsep}{-1ex}   \item{\begin{flushleft} \textbf{Sum rule:} $ \frac{\textrm{d}}{\textrm{d}x}\left(f(x)+g(x)\right)=\frac{\textrm{d}}{\textrm{d}x}f(x)+\frac{\textrm{d}}{\textrm{d}x}g(x) $ \end{flushleft}}   \item{\begin{flushleft} \textbf{Chain rule:} $ y=f\left(g(x)\right) $, if we make a substitution $ u=g(x) $, it becomes $ y=f(u) $, then $ \frac{\textrm{d}y}{\textrm{d}x}=\frac{\textrm{d}y}{\textrm{d}u}\cdot \frac{\textrm{d}u}{\textrm{d}x}$ Then, $ \begin{array}{l} w=w-\alpha \frac{\partial }{\partial w}\frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} +b-y^{(i)} \right)}^2 =w-\alpha \frac{1}{2m}\sum_{i=1}^m \frac{\partial }{\partial w}{\left(wx^{(i)} +b-y^{(i)} \right)}^2 =w-\alpha \frac{1}{2m}\sum_{i=1}^m 2\left(wx^{(i)} +b-y^{(i)} \right)\cdot x^{(i)} \newline \Rightarrow w=w-\alpha \frac{1}{m}\sum_{i=1}^m \left(wx^{(i)} +b-y^{(i)} \right)\cdot x^{(i)}  \end{array} $ Similarly, $ w=w-\frac{\alpha }{m}\sum_{i=1}^m \left(wx^{(i)} +b-y^{(i)} \right)\cdot x^{(i)} $ $ \begin{array}{l} b=b-\alpha \frac{\partial }{\partial b}\frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} +b-y^{(i)} \right)}^2 =b-\alpha \frac{1}{2m}\sum_{i=1}^m \frac{\partial }{\partial b}{\left(wx^{(i)} +b-y^{(i)} \right)}^2 =b-\alpha \frac{1}{2m}\sum_{i=1}^m 2\left(wx^{(i)} +b-y^{(i)} \right)\newline \Rightarrow b=b-\alpha \frac{1}{m}\sum_{i=1}^m \left(wx^{(i)} +b-y^{(i)} \right) \end{array} $ Consequently, the gradient descent algorithm for linear regression becomes,1) Randomly initialize $w$ and $b$ 2) Repeat until convergence:- $\displaystyle w=w-\alpha \frac{1}{m}\sum_{i=1}^m \left(f_{w,b} (x^{(i)} )-y^{(i)} \right)\cdot x^{(i)} =w-\alpha \frac{1}{m}\sum_{i=1}^m \left(wx^{(i)} +b-y^{(i)} \right)\cdot x^{(i)} $ \end{flushleft}}   \item{\begin{flushleft} $ \displaystyle b=b-\alpha \frac{1}{m}\sum_{i=1}^m \left(f_{w,b} (x^{(i)} )-y^{(i)} \right)=b-\alpha \frac{1}{m}\sum_{i=1}^m \left(wx^{(i)} +b-y^{(i)} \right)$ where you want to update $w$ and $b$ simultaneously.**As long as the learning rate is chosen properly, gradient descent will find the global minimum for the squared error cost function (as it is convex).**<a id="H_9D1F9310](#putting-it-altogetherbeginparbeginflushleftso-far-we-had-the-followingendflushleftendparbeginitemizesetlengthitemsep1ex-itembeginflushleft-linear-regression-model-endflushleft-itembeginflushleft-cost-function-squared-error-jwbfrac12msumi1m-leftfwb-xi-yi-right2-frac12msumi1m-leftwxi-byi-right2-wwalpha-fracpartial-partial-wjwb-bbalpha-fracpartial-partial-bjwb-wwalpha-fracpartial-partial-wjwb-bbalpha-fracpartial-partial-bjwb-wwalpha-fracpartial-partial-wjwb-fracrmdrmdxleftfxgxrightfracrmdrmdxfxfracrmdrmdxgx-yfleftgxright-ugx-yfu-fracrmdyrmdxfracrmdyrmducdot-fracrmdurmdx-endflushleftenditemizebeginparbeginflushleftthen-endflushleftendparbeginparbeginarrayl-wwalpha-fracpartial-partial-wfrac12msumi1m-leftwxi-byi-right2-walpha-frac12msumi1m-fracpartial-partial-wleftwxi-byi-right2-walpha-frac12msumi1m-2leftwxi-byi-rightcdot-xi-newline-rightarrow-wwalpha-frac1msumi1m-leftwxi-byi-rightcdot-xi-endarrayendparbeginparbeginflushleftsimilarly-endflushleftendparbeginparwwfracalpha-msumi1m-leftwxi-byi-rightcdot-xiendparbeginparbeginarrayl-bbalpha-fracpartial-partial-bfrac12msumi1m-leftwxi-byi-right2-balpha-frac12msumi1m-fracpartial-partial-bleftwxi-byi-right2-balpha-frac12msumi1m-2leftwxi-byi-rightnewline-rightarrow-bbalpha-frac1msumi1m-leftwxi-byi-right-endarrayendparbeginparbeginflushleftconsequently-the-gradient-descent-algorithm-for-linear-regression-becomesendflushleftendparbeginparbeginflushleft1-randomly-initialize-and-endflushleftendparbeginparbeginflushleft2-repeat-until-convergenceendflushleftendparbeginitemizesetlengthitemsep1ex-itembeginflushleft-displaystyle-wwalpha-frac1msumi1m-leftfwb-xi-yi-rightcdot-xi-walpha-frac1msumi1m-leftwxi-byi-rightcdot-xi-displaystyle-bbalpha-frac1msumi1m-leftfwb-xi-yi-rightbalpha-frac1msumi1m-leftwxi-byi-right-endflushleftenditemizebeginparbeginflushleftwhere-you-want-to-update-and-simultaneouslyendflushleftendparbeginparbeginflushleftas-long-as-the-learning-rate-is-chosen-properly-gradient-descent-will-find-the-global-minimum-for-the-squared-error-cost-function-as-it-is-convexendflushleftendparlabelh9d1f9310)
 
&emsp;&emsp;[Running Gradient Descent"></a>If every step of the gradient descent algorithm uses all the training samples it is called **batch gradient descent**. Let's see what happens if we run batch gradient descent algorithm on linear regression problem.Let's run the batch gradient descent algorithm and visualize the linear regression model $f_{w,b} (x)=wx+b$ and cost function $J(w,b)$ for each iteratio](#running-gradient-descentbeginparbeginflushleftif-every-step-of-the-gradient-descent-algorithm-uses-all-the-training-samples-it-is-called-batch-gradient-descent-lets-see-what-happens-if-we-run-batch-gradient-descent-algorithm-on-linear-regression-problemendflushleftendparbeginparbeginflushleftlets-run-the-batch-gradient-descent-algorithm-and-visualize-the-linear-regression-model-and-cost-function-jwb-for-each-iteratio)
 
&emsp;[Local Functions](#H_C771AA3B)
 
&emsp;[References](#H_A0CAEE9B)
 
<!-- End Toc -->

Supervised learning working principle could be summarized with a block diagram as follows.


![image_0.png](./supervised_learning_week1_media/image_0.png)


Here, the key questions is:

-  How are we going to represent $f$? 
-  What is the formula we will use to calculate $f$? 
<a id="H_40B92E36"></a>

# Linear Regression

Let's assume that we have a linear regression application, this would mean that $f_{w,b} (x)=f(x)=wx+b$, where $w$ and $b$ represent the weights and bias, respectively. Why we are choosing $f$ as a linear function (straight line) instead of some non\-linear function curve, parabola etc.? Since a linear function is relatively simple and easy to work with. let's use a line as a foundation. This particular model is called univariate linear regression (single input/feature x).


The big question in any AIML application is "how to measure how well the model fits the data?". Loss function, is our reference point for evaluating the model performance.


Squared error loss function for a univariate linear regression problem could be given by

 $$ J(w,b)=\frac{1}{2m}\sum_{i=1}^m {\left(f_{w,b} (x)-y^{(i)} \right)}^2 =\frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} +b-y^{(i)} \right)}^2 ~~~~(1) $$ 

In the AIML area, the coefficient $1/2m$ is adopted for the sake of simplicity in the calculations.


Let's simplify the cost function for a brief moment and assume that $b=0$. Hence, the cost function in (1) reduces to

 $$ J(w)=\frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} -y^{(i)} \right)}^2 ~~~~(2) $$ 

where $f_w (x)=wx$. Let's assume that the data fits into the following function $y=x$. If we plot the data points, where the number of training examples $m=5$,

```matlab
clear all;
close all;
clc;
clf;

% Let's assume we have the following data points
x = [0.15 1.38 2.21 3.88 4.67]; % features
y = x; % targets

figure
scatter(x,y,'ro','filled');
legend('data points',"Location","northwest");
grid on
xlabel('x')
ylabel('y')
```

![figure_0.png](./supervised_learning_week1_media/figure_0.png)

Now, we can plot $f_w (x)$ for the weights that falls in the range  $-a\le w\le a$, where $a=2$. Note that we already know that the true weight is $w=1$, however, our aim to approximate the true weight value by solely using the available data. If we plot $f_w (x)$ and $J(w)$ with respect to $w$.

```matlab
% Let's scan various values of w and calculate the loss
w_vec = -2:0.1:2; % weights

f = [];
J = [];
for i = 1:numel(w_vec)
    w = w_vec(i);
    
    f(i,:) = w.*x;
    J(i,:) = (1/(2*length(x)))*sum((f(i,:)-y).^2);
end

figure
tiledlayout(1,3)
nexttile
plot(x,f);
xlabel('$x$','Interpreter','latex');
ylabel('$f_w(x)$','Interpreter','latex');
xlim([0 max(x)]);

nexttile
plot(w_vec,J,'b-*','LineWidth',0.7)
xlabel('$w$','Interpreter','latex');
ylabel('$J(w)$','Interpreter','latex');

nexttile
plot(x,f(w_vec==w_vec(J==min(J)),:),'r');
xlabel('$w$','Interpreter','latex');
ylabel('$J(w)$','Interpreter','latex');
legend(['Opt. Model (w = ',num2str(w_vec(J==min(J))),')']);
xlim([0 max(x)]);
```

![figure_1.png](./supervised_learning_week1_media/figure_1.png)

In summary, we have the following:


**Model:** $f_{w,b} (x)=wx+b$ 


**Parameters:** $w,b$ 


**Cost Function:** $J(w,b)=\frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} +b-y^{(i)} \right)}^2$ 


**Objective:** $\underset{w,b}{\mathrm{minimize}} \left\lbrace J(w,b)\right\rbrace =\underset{w,b}{\mathrm{minimize}} \left\lbrace \frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} +b-y^{(i)} \right)}^2 \right\rbrace$ 


Note that minimizing the squared loss is also referred to as **least squares fit**.


Let's go back to the original example and do the same thing without setting $b=0$. Let's look at the house sizes and prices dataset

```matlab
data = importdata("house_sizes_and_prices_dataset.txt"); % Load the data (size (sq. feet) versus price ($w$000's)

% Visualize the data
figure
scatter(data(:,1),data(:,2),'ro','filled');
xlabel("size in feet^2")
ylabel("price in $1000's")
```

![figure_2.png](./supervised_learning_week1_media/figure_2.png)

One possible function is $f_{w,b} (x)=0.06x+50 $w$ 00's (y)")legend(["data points","$0.06x+50 $"],"Interpreter","latex","Location","northwest")```
\begin{center}![figure_3.png](./supervised_learning_week1_media/figure_3.png,width=56.196688409433015em)\end{center}\begin{flushleft}Note that this is not a particularly a good model, on the contrary, this is a pretty bad model since it consistently underestimates the housing prices. Let's look at what the cost function $ J_{w,b} (x)$ looks like for the given scenario.```matlabplotf = false; % Interactively plot f_{w,b}?
% Visualize the cost functionw_vec = -10:2:10; % weightsb_vec = -1e5:1e4:1e5; % bias
f = [];J = [];xPoints = linspace(min(data(:,1)),max(data(:,1)), 5);numColors = numel(w_vec)*numel(b_vec);cmap = lines(numColors);idxc = 0;figurefor idxw = 1:numel(w_vec)    w = w_vec(idxw);    for idxb = 1:numel(b_vec)        b = b_vec(idxb);            f{idxw,idxb} = w.*data(:,1) + b;        J(idxw,idxb) = (1/(2*length(data(:,1))))*sum((f{idxw,idxb}-data(:,2)).^2);
        % Animated line        if plotf            yPoints = w.*xPoints + b;            h = animatedline("Color",cmap(mod(idxc, size(cmap,1)) + 1, :));            xlabel("normalized size (x)")            ylabel("normalized price (y)")            for k = 1:length(xPoints)                addpoints(h, xPoints(k), w*xPoints(k) + b);                drawnow            end            idxc = idxc + 1;        end    endend
figuretiledlayout(1, 2)nexttilesurf(b_vec, w_vec, J)xlabel('b')ylabel('w')zlabel('J(w,b)')title('J(w,b) versus w and b')view([247.80 34.00])
nexttilecontour(b_vec, w_vec, J, 30)xlabel('b')ylabel('w')colorbar```
![figure_4.png](./supervised_learning_week1_media/figure_4.png,width=56.196688409433015em)Rather than manually trying to find the minimum value of the contour plot by picking a random ( $w,b$ ) pair, which is time consuming and wouldn't work with more complex problems with very complex cost functions, we would look for an algorithm that can find the minimum ( $w,b$ ) pair for that makes the cost function automatically/automagically minimum. There is an algorithm that does that called **gradient descent (GD)**, which is the one of the most important algorithm is AIML. Gradient descent and its variations is used not only in linear regression but almost every complex problem in AIML area.**Note:**-  The cost equation provides a measure of how well your predictions match your training data. }   -  Minimizing the cost can provide optimal values of $w,b$. <a id="H_BD5727CD"></a>

# Gradient Descent (GD) AlgorithmPlease note that the **squared loss cost function** ensures that the 'error surface' is convex like a soup bowl.It will always have a minimum that can be reached by following the gradient in all dimensions.-  Why? The overview of what we are aiming with gradient descent;-  Have some function $J(w,b) $ \end{flushleft}}   \item{\begin{flushleft} We want to minimize this function such that $ \underset{w,b}{\mathrm{min}} \left\lbrace J(w,b)\right\rbrace =\underset{w,b}{\mathrm{min}} \left\lbrace \frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} +b-y^{(i)} \right)}^2 \right\rbrace $, where the $ (w,b)$ pair that minimizes the given function becomes our solution. It turns out that the gradient descent algorithm can minimize not only linear regression cost function but any function that is convex. <a id="H_5965BE43"></a>

### Methodology1.  We start with some initial guesses of $w$ and $b$ (random initialization). For instance, in linear regression it won't matter what the initial values are. Hence, the common choice is setting them to zero $w=0$ and $b=0$. }   1.  Keep changing the parameters $w$ and $b$ a bit every time/iteration to reduce the cost $J(w,b)$. }   1.  Repeat step 2 until (hopefully) we settle at or near minimum. Note that for some cost functions that might not have a bowl shape (non-convex) it is possible to have more than 1 minimum value. For instance,```matlab[X,Y,Z] = peaks(40);figuresurf(X,Y,Z)colormap turboxlabel('$b$','Interpreter','latex');ylabel('$w$','Interpreter','latex');zlabel('$J(w,b)$','Interpreter','latex');```
![figure_5.png](./supervised_learning_week1_media/figure_5.png)What gradient descent algorithm does is it sits on a location and looks around and asks; "if I wanted to take a baby step in one direction to go downhill as quickly as possible to one of these valleys what direction should I choose to take a step?"-  If you want to walk down the hill as efficiently as possible it turns out that the best direction is the direction towards the **steepest descent**. In the following figure we can see two different randomly initiated gradient descent algorithms, which converge to different local minimum points.![image_1.png](./supervised_learning_week1_media/image_1.png)<a id="H_81A0C8E0"></a>

## Implementation-  In each step, update the weight(s) as follows: $w=w-\alpha \frac{\partial }{\partial w}J(w,b)$. It means that update the $w$ such that take the current value of the $w$ and subtract the scaled derivative of the cost function with respect to $w$ (Here, the equal sign represents the assignment operation). The parameter $\alpha $ is the \textbf{learning rate}, where $ 0\le \alpha \le 1 $. The learning rate controls how big a step you take towards the steepest descent. \end{flushleft}}   \item{\begin{flushleft} In each step, also update the bias as follows: $ b=b-\alpha \frac{\partial }{\partial b}J(w,b)$. }   -  In gradient descent, we need to repeat the above steps until convergence. What is convergence? Convergence means that you reached the point at local/global minimum, where the parameters no longer change much in each step. }   -  There is one more subtle detail for correctly implementing the gradient descent; you will be updating the parameters $w$ and $b$, this update takes place for both parameters. In gradient descent, you want to simultaneously update the parameters $w$ and $b$. **Correct way (simultaneous update) of implementing gradient descent:**\begin{enumerate}\setlength{\itemsep}{-1ex}   \item{ $\displaystyle w_{\textrm{tmp}} =w-\alpha \frac{\partial }{\partial w}J(w,b) $ \end{flushleft}}   \item{\begin{flushleft} $ \displaystyle b_{\textrm{tmp}} =b-\alpha \frac{\partial }{\partial b}J(w,b) $ \end{flushleft}}   \item{\begin{flushleft} $ \displaystyle w=w_{\textrm{tmp}} $ \end{flushleft}}   \item{\begin{flushleft} $ \displaystyle b=b_{\textrm{tmp}} $ \end{flushleft}}\end{enumerate}\begin{flushleft}where $ w_{\textrm{tmp}} $ and $ b_{\textrm{tmp}}$ represents the temporary values for $w$ and $b$, respectively.**Incorrect way of implementing gradient descent:**\begin{enumerate}\setlength{\itemsep}{-1ex}   \item{ $\displaystyle w_{\textrm{tmp}} =w-\alpha \frac{\partial }{\partial w}J(w,b) $ \end{flushleft}}   \item{\begin{flushleft} $ \displaystyle w=w_{\textrm{tmp}} $ \end{flushleft}}   \item{\begin{flushleft} $ \displaystyle b_{\textrm{tmp}} =b-\alpha \frac{\partial }{\partial b}J(w,b) $ \end{flushleft}}   \item{\begin{flushleft} $ \displaystyle b=b_{\textrm{tmp}} $ \end{flushleft}}\end{enumerate}\begin{flushleft}It turns out that even if you implement the gradient descent incorrectly, it would still work more less the same as the correct implementation, however, it is important to implement the algorithm according to the correct theoretical insight, which will ensure our long term success and will avoid unexpected problems in the future.\end{flushleft}\label{H_99731A70}\matlabheadingtwo{Gradient Descent Intuition}\begin{flushleft}In order to understand why the weight and bias update formulas\end{flushleft}\begin{flushleft}$ w=w-\alpha \frac{\partial }{\partial w}J(w,b) $, \end{flushleft}\begin{flushleft}$ b=b-\alpha \frac{\partial }{\partial b}J(w,b)$ are used. Let's start with a simpler example, which contains one parameter. Thus, the cost function is in the form of $J(w)$ and the gradient descent based weight update formula becomes $w=w-\alpha \frac{\partial }{\partial w}J(w) $. Let's look at the previous simple single parameter cost function example and try to understand:\end{flushleft}\begin{itemize}\setlength{\itemsep}{-1ex}   \item{\begin{flushleft} What the learning rate ($ \alpha $) does \end{flushleft}}   \item{\begin{flushleft} What the derivative part ($ \frac{\partial }{\partial w}J(w,b) $ or $ \frac{\partial }{\partial b}J(w,b) $) does \end{flushleft}}   \item{\begin{flushleft} Why they are multiplied \end{flushleft}}   \item{\begin{flushleft} Why the multiplication is subtracted from the initial value and why it make sense \end{flushleft}}\end{itemize}\begin{flushleft}Since this example has only one parameter, the gradient descent algorithm looks like $ w=w-\alpha \frac{\partial }{\partial w}J(w)$. And we try to minimize the cost function by only setting $w$ such that $\underset{w}{\mathrm{min}} \left\lbrace J(w)\right\rbrace $.\end{flushleft}\begin{flushleft}Let's initialize the gradient descent in some random points similar to the previous example with two parameters. In order to plot the gradient of the cost function, we should first calculate it. The cost function is given by\end{flushleft}\begin{flushleft}$ J(w)=\frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} -y^{(i)} \right)}^2 $, where we need to calculate the $ \frac{\partial }{\partial w}J(w)=\frac{\partial }{\partial w}\left(\frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} -y^{(i)} \right)}^2 \right)$. Since we know that the derivative operation is distributive on the sum operation, $ \frac{\partial }{\partial w}J(w)=\frac{1}{2m}\sum_{i=1}^m \frac{\partial }{\partial w}{\left(wx^{(i)} -y^{(i)} \right)}^2 =\frac{1}{2m}\sum_{i=1}^m 2\left(wx^{(i)} -y^{(i)} \right)\cdot x^{(i)} =\frac{1}{m}\sum_{i=1}^m \left(wx^{(i)} -y^{(i)} \right)\cdot x^{(i)} ~~(3) $ Similarly, we can also obtain the rate of change of the cost function w.r.t. $b$ as follows: $ \frac{\partial }{\partial b}J(w)=\frac{1}{2m}\sum_{i=1}^m \frac{\partial }{\partial w}{\left(wx^{(i)} -y^{(i)} \right)}^2 =\frac{1}{2m}\sum_{i=1}^m 2\left(wx^{(i)} -y^{(i)} \right)=\frac{1}{m}\sum_{i=1}^m \left(wx^{(i)} -y^{(i)} \right)~~(4) $ -  Let's visualize the effect of the selected weight on the $J(w,b) $ and $ \frac{\partial }{\partial w}J(w)$: ```matlabw_selected = -2.5;hInteractiveGradPlot(w_selected);```
![figure_6.png](./supervised_learning_week1_media/figure_6.png)<a id="H_53BAA125"></a>

## Selection of the Learning RateSelection of the learning rate, $\alpha $, is also really important in gradient descent implementation.\end{flushleft}\begin{flushleft}To understand what $ \alpha $ is doing in gradient descent algorithm, let's explore the case where $ \alpha $ is either too small or too large.\end{flushleft}\begin{itemize}\setlength{\itemsep}{-1ex}   \item{\begin{flushleft} If $ \alpha$ is too small: The gradient descent will work but you end up increasing/decreasing $w$ but **very slowly**.It means that you will need a lot of steps (it will take a very long time) to reach to the minimum point. }   \item{ If $\alpha$ is too large: You update $w$ with a giant step and the cost might even increase over iterations (means you might go further from the minimum point). In this case gradient descent will overshoot and never reach the minimum, which means that it fails to **converge** to a solution but also **diverge**. }\end{itemize}![image_2.png](./supervised_learning_week1_media/image_2.png)-  If you are at a local minimum already, further gradient descent steps won't change anything since the derivative of the cost function equals to zero, which means that $w$ won't be updated. It is important note that as we approach the local minimum, gradient descent will automatically take smaller steps due to the smaller derivative of the cost function. Therefore, gradient descent can reach local minimum without decreasing the learning rate. **In other words, gradient descent can reach local minimum with a fixed learning rate.**We can use gradient descent algorithm to minimize any cost function $J$, not just the mean squared error cost function that we are using for linear regression.<a id="H_3D466374"></a>

## Putting It AltogetherSo far we had the following:-  **Linear regression model:** $f_{w,b} (x)=wx+b$ }   -  **Cost function (squared error):** $J(w,b)=\frac{1}{2m}\sum_{i=1}^m {\left(f_{w,b} (x^{(i)} )-y^{(i)} \right)}^2 =\frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} +b-y^{(i)} \right)}^2 $ \end{flushleft}}   \item{\begin{flushleft} \textbf{Gradient descent algorithm:} Repeat until convergence $ w=w-\alpha \frac{\partial }{\partial w}J(w,b) $ and $ b=b-\alpha \frac{\partial }{\partial b}J(w,b) $ \end{flushleft}}\end{itemize}\begin{flushleft}The natural question is how to calculate $ w=w-\alpha \frac{\partial }{\partial w}J(w,b) $ and $ b=b-\alpha \frac{\partial }{\partial b}J(w,b) $?\end{flushleft}\begin{flushleft}Let's start with $ w=w-\alpha \frac{\partial }{\partial w}J(w,b) $. By using sum and chain rules we can find the partial derivative as follows:\end{flushleft}\begin{itemize}\setlength{\itemsep}{-1ex}   \item{\begin{flushleft} \textbf{Sum rule:} $ \frac{\textrm{d}}{\textrm{d}x}\left(f(x)+g(x)\right)=\frac{\textrm{d}}{\textrm{d}x}f(x)+\frac{\textrm{d}}{\textrm{d}x}g(x) $ \end{flushleft}}   \item{\begin{flushleft} \textbf{Chain rule:} $ y=f\left(g(x)\right) $, if we make a substitution $ u=g(x) $, it becomes $ y=f(u) $, then $ \frac{\textrm{d}y}{\textrm{d}x}=\frac{\textrm{d}y}{\textrm{d}u}\cdot \frac{\textrm{d}u}{\textrm{d}x}$ Then, $ \begin{array}{l} w=w-\alpha \frac{\partial }{\partial w}\frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} +b-y^{(i)} \right)}^2 =w-\alpha \frac{1}{2m}\sum_{i=1}^m \frac{\partial }{\partial w}{\left(wx^{(i)} +b-y^{(i)} \right)}^2 =w-\alpha \frac{1}{2m}\sum_{i=1}^m 2\left(wx^{(i)} +b-y^{(i)} \right)\cdot x^{(i)} \newline \Rightarrow w=w-\alpha \frac{1}{m}\sum_{i=1}^m \left(wx^{(i)} +b-y^{(i)} \right)\cdot x^{(i)}  \end{array} $ Similarly, $ w=w-\frac{\alpha }{m}\sum_{i=1}^m \left(wx^{(i)} +b-y^{(i)} \right)\cdot x^{(i)} $ $ \begin{array}{l} b=b-\alpha \frac{\partial }{\partial b}\frac{1}{2m}\sum_{i=1}^m {\left(wx^{(i)} +b-y^{(i)} \right)}^2 =b-\alpha \frac{1}{2m}\sum_{i=1}^m \frac{\partial }{\partial b}{\left(wx^{(i)} +b-y^{(i)} \right)}^2 =b-\alpha \frac{1}{2m}\sum_{i=1}^m 2\left(wx^{(i)} +b-y^{(i)} \right)\newline \Rightarrow b=b-\alpha \frac{1}{m}\sum_{i=1}^m \left(wx^{(i)} +b-y^{(i)} \right) \end{array} $ Consequently, the gradient descent algorithm for linear regression becomes,1) Randomly initialize $w$ and $b$ 2) Repeat until convergence:- $\displaystyle w=w-\alpha \frac{1}{m}\sum_{i=1}^m \left(f_{w,b} (x^{(i)} )-y^{(i)} \right)\cdot x^{(i)} =w-\alpha \frac{1}{m}\sum_{i=1}^m \left(wx^{(i)} +b-y^{(i)} \right)\cdot x^{(i)} $ \end{flushleft}}   \item{\begin{flushleft} $ \displaystyle b=b-\alpha \frac{1}{m}\sum_{i=1}^m \left(f_{w,b} (x^{(i)} )-y^{(i)} \right)=b-\alpha \frac{1}{m}\sum_{i=1}^m \left(wx^{(i)} +b-y^{(i)} \right)$ where you want to update $w$ and $b$ simultaneously.**As long as the learning rate is chosen properly, gradient descent will find the global minimum for the squared error cost function (as it is convex).**<a id="H_9D1F9310"></a>

## Running Gradient DescentIf every step of the gradient descent algorithm uses all the training samples it is called **batch gradient descent**. Let's see what happens if we run batch gradient descent algorithm on linear regression problem.Let's run the batch gradient descent algorithm and visualize the linear regression model $f_{w,b} (x)=wx+b$ and cost function $J(w,b)$ for each iteration

```matlab
% Batch gradient descent implementation
x = 1:5;
y = x-2;
%x = (data(:,1)-min(data(:,1)))/(max(data(:,1))-min(data(:,1)));
%y = (data(:,2)-min(data(:,2)))/(max(data(:,2))-min(data(:,2)));
alpha = 0.015; % learning rate is between [0 1], best = 0.015, too large = 0.17 too small = 1e-5.
verbose = true;

 
[J_vec,w_vec,b_vec] = batchGradientDescent(x,y,alpha,verbose);
```

```matlabTextOutput
Iteration #0 | Cost: 249.5352
Iteration #20 | Cost: 5.3772
Iteration #40 | Cost: 4.7691
Iteration #60 | Cost: 4.3085
Iteration #80 | Cost: 3.8924
Iteration #100 | Cost: 3.5165
Iteration #120 | Cost: 3.1769
Iteration #140 | Cost: 2.8701
Iteration #160 | Cost: 2.5929
Iteration #180 | Cost: 2.3425
Iteration #200 | Cost: 2.1163
Iteration #220 | Cost: 1.9119
Iteration #240 | Cost: 1.7273
Iteration #260 | Cost: 1.5605
Iteration #280 | Cost: 1.4098
Iteration #300 | Cost: 1.2737
Iteration #320 | Cost: 1.1507
Iteration #340 | Cost: 1.0395
Iteration #360 | Cost: 0.93915
Iteration #380 | Cost: 0.84845
Iteration #400 | Cost: 0.76652
Iteration #420 | Cost: 0.69249
Iteration #440 | Cost: 0.62562
Iteration #460 | Cost: 0.5652
Iteration #480 | Cost: 0.51062
Iteration #500 | Cost: 0.46131
Iteration #520 | Cost: 0.41676
Iteration #540 | Cost: 0.37651
Iteration #560 | Cost: 0.34015
Iteration #580 | Cost: 0.3073
Iteration #600 | Cost: 0.27763
Iteration #620 | Cost: 0.25082
Iteration #640 | Cost: 0.2266
Iteration #660 | Cost: 0.20471
Iteration #680 | Cost: 0.18494
Iteration #700 | Cost: 0.16708
Iteration #720 | Cost: 0.15095
Iteration #740 | Cost: 0.13637
Iteration #760 | Cost: 0.1232
Iteration #780 | Cost: 0.1113
Iteration #800 | Cost: 0.10055
Iteration #820 | Cost: 0.090844
Iteration #840 | Cost: 0.082071
Iteration #860 | Cost: 0.074146
Iteration #880 | Cost: 0.066985
Iteration #900 | Cost: 0.060517
Iteration #920 | Cost: 0.054672
Iteration #940 | Cost: 0.049393
Iteration #960 | Cost: 0.044623
Iteration #980 | Cost: 0.040313
Iteration #1000 | Cost: 0.03642
Iteration #1020 | Cost: 0.032903
Iteration #1040 | Cost: 0.029726
Iteration #1060 | Cost: 0.026855
Iteration #1080 | Cost: 0.024262
Iteration #1100 | Cost: 0.021919
Iteration #1120 | Cost: 0.019802
Iteration #1140 | Cost: 0.01789
Iteration #1160 | Cost: 0.016162
Iteration #1180 | Cost: 0.014601
Iteration #1200 | Cost: 0.013191
Iteration #1220 | Cost: 0.011917
Iteration #1240 | Cost: 0.010766
Iteration #1260 | Cost: 0.0097267
Iteration #1280 | Cost: 0.0087874
Iteration #1300 | Cost: 0.0079388
Iteration #1320 | Cost: 0.0071721
Iteration #1340 | Cost: 0.0064795
Iteration #1360 | Cost: 0.0058538
Iteration #1380 | Cost: 0.0052885
Iteration #1400 | Cost: 0.0047778
Iteration #1420 | Cost: 0.0043164
Iteration #1440 | Cost: 0.0038995
Iteration #1460 | Cost: 0.003523
Iteration #1480 | Cost: 0.0031827
Iteration #1500 | Cost: 0.0028754
Iteration #1520 | Cost: 0.0025977
Iteration #1540 | Cost: 0.0023468
Iteration #1560 | Cost: 0.0021202
Iteration #1580 | Cost: 0.0019155
Iteration #1600 | Cost: 0.0017305
Iteration #1620 | Cost: 0.0015634
Iteration #1640 | Cost: 0.0014124
Iteration #1660 | Cost: 0.001276
Iteration #1680 | Cost: 0.0011528
Iteration #1700 | Cost: 0.0010414
==================================
BGD Stopped: minimized cost function (0.001000.2)
```

```matlab
numIter = numel(J_vec)-1;
```

```matlab
idxIter =445;
plotMode = "cumulative";
hInteractiveLinearRegression(x,y,w_vec,b_vec,J_vec,idxIter,plotMode)
set(gcf,'position',[0 0 1100 600]);
```

![figure_7.png](./supervised_learning_week1_media/figure_7.png,width=110.38635223281486em)
<a id="H_C771AA3B"></a>

# Local Functions
```matlab
function [J_vec,w_vec,b_vec] = batchGradientDescent(x,y,alpha,verbose)

% Step 1: Randomly initialize the parameters w and b
stopFlag = false;
m = length(x);
iter = 0; % iteration number
J_vec = [];
w_vec = [];
b_vec = [];
w = randsrc(1,1,-10:0.001:10);
b = randsrc(1,1,-10:0.001:10);
condStop = 1e-3;
maxIter = 1e4; % Maximum number of iterations that will terminate the search

% Step 2: Update w and b by using BGD until the convergence criteria is met
while ~stopFlag
    dJdw = 0;
    dJdb = 0;
    
    for i = 1:m
        dJdw = dJdw + (1/m)*( (w*x(i)+b-y(i))*x(i) );
        dJdb = dJdb + (1/m)*( (w*x(i)+b-y(i)) );
    end
    
    w = w - alpha*dJdw; % update w
    b = b - alpha*dJdb; % update b

    w_vec = cat(1,w_vec,w); % record w history
    b_vec = cat(1,b_vec,b); % record b history

    % Calculate the cost function
    J_tmp = 0;
    for i = 1:m
        J_tmp = J_tmp + (w*x(i)+b-y(i))^2;
    end
    J = (1/(2*m))*J_tmp;
    J_vec = cat(1,J_vec,J);

    if verbose
        if mod(iter,20) == 0
            disp(['Iteration #',num2str(iter),' | Cost: ',num2str(J)])
        end
    end

    % if the cost function is small enough stop iterating (convergence)
    % OR if the number of iterations reaches to a limit stop iterating (divergence)
    if iter == maxIter - 1 || J <= condStop 
        stopFlag = true;
        disp("==================================")
        if iter == maxIter - 1
            disp(['BGD Stopped: max number of iterations ',sprintf('(%d)',iter)])
        else
            disp(['BGD Stopped: minimized cost function ',sprintf('(%f.2)',J)])
        end
    end

    iter = iter + 1; % update the iteration number
end

end
```
<a id="H_A0CAEE9B"></a>

# References

\[1\] Coursera \- "[Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning/)"

