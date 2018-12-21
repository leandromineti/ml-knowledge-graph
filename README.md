# Machine Learning Knowledge Graph

![](cover.png)

Besides enumerating the topics related to Machine Learning and Statistical Modeling, I
am currently building an [interactive graph](https://leandromineti.github.io/ml-knowledge-graph/) 
to illustrate its relations. At the end of the page you can find a list of **free**
online resources covering the subjects and a FAQ.

## Curriculum 

- Logic
    - Gödel's incompleteness theorems
- Set theory
    - Empty set
    - Finite and infinite sets
    - Operations on sets
        - Complement
        - Union
        - Intersection
- Linear Algebra
    - Matrix transformation
    - Eigenstuff
    - Matrix decomposition
        - Singular Value Decomposition
        - Non-negative Matrix Factorization
- Calculus
    - Limits
    - Derivatives
        - Partial derivatives
            - Gradient
    - Integrals
    - Taylor series
        - Maclaurin series
    - Fourrier series
        - Fourrier transform
            - Laplace transform
- Optimization
    - Iterative methods
        - Newton's method
        - Gradient descent
        - Expectation Maximization
            - Baum-Welch algorithm
    - Heuristics
        - Evolutionary algorithms
- Measure theory
    - Sigma-algebra
- Probability
    - Sample Space
    - Kolmogorov axioms
    - Cox's theorem
    - Relative frequency and probability
    - Counting methods
        - Multiplication rule
        - Permutation
        - Combination and Binomial coefficient
        - Arrangement
    - Conditional probability
    - Bayes' Theorem
        - Posterior probability distribution
    - Random Variables
        - Algebra of random variables
        - Expected value
        - Variance
        - Distributions
            - Exponential family
                - Normal distribution
                - Bernoulli distribution
            - Moment-generating function
                - Characteristic function
            - Multivariate distributions
                - Joint distribution
                - Marginal distribution
                - Conditional distribution
    - Probability inequalities
        - Chebyshev's inequality
        - Bernstein inequalities
            - Chernoff bound
            - Hoeffding's inequality
- Statistics
    - Sampling distribution
    - Law of large numbers
    - Central Limit Theorem
    - Resampling
        - Jacknife
        - Bootstrap
    - Monte Carlo method
    - Likelihood function
    - Random Field
        - Stochastic process
            - Time-series analysis
        - Markov Chain
    - Inference
        - Hypothesis testing
            - ANOVA
        - Survival analysis
        - Estimators
            - Mean Square Error
            - Bias-variance tradeoff
        - Multivariate analysis
            - Covariance matrix
            - Dimensionality reduction
                - Feature selection
                    - Filter methods
                    - Wrapper methods
                    - Embedded methods
                - Feature extraction
                    - Linear
                        - Principal Component Analysis
                        - Linear Discriminant Analysis
                    - Nonlinear
                        - t-SNE
                        - UMAP
            - Factor Analysis
        - Mixture models
            - Method of moments
            - Spectral method
        - Parametric inference
            - Regression
                - Linear regression
                - Quantile regression
                - Autoregressive models
                - Generalized Linear Models
                    - Logistic regression
                    - Multinomial regression
        - Bayesian Inference
            - Maximum a posteriori estimation
            - MCMC
            - Variational inference
        - Probabilistic Graphical Models
            - Bayesian Networks
                - Hidden Markov Models
                    - Maximum-entropy Markov model
            - Markov Random Field
                - Boltzmann machine
            - Latent Dirichlet Allocation
            - Conditional Random Field
        - Nonparametric inference
            - Additive models
                - Generalized additive models
            - Kernel density estimation
        - Generative and discriminative models
- Machine Learning
    - Statistical Learning Theory
        - Vapnik-Chervonenkis theory
        - Hypothesis set
            - No free lunch theorem 
        - Regularization
            - LASSO
            - Ridge
            - Elastic Net
            - Early stopping
            - Dropout
    - Cross-validation
        - Hyperparameter optimization
        - Automated Machine Learning
    - k-NN
    - Naive Bayes
    - Support Vector Machines
        - Kernel trick
    - Decision trees
        - Random Forest
    - Neural Networks
        - Training
            - Backpropagation
            - Activation function
                - Sigmoid
                - Softmax
                - Tanh
                - ReLU
        - Architecture
            - Feedforward networks
                - Perceptron
                - Multilayer perceptron
                    - Convolutional Neural Networks
                        - Deep Q-Learning
                        - Temporal Convolutional Networks
                - Autoencoder
                    - Variational autoencoder
            - Recurrent networks
                - LSTM
                - Gated Recurrent Unit
                - Neural Turing Machine
                - Attention mechanism
            - Restricted Boltzmann machine
                - Deep Belief Network
    - Adversarial Machine Learning
        - Generative Adversarial Networks
    - Ensemble
        - Bagging
        - Boosting
        - Stacking
    - Meta-learning
- Information Theory
    - Entropy
        - Maximum entropy models
    - Kullback–Leibler divergence
    - Signal processing
        - Kalman filter
- Computer Science
    - Algorithms
        - Computability
            - Church–Turing thesis
        - Complexity
- Artificial Intelligence
    - Cognitive Science
    - Cybernetics
        - Control theory
    - Symbolic
        - Logic-based
    - Natural Language Processing
        - Information Retrieval
            - Set-theoretic models
                - Standard Boolean model
                - Extended Boolean model
                - Fuzzy retrieval
            - Algebraic models
            - Probabilistic models
        - Turing test
        
## **Free** resources and references

### Mathematics

- Tool: [Khan Academy](https://www.khanacademy.org/)

#### Linear Algebra

- Playlist: [Essence of Linear Algebra by 3blue1brown](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- Playlist: [MIT 18.06 - Linear Algebra](https://www.youtube.com/watch?v=ZK3O402wf1c&list=PLE7DDD91010BC51F8)
- Course: [Computational Linear Algebra](https://www.fast.ai/2017/07/17/num-lin-alg/)

#### Calculus

- Article: [The Matrix Calculus you need for Deep Learning](https://arxiv.org/pdf/1802.01528.pdf)
- Playlist: [Essence of Calculus by 3blue1brown](https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

### Statistics

- Tool: [Seeing Theory](https://seeing-theory.brown.edu/)

#### Uncertainty and hypothesis testing

- Article: [The hacker's guide to uncertainty](https://erikbern.com/2018/10/08/the-hackers-guide-to-uncertainty-estimates.html)
- Article: [There is only one test!](http://allendowney.blogspot.com/2011/05/there-is-only-one-test.html)
- Article: [There is still only one test](http://allendowney.blogspot.com/2016/06/there-is-still-only-one-test.html)

#### Bayesian inference

- Article: [Frequentism and Bayesianism: A Practical Introduction](http://jakevdp.github.io/blog/2014/03/11/frequentism-and-bayesianism-a-practical-intro/)
- Article: [Towards A Principled Bayesian Workflow](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html)

#### Generative vs. Discriminative models

- Article: [On Discriminative vs. Generative classifiers: A comparison of logistic regression and naive Bayes](http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)
- Article: [Generative and discriminative classifiers: Naive Bayes and Logistic Regression](http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf)

### Information Theory

- Article: [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)

### Machine Learning

- Article: [Statistical Modeling: the two cultures](http://www2.math.uu.se/~thulin/mm/breiman.pdf)
- Article: [A visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
- Article: [Model tuning and the Bias-Variance tradeoff](http://www.r2d3.us/visual-intro-to-machine-learning-part-2/)
- Tool: [Arxiv Sanity Preserver](http://www.arxiv-sanity.com/)
- Tool: [Google Colaboratory](https://colab.research.google.com)
- Tool: [Kaggle](https://www.kaggle.com/)
- Tool: [Seedbank: Collection of Interactive Machine Learning Examples](https://research.google.com/seedbank/)
- Course: [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)
- Course: [Machine Learning for Coders](https://course.fast.ai/ml)
- Book: [An Introduction to Statistical Learning](https://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf)
- Book: [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf)
- Journal: [Distill](https://distill.pub/)

#### Statistical Learning Theory

- Playlist: [Caltech CS 156 - Learning from data](https://www.youtube.com/watch?v=mbyG85GZ0PI&list=PLD63A284B7615313A)

#### Neural Networks and Deep Learning

- Playlist: [Neural Networks by 3blue1brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- Tool: [A Neural Network Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.54960&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
- Course: [Deep Learning for Coders](https://course.fast.ai/index.html)
- Course: [Deep Learning for Coders 2](https://course.fast.ai/part2.html)
- Book: [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- Book: [Deep Learning](https://www.deeplearningbook.org/)

#### Ensemble

- Article: [Kaggle Ensembling Guide](https://mlwave.com/kaggle-ensembling-guide/)

#### Time Series

- Book: [Forecasting: Principles and Practice](https://otexts.org/fpp2/)

### Optimization

- Article: [How optimization for machine learning works](https://brohrer.github.io/how_optimization_works_1.html)
- Article: [Learning to Optimize with Reinforcement Learning](https://bair.berkeley.edu/blog/2017/09/12/learning-to-optimize-with-rl/)
- Blog: [Off The Convex Path](http://www.offconvex.org/)

## FAQ

- **Why make a distinction between Machine Learning and Artificial Intelligence?**

I followed the approach explored by Russel and Norvig [1]. In that sense, Artificial
Intelligence is a broader field that encompasses Machine Learning. 

### Artificial Intelligence

- Course: [Text as Data](https://cbail.github.io/textasdata/Text_as_Data.html)

### References

[1] Russell, S. J., & Norvig, P. (2016). Artificial intelligence: a modern approach. Malaysia; Pearson Education Limited,.
