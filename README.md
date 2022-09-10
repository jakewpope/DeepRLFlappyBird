# Testing Neural Network Architectures for FlappyBird

This work implements different variations of convolutional neural networks to learn to play the game Flappy Bird. The architectures tested in this work are based on previous works deep reinforcement learning and other machine learning projects. These studies showed a convolutional neural network based on the image classification architecture AlexNet is effective at learning the game Flappy Bird with deep Q-learning.

This project is based on the tutorial found at https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial
and the corresponding code at https://github.com/nevenp/dqn_flappy_bird

##### Dependencies:
* Python 3.6.5
* Run `pip3 install -r requirements.txt` to install dependencies.

##### How to run:
* Run `python dqn-alexnet.py test` to run pretrained neural network model.
* Run `python dqn-alexnet.py train` to train the model from the beginning. You can also increase FPS in game/flappy_bird.py script for faster training.

* Run `python dqn-deepmind.py test` to run pretrained neural network model.
* Run `python dqn-deepmind.py train` to train the model from the beginning. You can also increase FPS in game/flappy_bird.py script for faster training.

* Run `python dqn-random.py test` to run pretrained neural network model.
* Run `python dqn-random.py train` to train the model from the beginning. You can also increase FPS in game/flappy_bird.py script for faster training.

References:
* https://github.com/sourabhv/FlapPyBird
* https://github.com/yenchenlin/DeepLearningFlappyBird -> modified FlapPyBird game engine adjusted for reinforcement learning is used from this TensorFlow project
* https://ai.intel.com/demystifying-deep-reinforcement-learning/
* https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
* https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
* http://cs229.stanford.edu/proj2015/362_report.pdf
* https://en.wikipedia.org/wiki/Convolutional_neural_network
* https://en.wikipedia.org/wiki/Reinforcement_learning
* https://en.wikipedia.org/wiki/Markov_decision_process
* http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
* https://en.wikipedia.org/wiki/Flappy_Bird
* https://pytorch.org/
