# Neural Network From Scratch
This is a small'ish project aiming to create a simple/basic neural network which utilitses backpropagation. 

I originally planned to make a game in Unity using my own simple neural network (which was created on the spot). However, I figured that things may a little more complicated than that, so changed focus onto creating a tiny simple neural network framework.

Note: that my knowledge of ML is quite limited and am quite interested in learning.

## Built using
- C# .NET

## Contents
- A neural network struct
  - Holds variable number of layers
  - Allows predictions via forward propagation
  - Allows construction of: 
    - a random neural network
    - a similiar neural network, with random changes
    - a neural network, with changes determined by backpropagation training
- A layer struct
  - Holds weight between edge (input,output) nodes
  - Holds bias weight *note that the last inputNode index in "weights" represents the bias
  - Allows predictions via forward propagation
  - Allows construction of 
    - a random layer
    - a similiar layer, with random changes
    - a layer, with changes determined by nudges which is derrived from backpropagation training

## Author
Mark Connelly
