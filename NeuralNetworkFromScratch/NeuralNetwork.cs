using System.Collections.Generic;
using System;


namespace NeuralNetworks
{
	public struct NeuralNetwork
	{
		/*
			Biases, 1*weight to each node
			  |     |
			  V     V

			(1) _  (1) _
			     \      \
			var1 - node - output1
			     X      X 
			var2 - node - output2
			     X      X 
			var3 - node - output3

					^		^
					|		|
					Has Relu
		*/

		////////////////////////////////// FIELDS

		private Layer[] layers; // (excluding input layer)

		////////////////////////////////// PROPERTIES

		public int InputSize
		{
			get
			{
				return layers[0].InputSize;
			}
		}

		public int OutputSize
		{
			get
			{
				return layers[layers.Length - 1].OutputSize;
			}
		}

		public int NumFunctionalLayers
		{
			get
			{
				return layers.Length;
			}
		}

		public int[] Shape
		{
			get
			{
				int[] shape = new int[layers.Length + 1];

				// Input Layer
				shape[0] = layers[0].InputSize;

				// Hidden and Output Layers
				for (int i = 0; i < layers.Length; i++)
					shape[i+1] = layers[i].OutputSize;
				
				return shape;
			}
		}

		////////////////////////////////// CONSTRUCTORS

		public NeuralNetwork(Layer[] layers)
		{
			this.layers = layers;
		}

		////////////////////////////////// PUBLIC

		public float[] Predict(float[] input)
		{
			// Temporary value
			float[] nextResult = input;

			// Pass through each layer
			for (int i = 0; i < layers.Length; i++)
				nextResult = layers[i].ForwardsPropagate(nextResult);

			return nextResult;
		}

		////////////////////////////////// PRIVATE

		////////////////////////////////// STATIC

		public static NeuralNetwork RandomNetwork(int[] shape, int[] weightRange)
		{
			// Initialise List
			Layer[] newLayers = new Layer[shape.Length - 1];
			
			// For each layer
			for (int i = 0; i < shape.Length - 1; i++)
			{
				int inputSize = shape[i];
				int outputSize = shape[i + 1];
				newLayers[i] = Layer.RandomLayer(inputSize, outputSize, weightRange);
			}

			return new NeuralNetwork(newLayers);
		}

		public static NeuralNetwork SimiliarNetwork(NeuralNetwork network, int[] weightRange, float percentChange = 0.3f)
		{
			// Initialise List
			Layer[] newLayers = new Layer[network.NumFunctionalLayers];

			// For each layer
			for (int i = 0; i < network.layers.Length; i++)
			{
				Layer layer = network.layers[i];
				newLayers[i] = Layer.SimiliarLayer(layer, weightRange, percentChange);
			}

			return new NeuralNetwork(newLayers);
		}

		public static NeuralNetwork BackpropagationTraining(NeuralNetwork network, float[][] examples, float[][] labels)
		{
			// Initialise List
			Layer[] newLayers = new Layer[network.NumFunctionalLayers];

			// Get predictions 
			float[][] predictions = new float[examples.Length][];
			for (int i = 0; i < examples.Length; i++)
				predictions[i] = network.Predict(examples[i]);

			// Get average differences (ERROR)
			float[] averageDifference = new float[network.OutputSize]; // Difference for each node in the last layer
			for (int i = 0; i < predictions.Length; i++) // Sum all predictions for example i for node j
				for (int j = 0; j < network.OutputSize; j++)
					averageDifference[j] += labels[i][j] - predictions[i][j];
			for (int i = 0; i < averageDifference.Length; i++) // Divide by number of examples
				averageDifference[i] /= examples.Length;

			// Apply Nudges
			for (int i = network.layers.Length - 1; i >= 0; i++)
			{
				newLayers[i].BackwardsPropagate()
			}

			return new NeuralNetwork(newLayers);
		}

		////////////////////////////////// 
	}
}
