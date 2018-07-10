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

		public static NeuralNetwork BackpropagationTraining(NeuralNetwork network, float[][] examples, float[][] target)
		{
			// Initialise variables
			Layer[] newLayers = new Layer[network.NumFunctionalLayers];
			float[][] predictions = new float[examples.Length][];

			float[,] tempNudges = new float[network.Shape[network.NumFunctionalLayers - 1], network.OutputSize];
			float[] tempError = new float[network.OutputSize];

			// Get predictions 
			for (int i = 0; i < examples.Length; i++)
				predictions[i] = network.Predict(examples[i]);

			// Get error for output layer (average difference)
			for (int outputNode = 0; outputNode < network.OutputSize; outputNode++) 
				for (int example = 0; example < predictions.Length; example++)
					// error[j] = (target[i,j] - prediction[i,j]) + ... for example i from 0 to n
					tempError[outputNode] += target[example][outputNode] - predictions[example][outputNode];

			for (int i = 0; i < tempError.Length; i++)
				// error[j] = Sum of error of all examples / number of examples
				tempError[i] /= examples.Length;

			// Get Nudges and new nudged Layer
			/*
			 tempNudges = layer.GetDesiredNudges()
			newLayers[newLayers.Length - 1] = Layer.NudgedLayer(network.layers[network.layers.Length - 1], tempError);
			*/

			// For each preceding layer
			for (int x = 0; x < network.layers.Length - 1; x++)
			{
				Layer layer = network.layers[x];
				// Backpropagation
				/*
				for (int i = 0; i < layer.InputSize; i++)
				{
					tempError[i] = 0;

					for (int j = 0; j < layer.OutputSize; j++) // output node j
					{
						tempError[i] += tempNudges[i,j];
					}
				}
				*/
				//newLayers[x] = Layer.NudgedLayer(layer, tempNudges);
			}

			return new NeuralNetwork(newLayers);
		}

		////////////////////////////////// 
	}
}
