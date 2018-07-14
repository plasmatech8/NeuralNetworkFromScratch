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

		/*
		 * 
		 * TODO: Testing.
		 * TODO: Verify knowledge of backpropagation.
		 * 
		 */
		public static NeuralNetwork BackpropagationTraining(NeuralNetwork network, float[][] examples, float[][] target, float learningRateMultiplier = 1f)
		{
			int numExamples = examples.Length;
			int numOutputs = network.OutputSize;
			int numLayers = network.NumFunctionalLayers;

			float[][] nodeErrors = new float[numLayers][]; // float[layer][node]
			float[][,] edgeDeltas = new float[numLayers][,]; // float[layer][inputNode, outputNode (layer)] Note: that edges refer to layer from layer-1 to layer

			// OUTPUT NODE ERROR:
			// Calculate Average Error for each output node
			// NodeError[outputLayer][node] = Average(target[example][node] - prediction[node]) over all examples
			float[] outputError = new float[network.OutputSize];
			for (int i = 0; i < numExamples; i++) // Example
			{
				float[] prediction = network.Predict(examples[i]);
				for (int j = 0; j < numOutputs; j++) // Output Node
					outputError[j] += target[i][j] - prediction[j]; // Prediction is bigger = negative error
			}
			for (int j = 0; j < numOutputs; j++) // Output Node
				outputError[j] /= numExamples;
			nodeErrors[numLayers - 1] = outputError;
			
			// Get delta for each edge
			edgeDeltas[numLayers - 1] = network.layers[numLayers - 1].GetDesiredNudges(outputError);

			// Backpropagation
			for (int i = numLayers - 2; i >= 0; i--) // From layer (L - 1) to 1. -1 = input nodes
			{
				Layer hiddenLayer = network.layers[i];
				Layer aboveLayer = network.layers[i + 1];

				// HIDDEN NODE ERROR:
				// Calculate node errors based on above-layer edge deltas
				// NodeError[layer][node1] = Sum(edgeDelta[layer above][node1, node2])
				nodeErrors[i] = aboveLayer.GetInputNodeErrors(edgeDeltas[i+1]);

				// EDGE DELTA:
				// Get delta for each edge
				// nudge[i, j] = (weight[i, j]/totalWeight[j]) * error[j]
				edgeDeltas[i] = hiddenLayer.GetDesiredNudges(nodeErrors[i]);
			}
			Console.WriteLine("Output error tot: " + string.Join(" ", outputError));

			Layer[] newLayers = new Layer[numLayers];
			for (int i = 0; i < numLayers; i++)
				newLayers[i] = Layer.NudgedLayer(network.layers[i], edgeDeltas[i], learningRateMultiplier);
			
			return new NeuralNetwork(newLayers);
		}

		////////////////////////////////// 
	}
}
