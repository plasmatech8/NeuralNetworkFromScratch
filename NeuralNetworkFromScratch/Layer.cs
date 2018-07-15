using System.Collections.Generic;
using System;

namespace NeuralNetworks
{
	public struct Layer
	{
		////////////////////////////////// FIELDS

		private readonly bool doRelu;
		public float[,] weights;   // [InputNode, OuputNode] - Bias is the last input node

		////////////////////////////////// PROPERTIES

		public int InputSize
		{
			get
			{
				return weights.GetLength(0) - 1;
			}
		}

		public int OutputSize
		{
			get
			{
				return weights.GetLength(1);
			}
		}

		////////////////////////////////// CONSTRUCTORS

		public Layer(float[,] weights, bool doRelu = true)
		{
			this.doRelu = doRelu;
			this.weights = weights;
		}
		
		////////////////////////////////// PUBLIC

		public float[] ForwardsPropagate(float[] input)
		{
			// Incorrect input size
			if (input.Length != InputSize)
				throw new ArgumentException("Incorrect input size for layer");

			float[] result = new float[OutputSize];

			// node = 1 * bias + input * weight + ...
			for (int j = 0; j < OutputSize; j++)
			{
				// Node * Weights
				for (int i = 0; i < InputSize; i++)
					result[j] += weights[i, j] * input[i];

				// 1 * Bias (weight)
				result[j] += weights[InputSize, j]; // (Highest index)
			}

			// Relu activation function
			if (doRelu)
				for (int j = 0; j < OutputSize; j++)
					result[j] = Math.Max(0, result[j]);

			return result;
		}

		public float[,] GetDesiredNudges(float[] error)
		{
			/*
			 * Desired Nudges/Edge deltas describes the desired movement of edge weights based on its responsibility for an error on the output.
			 * nudge[i, j] = (weight[i, j]/totalWeight[j]) * error[j]
			 */

			// nudges[i,j] = weight[i,j]
			float[,] nudges = new float[InputSize + 1, OutputSize];

			for (int j = 0; j < OutputSize; j++) // Output Node
			{
				float sumWeight = SumWeightsToOutput(j);

				for (int i = 0; i < InputSize + 1; i++) // Input node + Bias node
				{
					// nudge[i, j] = (weight[i, j]/totalWeight[j]) * error[j]
					nudges[i, j] = (weights[i, j] / sumWeight) * error[j];
				}
			}
			return nudges;
		}

		public float[] GetInputNodeErrors(float[,] edgeDeltas, bool inclBias = true)
		{
			/*
			 * Aggregates edge deltas into an error for each input node
			 * NodeError[node1] = Sum(edgeDelta[node1,node2])
			 */

			float[] nodeErrors;
			if (inclBias)
				nodeErrors = new float[InputSize];
			else
				nodeErrors = new float[InputSize + 1];

			for (int k = 0; k < nodeErrors.Length; k++) // input
				for (int j = 0; j < OutputSize; j++) // output
				{
					// NodeError[node1] = Sum(edgeDelta[node1,node2])
					nodeErrors[k] += edgeDeltas[k, j];
				}
			return nodeErrors;
		}

		////////////////////////////////// PRIVATE

		private float SumWeightsToInput(int inputNode)
		{
			float sum = 0;
			for (int i = 0; i < OutputSize; i++)
				sum += weights[inputNode, i];

			return sum;
		}

		private float SumWeightsToOutput(int outputNode, bool inclBias = true)
		{
			float sum = 0;
			for (int i = 0; i < InputSize + (inclBias ? 1 : 0); i++)
				sum += weights[i, outputNode];

			return sum;
		}

		////////////////////////////////// STATIC

		public static Layer RandomLayer(int inputSize, int outputSize, int[] weightRange, bool doRelu = true)
		{
			// Initialise 2D array of floats
			float[,] weights = new float[inputSize + 1, outputSize];

			// Random
			Random rng = new Random(Guid.NewGuid().GetHashCode());

			// Fill 2D array
			for (int i = 0; i < inputSize + 1; i++)
				for (int j = 0; j < outputSize; j++)
					weights[i,j] = rng.Next(weightRange[0] * 50, weightRange[1] * 50) / 50f;

			// Return
			return new Layer(weights, doRelu);
		}

		public static Layer SimilarLayer(Layer layer, float maxChangeAmount = 0.2f)
		{
			// Init new layer
			float[,] newWeights = (float[,])layer.weights.Clone();

			// Init Random
			Random rng = new Random(Guid.NewGuid().GetHashCode());

			// Fill 2D array
			for (int i = 0; i < newWeights.GetLength(0); i++)
				for (int j = 0; j < newWeights.GetLength(1); j++)
				{
					Console.WriteLine(">" + (newWeights[i, j]).ToString());
					newWeights[i, j] += (float)(rng.NextDouble() * maxChangeAmount * 2 - maxChangeAmount);
					Console.WriteLine("?" + (newWeights[i, j]).ToString());
					//Console.WriteLine("?" + (rng.NextDouble() * maxChangeAmount * 2 - maxChangeAmount).ToString());
				}
				
			return new Layer(newWeights, layer.doRelu);
		}

		public static Layer NudgedLayer(Layer layer, float[,] nudges, float learningRateMultiplier = 1f)
		{
			// Initialise 2D array of floats
			float[,] weights = layer.weights;

			// Fill 2D array
			for (int i = 0; i < weights.GetLength(0); i++) // Input node
				for (int j = 0; j < weights.GetLength(1); j++) // Output node
					weights[i,j] += nudges[i,j] * learningRateMultiplier;

			// Return
			return new Layer(weights, layer.doRelu);
			
		}
	}
}