using System.Collections.Generic;
using System;

namespace NeuralNetworks
{
	public struct Layer
	{
		////////////////////////////////// FIELDS

		private float[,] weights;   // [InputNode, OuputNode] - Bias is the last input node

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

		public Layer(float[,] weights)
		{
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

			return result;
		}

		public float[,] GetDesiredNudges(float[] error)
		{
			// nudges[i,j] = weight[i,j]
			float[,] nudges = new float[InputSize + 1, OutputSize];

			// nudge_ij	= ( % weight responsibility ) * error_j
			//			= ( weight_ij / sumWeight_j ) * error_j

			for (int j = 0; j < OutputSize; j++) // Output Node
			{
				float sumWeight = SumWeightsToOutput(j);

				for (int i = 0; i < InputSize + 1; i++) // Input node + Bias node
				{
					// nudge_ij = (weight_ij / sumWeight_j) * error_j
					nudges[i, j] = (weights[i, j] / sumWeight) * error[j];
				}
			}
			return nudges;
		}

		////////////////////////////////// PRIVATE

		private float SumWeightsToInput(int inputNode)
		{
			float sum = 0;
			for (int i = 0; i < OutputSize; i++)
				sum += weights[inputNode, i];

			return sum;
		}

		private float SumWeightsToOutput(int outputNode)
		{
			float sum = 0;
			for (int i = 0; i < InputSize + 1; i++)
				sum += weights[i, outputNode];

			return sum;
		}

		////////////////////////////////// STATIC

		public static Layer RandomLayer(int inputSize, int outputSize, int[] weightRange)
		{
			// Initialise 2D array of floats
			float[,] weights = new float[inputSize + 1, outputSize];

			// Random
			Random rng = new Random(Guid.NewGuid().GetHashCode());

			// Fill 2D array
			for (int i = 0; i < inputSize + 1; i++)
				for (int j = 0; j < outputSize; j++)
					weights[i,j] = rng.Next(weightRange[0] * 20, weightRange[1] * 20) / 20f;

			// Return
			return new Layer(weights);
		}

		/*
		 * 
		 * TODO: Similiar layer should have edge weights changed by an random amount in a range. 
		 * Not randomly chosen edges to change. 
		 * 
		 */
		public static Layer SimiliarLayer(Layer layer, int[] weightRange, float percentChange = 0.3f)
		{
			// Temporary values
			int inputSize = layer.InputSize;
			int outputSize = layer.OutputSize;

			// Initialise 2D array of floats
			float[,] weights = new float[inputSize + 1, outputSize];

			// Random
			Random rng = new Random(Guid.NewGuid().GetHashCode());

			// Fill 2D array
			for (int i = 0; i < inputSize + 1; i++)
				for (int j = 0; j < outputSize; j++)
					if (rng.Next(1, 101) < percentChange * 100) // chance of changing a value to a new random float
						weights[i, j] = rng.Next(weightRange[0] * 20, weightRange[1] * 20) / 20f;

			// Return
			return new Layer(weights);
		}

		/*
		 * 
		 * TODO: Unit Testing
		 * 
		 */
		public static Layer NudgedLayer(Layer layer, float[,] nudges, float learningRateMultiplier = 1f)
		{
			// Initialise 2D array of floats
			float[,] weights = layer.weights;

			// Fill 2D array
			for (int i = 0; i < weights.GetLength(0); i++) // For each Input node
				for (int j = 0; j < weights.GetLength(1); j++) // For each Input node
					weights[i,j] += nudges[i,j] * learningRateMultiplier;

			// Return
			return new Layer(weights);
			
		}
	}
}