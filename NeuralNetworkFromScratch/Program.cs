﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworks;
using System.Diagnostics;

namespace NeuralNetworkFromScratch
{
	class Program
	{
		static void InspectNetwork(NeuralNetwork network)
		{
			Debug.WriteLine("///////////////////////////////////////////////");
			Debug.WriteLine("Input Size: " + network.InputSize.ToString());
			Debug.WriteLine("NumFunctionalLayers: " + network.NumFunctionalLayers);
			Debug.WriteLine("OutputSize: " + network.OutputSize);
			Debug.WriteLine("Shape: [{0}]", string.Join(", ", network.Shape));

			float[] input = new float[network.InputSize];
			Random rng = new Random(Guid.NewGuid().GetHashCode());
			for (int i = 0; i < network.InputSize; i++)
				input[i] = rng.Next(-5, 5);
			float[] output = network.Predict(input);

			Debug.WriteLine("Input: [{0}]", string.Join(", ", input));
			Debug.WriteLine("Output: [{0}]", string.Join(", ", output));
			Debug.WriteLine("///////////////////////////////////////////////");
		}

		static void TestNetwork()
		{
			Debug.WriteLine("///////////////////////////////////////////////");

			NeuralNetwork network = NeuralNetwork.RandomNetwork(new int[] { 1, 1 }, new int[] { -5, 5 });
			float[][] examples = new float[4][]
			{
				new float[2]{ 3,20 },
				new float[2]{ 4,25 },
				new float[2]{ 5,30 },
				new float[2]{ 6,35 }
			};
			//{ // I cannot account for parabolic data
			//	new float[2]{ 1,1 },
			//	new float[2]{ 2,4 },
			//	new float[2]{ 3,9 },
			//	new float[2]{ 4,16 }
			//};

			bool done = false;
			double bestLoss = Double.PositiveInfinity;
			NeuralNetwork bestNetwork = network;
			int iteration = 0;

			while (!done)
			{
				network = NeuralNetwork.SimiliarNetwork(network, new int[] { -10, 10 });
				//InspectNetwork(network);
				double loss = 0d;
				
				// For each example
				for (int i = 0; i < examples.Length; i++)
				{
					float[] feature = new float[1] { examples[i][0] };
					float[] label = new float[1] { examples[i][1] };
					float[] prediction = network.Predict(feature);

					// For each output value
					for (int j = 0; j < label.Length; j++)
					{
						double predictionValue = prediction[j];
						double labelValue = label[j];
						loss += Math.Pow(labelValue - predictionValue, 2d);
					}
				}

				if (loss < 1f)
					done = true;

				if (loss > bestLoss)
					network = bestNetwork;
				

				if (loss < bestLoss)
				{
					bestLoss = loss;
					Console.WriteLine("(" + iteration.ToString() + ") Best Loss: " + bestLoss.ToString());
				}

				iteration += 1;

				Debug.WriteLine("L2 Loss: " + loss.ToString());
				Debug.WriteLine("================= Best Loss: " + bestLoss.ToString());
			}
			Debug.WriteLine("///////////////////////////////////////////////");
		}

		static void TestBackprop()
		{
			Debug.WriteLine("///////////////////////////////////////////////");

			NeuralNetwork network = NeuralNetwork.RandomNetwork(new int[] { 1, 1 }, new int[] { -5, 5 });
			float[][] examples = new float[4][]
			{
				new float[1]{ 3 },
				new float[1]{ 4 },
				new float[1]{ 5 },
				new float[1]{ 6 }
			};
			float[][] targets = new float[4][]
			{
				new float[1]{ 20 },
				new float[1]{ 25 },
				new float[1]{ 30 },
				new float[1]{ 35 }
			};

			bool done = false;
			double bestLoss = float.PositiveInfinity;
			int iteration = 0;

			while (!done)
			{
				// 1) Calculate Loss
				double loss = 0;

				// For each example
				for (int i = 0; i < examples.Length; i++)
				{
					float[] features = new float[1] { examples[i][0] };
					float[] labels = new float[1] { targets[i][0] };
					
					float[] prediction = network.Predict( features );
					loss += Math.Pow(labels[0] - prediction[0], 2d);
				}

				if (loss < 1f)
					done = true;

				if (loss < bestLoss)
				{
					bestLoss = loss;
					Console.WriteLine("(" + iteration.ToString() + ") Best Loss: " + bestLoss.ToString());
				}
				Console.WriteLine("Loss: " + loss);
				iteration += 1;
				network = NeuralNetwork.BackpropagationTraining(network, examples, targets, 0.001f);

				Debug.WriteLine("L2 Loss: " + loss.ToString());
				Debug.WriteLine("================= Best Loss: " + bestLoss.ToString());
			}



			Debug.WriteLine("///////////////////////////////////////////////");
		}

		static void Main(string[] args)
		{
			NeuralNetwork network = NeuralNetwork.RandomNetwork(new int[] { 2, 3, 2 }, new int[] { 0, 1 });
			InspectNetwork(network);


			//TestNetwork();
			TestBackprop();


			Debug.WriteLine("Done.");
			Console.WriteLine("Done.");
			Console.ReadLine();
		}
	}
}
