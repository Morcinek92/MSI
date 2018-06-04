﻿using System;
using FANNCSharp;
#if FANN_FIXED
using FANNCSharp.Fixed;
using DataType = System.Int32;
#elif FANN_DOUBLE
using FANNCSharp.Double;
using DataType = System.Double;
#else
using FANNCSharp.Float;
using DataType = System.Single;
#endif
namespace XorTrain
{
    class XorTrain
    {

        static int PrintCallback(NeuralNet net, TrainingData train, uint max_epochs, uint epochs_between_reports, float desired_error, uint epochs, Object user_data)
        {
            Console.WriteLine(String.Format("Epochs     " + String.Format("{0:D}", epochs).PadLeft(8) + ". Current Error: " +
                              String.Format("{0:F}", net.MSE).PadRight(8)));
            return 0;
        }

        static void XorTest()
        {
            Console.WriteLine("\nXOR test started.");

            const float learning_rate = 0.7f;
            const uint num_layers = 3;
            const uint num_input = 900;
            const uint num_hidden = 10;
            const uint num_output = 5;
            const float desired_error = 0.001f;
            const uint max_iterations = 300000;
            const uint iterations_between_reports = 1000;

            Console.WriteLine("\nCreating network.");

            using (NeuralNet net = new NeuralNet(NetworkType.LAYER, num_layers, num_input, num_hidden, num_output))
            {
                net.LearningRate = learning_rate;

                net.ActivationSteepnessHidden = 1.0F;
                net.ActivationSteepnessOutput = 1.0F;

                net.ActivationFunctionHidden = ActivationFunction.SIGMOID;
                net.ActivationFunctionOutput = ActivationFunction.SIGMOID;

                // Output network type and parameters
                Console.Write("\nNetworkType                         :  ");
                switch (net.NetworkType)
                {
                    case NetworkType.LAYER:
                        Console.WriteLine("LAYER");
                        break;
                    case NetworkType.SHORTCUT:
                        Console.WriteLine("SHORTCUT");
                        break;
                    default:
                        Console.WriteLine("UNKNOWN");
                        break;
                }
                net.PrintParameters();

                Console.WriteLine("\nTraining network.");

                using (TrainingData data = new TrainingData())
                {
                    if (data.ReadTrainFromFile("xor.data"))
                    {
                        // Initialize and train the network with the data
                        net.InitWeights(data);

                        Console.WriteLine("Max Epochs " + String.Format("{0:D}", max_iterations).PadLeft(8) + ". Desired Error: " + String.Format("{0:F}", desired_error).PadRight(8));
                        net.SetCallback(PrintCallback, null);
                        net.TrainOnData(data, max_iterations, iterations_between_reports, desired_error);

                        Console.WriteLine("\nTesting network.");

                        for (uint i = 0; i < data.TrainDataLength; i++)
                        {
                            // Run the network on the test data
                            DataType[] calc_out = net.Run(data.Input[i]);


                            Console.WriteLine($"Symulacja: {calc_out[0]} {calc_out[1]} {calc_out[2]} {calc_out[3]} {calc_out[4]}");
                            Console.WriteLine($"Dane uczace: {data.OutputAccessor[(int)i][0].ToString()} {data.OutputAccessor[(int)i][1].ToString()} {data.OutputAccessor[(int)i][2].ToString()} " +
                                $"{data.OutputAccessor[(int)i][3].ToString()} {data.OutputAccessor[(int)i][4].ToString()}");

                            Console.WriteLine("XOR test ({0}, {1}) -> {2}, should be {3}, difference = {4}",
                                data.InputAccessor[(int)i][0].ToString("+#;-#"),
                                data.InputAccessor[(int)i][1].ToString("+#;-#"),
                                calc_out[0] == 0 ? 0.ToString() : calc_out[0].ToString("+#.#####;-#.#####"),
                                data.OutputAccessor[(int)i][0].ToString("+#;-#"),
                                FannAbs(calc_out[0] - data.Output[i][0]));
                        }

                        Console.WriteLine("\nSaving network.");

                        // Save the network in floating point and fixed point
                        net.Save("xor_float.net");
                        uint decimal_point = (uint)net.SaveToFixed("xor_fixed.net");
                        data.SaveTrainToFixed("xor_fixed.data", decimal_point);

                        Console.WriteLine("\nXOR test completed.");
                    }
                }
            }
        }
        static int Main(string[] args)
        {
            try
            {
                XorTest();
            }
            catch
            {
                Console.Error.WriteLine("\nAbnormal exception.");
            }
            Console.ReadKey();
            return 0;
        }

        static DataType FannAbs(DataType value)
        {
            return (((value) > 0) ? (value) : -(value));
        }
    }
}