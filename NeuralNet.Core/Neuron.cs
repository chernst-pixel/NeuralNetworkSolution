using System.Text.Json.Serialization;

namespace NeuralNet.Core
{
    public class Neuron
    {
        public double[] weights { get; set; }
        public double  bias { get; set; }
        private double pre_activation;
        private static readonly Random rand = new Random();

        public Neuron(int input)
        {
            this.weights = new double[input];
            double std   = Math.Sqrt(2.0 / input);

            for (int weight = 0; weight < weights.Length; weight++)
            {
                this.weights[weight] = NextGaussian() * std;
            }

            this.bias = 0.0;
        }

        [JsonConstructor]
        public Neuron(double[] weights, double bias)
        {
            this.weights = weights;
            this.bias    = bias;
        }

        public double Forward(double[] inputs)
        {
            this.pre_activation = 0;

            for (int input = 0; input < inputs.Length; input++)
            {
                this.pre_activation += inputs[input] * this.weights[input];
            }

            this.pre_activation += this.bias;
            return this.pre_activation;
        }


        private static double NextGaussian()
        {
            double u1 = 1.0 - rand.NextDouble();
            double u2 = 1.0 - rand.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) *
                   Math.Cos(2.0 * Math.PI * u2);
        }


    }
}
