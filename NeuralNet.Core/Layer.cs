using System.Text.Json.Serialization;

namespace NeuralNet.Core
{ 
    public class Layer
    {

        public  Neuron[]    neurons { get; set; }
        public  int         input   { get; set; }
        public  int         output  { get; set; }
        private double      learningRate = 0.02;
        public  double[]    forward_input, forward, gradsB;
        private double[,]   gradsW;

        public Layer(int input, int output)
        {
            this.input   = input;
            this.output  = output;
            this.neurons = new Neuron[output];

            for (int neuron = 0; neuron < output; neuron++)
            {
                this.neurons[neuron] = new Neuron(input);
            }
        }

        [JsonConstructor]
        public Layer(Neuron[] neurons)
        {
            this.neurons = neurons;
            this.output  = neurons.Length;
            this.input   = neurons[0].weights.Length;
        }

        public double[] Forward(double[] input, bool is_output_layer)
        {
            forward_input = input;
            forward = new double[neurons.Length];

            for (int neuron = 0; neuron < neurons.Length; neuron++)
            {
                forward[neuron] = neurons[neuron].Forward(this.forward_input);
            }

            if (is_output_layer) return Softmax(forward);
            else                 return ReLU(forward);
        }

        public double[] Backpropagation(double[] grads_activation, bool is_output_layer)
        {
            double[] grads_pre_activation = new double[grads_activation.Length];

            if (is_output_layer)
            {
                grads_pre_activation = grads_activation;
            }
            else
            {
                for (int i = 0; i < grads_activation.Length; i++)
                {
                    if (forward[i] > 0)
                    {
                        grads_pre_activation[i] = grads_activation[i];
                    }
                    else
                    {
                        grads_pre_activation[i] = 0;
                    }
                }
            }

            gradsW = new double[this.output, this.input];
            gradsB = new double[this.output];

            for (int neuron = 0; neuron < this.output; neuron++)
            {
                for (int weight = 0; weight < this.input; weight++)
                {
                    gradsW[neuron, weight] = grads_pre_activation[neuron] * this.forward_input[weight];
                }
                gradsB[neuron] = grads_pre_activation[neuron];
            }

            return CalculateHiddenGrads(grads_pre_activation);
        }

        private double[] CalculateHiddenGrads(double[] grads_pre_activation)
        {
            double[] hidden_grads = new double[this.input];

            for (int inputNeuron = 0; inputNeuron < this.input; inputNeuron++)
            {
                hidden_grads[inputNeuron] = 0;

                for (int neuron = 0; neuron < this.output; neuron++)
                {
                    hidden_grads[inputNeuron] += grads_pre_activation[neuron] * neurons[neuron].weights[inputNeuron];
                }
            }

            return hidden_grads;
        }

        public void UpdateWeights()
        {
            for (int neuron = 0; neuron < neurons.Length; neuron++)
            {
                for (int weight = 0; weight < neurons[neuron].weights.Length; weight++)
                {
                    neurons[neuron].weights[weight] -= learningRate * gradsW[neuron, weight];
                }

                neurons[neuron].bias -= learningRate * gradsB[neuron];
            }
        }

        private double[] ReLU(double[] forward)
        {
            double[] activation = new double[forward.Length];

            for (int i = 0; i < forward.Length; i++)
            {
                activation[i] = Math.Max(0.0, forward[i]);
            }

            return activation;
        }

        private double[] Softmax(double[] forward)
        {
            double   max         = forward.Max();
            double   sum         = 0.0;
            double[] exponential = new double[forward.Length];
            double[] activation  = new double[forward.Length];

            for (int i = 0; i < forward.Length; i++)
            {
                exponential[i] = Math.Exp(forward[i] - max);
                sum += exponential[i];
            }

            for (int i = 0; i < forward.Length; i++)
            {
                activation[i] = exponential[i] / sum;
            }

            return activation;
        }

    }
}
