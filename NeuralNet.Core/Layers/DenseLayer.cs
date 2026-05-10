using NeuralNet.Core.Models;
using NeuralNet.Core.Serialization;

namespace NeuralNet.Core.Layers
{ 
    public class DenseLayer : ILayer
    {
        public  Neuron[]    neurons { get; set; }
        public  int         inputSize   { get; set; }
        public  int         outputSize  { get; set; }
        private double[]    input;          
        private double[]    preActivation;
        private double[]    gradsBias;
        private double[,]   gradsWeights;

        public DenseLayer(int inputSize, int outputSize)
        {
            this.inputSize   = inputSize;
            this.outputSize  = outputSize;
            neurons = new Neuron[outputSize];

            for (int neuron = 0; neuron < outputSize; neuron++)
            {
                neurons[neuron] = new Neuron(inputSize);
            }
        }
        public object Forward(object input)
        {
            this.input = (double[])input;
            preActivation = new double[neurons.Length];

            for (int neuron = 0; neuron < neurons.Length; neuron++)
            {
                preActivation[neuron] = neurons[neuron].Forward(this.input);
            }

            return preActivation;
        }

        public object Backward(object gradOutput)
        {
            double [] gradsPreActivation = (double[])gradOutput;
            gradsWeights = new double[outputSize, inputSize];
            gradsBias = new double[outputSize];

            for (int neuron = 0; neuron < outputSize; neuron++)
            {
                for (int weight = 0; weight < inputSize; weight++)
                {
                    gradsWeights[neuron, weight] = gradsPreActivation[neuron] * input[weight];
                }
                gradsBias[neuron] = gradsPreActivation[neuron];
            }

            return CalculateHiddenGrads(gradsPreActivation);
        }

        private double[] CalculateHiddenGrads(double[] gradsPreActivation)
        {
            double[] hiddenGrads = new double[inputSize];

            for (int inputNeuron = 0; inputNeuron < inputSize; inputNeuron++)
            {
                hiddenGrads[inputNeuron] = 0;

                for (int neuron = 0; neuron < outputSize; neuron++)
                {
                    hiddenGrads[inputNeuron] += gradsPreActivation[neuron] * neurons[neuron].weights[inputNeuron];
                }
            }

            return hiddenGrads;
        }

        public void Update(double learningRate)
        {
            for (int neuron = 0; neuron < neurons.Length; neuron++)
            {
                for (int weight = 0; weight < neurons[neuron].weights.Length; weight++)
                {
                    neurons[neuron].weights[weight] -= learningRate * gradsWeights[neuron, weight];
                }

                neurons[neuron].bias -= learningRate * gradsBias[neuron];
            }
        }

        public LayerModel ToModel()
        {
            return new LayerModel
            {
                Type        = "Dense",
                Weights     = TensorSerialization.ToJagged ( CloneWeights() ),
                Biases      = CloneBias(),
                InputSize   = inputSize,
                OutputSize  = outputSize
            };
        }

        private double[,] CloneWeights()
        {
            double[,] weights = new double[outputSize, inputSize];
            for (int neuron = 0; neuron < outputSize; neuron++)
            {
                for (int weight = 0; weight < inputSize; weight++)
                {
                    weights[neuron, weight] = neurons[neuron].weights[weight];
                }
            }
            
            return weights;   
        }
        private double[] CloneBias()
        {
            double[] biases = new double[outputSize];
            for (int neuron = 0; neuron < outputSize; neuron++)
            {
                biases[neuron] = neurons[neuron].bias;
            }

            return biases;
        }


        public static DenseLayer FromModel(LayerModel model)
        {
            DenseLayer layer = new DenseLayer(
                model.InputSize!.Value,
                model.OutputSize!.Value);

            layer.SetWeightsAndBiases(
                TensorSerialization.ToRect((double[][])model.Weights!.Clone() ),
                (double[])model.Biases!.Clone());

            return layer;
        }

        private void SetWeightsAndBiases(double[,] weights, double[] biases)
        {
            for (int neuron = 0; neuron < outputSize; neuron++)
            {
                for (int weight = 0; weight < inputSize; weight++)
                {
                    neurons[neuron].weights[weight] = weights[neuron,weight];
                }
                neurons[neuron].bias = biases[neuron];
            }
        }
    }
}
