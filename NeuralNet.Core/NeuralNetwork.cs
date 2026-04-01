using System.Text.Json.Serialization;

namespace NeuralNet.Core
{

    public class NeuralNetwork
    {
        const int INPUTINDEX  = 0;
        const int OUTPUTINDEX = 1;
        double[]  forward_input, outputs;
        public Layer[] layers { get; set; }
        public NeuralNetwork(int[][] layer_structure)
        {
            layers = new Layer[layer_structure.Length];
            for (int layer = 0; layer < layer_structure.Length; layer++)
            {
                layers[layer] = new Layer(layer_structure[layer][INPUTINDEX], layer_structure[layer][OUTPUTINDEX]);
            }
        }

        [JsonConstructor]
        public NeuralNetwork(Layer[] layers)
        {
            this.layers = layers;
        }

        public double[] Activation(double[] forward_input)
        {
            this.forward_input = forward_input;
            for (int layer = 0; layer < layers.Length; layer++)
            {
                this.outputs = this.layers[layer].Forward(this.forward_input, IsOutputLayer(layer));
                this.forward_input       = this.outputs;
            }
            return this.outputs;
        }

        public double[] TrainSample(double[] input, double[] target)
        {
            this.outputs = Activation(input);
            Backpropagation(this.outputs, target);
            return this.outputs;
        }

        public void Backpropagation(double[] activations, double[] target)
        {
            double[] grads_activation = new double[target.Length];
            for (int neuron = 0; neuron < layers[layers.Length - 1].neurons.Length; neuron++)
            {
                grads_activation[neuron] = activations[neuron] - target[neuron];
            }
            for (int layer = layers.Length - 1; layer > -1; layer--)
            {
                grads_activation = layers[layer].Backpropagation(grads_activation, IsOutputLayer(layer));
            }
            foreach (Layer layer in layers)
            {
                layer.UpdateWeights();
            }
        }

        private bool IsOutputLayer(int layer_number)
        {
            if (layer_number == layers.Length - 1) return true;
            else return false;
        }

        public double CrossEntropyLoss(double[] outputs, double[] target)
        {
                  double loss = 0.0;
            const double eps  = 1e-12; 

            for (int i = 0; i < outputs.Length; i++)
            {
                if (target[i] > 0)
                {
                    loss -= Math.Log(outputs[i] + eps);
                }
            }

            return loss;
        }


    }
}
