using NeuralNet.Core.Layers;
using NeuralNet.Core.Losses;
using NeuralNet.Core.Models;
using System.Text.Json.Serialization;

namespace NeuralNet.Core
{

    public class NeuralNetwork
    {
        double  learningRate = 0.01;
        public List<ILayer> layers  { get; set; } 
        public ILoss        loss    { get; set; }
        public NeuralNetwork()
        {
            layers = new();
            layers.Add(new DenseLayer(28*28, 128));
            layers.Add(new ReLULayer());
            layers.Add(new DenseLayer(128, 10));
            
            this.loss = new SoftmaxCrossEntropyLoss();
        }

        [JsonConstructor]
        public NeuralNetwork(List<ILayer> layers, ILoss loss)
        {
            this.layers         = layers;
            this.loss           = loss;
        }

        public double[] Forward(object input)
        {
            object activation = input;

            foreach (ILayer layer in layers)
            {
                activation  = layer.Forward(activation);
            }

            return (double[])activation;
        }

        public double TrainSample(double[] input, double[] target)
        {
            double[]    preActivation   = Forward(input);
            double      loss            = this.loss.Forward(preActivation, target);

            BackwardAndUpdate();
            return loss;
        }

        public void BackwardAndUpdate()
        {
            object grads_activation = loss.Backward();
            for (int layer = layers.Count() - 1; layer > -1; layer--)
            {
                grads_activation = layers[layer].Backward(grads_activation);
            }
            foreach (ILayer layer in layers)
            {
                layer.Update(this.learningRate);
            }
        }

        public double[] Predict(double[] input)
        {
            double[] preActivation = Forward(input);
            return Activations.Softmax(preActivation);
        }
        public NeuralNetworkModel ExportModel()
        {
            NeuralNetworkModel model = new NeuralNetworkModel();

            foreach (ILayer layer in layers)
            {
                model.Layers.Add(layer.ToModel());
            }

            return model;
        }


        public void ClearLayers() => layers.Clear();
        public void AddLayer(ILayer layer) => layers.Add(layer);


    }
}
