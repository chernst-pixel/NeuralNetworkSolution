namespace NeuralNet.Core.Models
{
    public class LayerModel
    {

        public string        Type       { get; set; }
        public double[][]? Weights { get; set; }
        public double[]? Biases { get; set; }

        public int?          InputSize  { get; set; }
        public int?          OutputSize { get; set; }

    }
}
