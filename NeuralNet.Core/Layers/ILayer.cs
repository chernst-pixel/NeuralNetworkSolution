using NeuralNet.Core.Models;

namespace NeuralNet.Core.Layers
{
    public interface ILayer
    {
        object  Forward     (object input);
        object  Backward    (object gradOutput);
        void    Update      (double learningRate);
        LayerModel ToModel();
    }
}
