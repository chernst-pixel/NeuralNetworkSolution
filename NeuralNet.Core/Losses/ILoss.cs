namespace NeuralNet.Core.Losses
{
    public interface ILoss
    {
        double[] results { get; }

        double Forward(double[] logits, double[] yTrue);
        double[] Backward();
    }
}
