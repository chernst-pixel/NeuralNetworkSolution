namespace NeuralNet.Core.Losses
{
    public interface ILoss
    {
        double Forward(double[] logits, double[] yTrue);
        double[] Backward();
    }
}
