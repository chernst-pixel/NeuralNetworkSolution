namespace NeuralNet.Core.Losses
{
    public class SoftmaxCrossEntropyLoss : ILoss
    {
        public  double[] results { get; private set; }
        private double[] yTrue;

        public double[] Backward()
        {
            if (results == null || yTrue == null)
                throw new InvalidOperationException("Backward called before Forward.");


            double[] gradsLogits = new double[results.Length];
            for (int grad = 0; grad < results.Length; grad++)
            {
                gradsLogits[grad] = results[grad] - yTrue[grad];
            }
            return gradsLogits;
        }

        public double Forward(double[] logits, double[] yTrue)
        {
            this.yTrue      = (double[])yTrue.Clone();
            results    = Softmax(logits);
            return CrossEntropyLoss(results, yTrue);
        }


        private double[] Softmax(double[] logits)
        {
            double max = logits.Max();
            double sum = 0.0;
            double[] exponential = new double[logits.Length];
            double[] activation = new double[logits.Length];

            for (int i = 0; i < logits.Length; i++)
            {
                exponential[i] = Math.Exp(logits[i] - max);
                sum += exponential[i];
            }

            for (int i = 0; i < logits.Length; i++)
            {
                activation[i] = exponential[i] / sum;
            }

            return activation;
        }

        private double CrossEntropyLoss(double[] outputs, double[] target)
        {
            double loss = 0.0;
            const double eps = 1e-12;

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
