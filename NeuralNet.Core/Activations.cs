using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Core
{
    static class Activations
    {
        public static double[] Softmax(double[] logits)
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
    }
}
