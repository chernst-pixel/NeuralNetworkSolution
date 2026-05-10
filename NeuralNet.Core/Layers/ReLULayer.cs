using NeuralNet.Core.Models;
using System.Text.Json.Serialization;

namespace NeuralNet.Core.Layers
{
    public class ReLULayer : ILayer
    {
        private object mask;

        public object Backward(object gradOutput)
        {
            if (mask == null)
                throw new InvalidOperationException("Backward called before Forward.");

            switch (gradOutput)
            {
                case double[] g1 when mask is bool[] m1:
                    return Backward1D(g1, m1);
                case double[,,] g3 when mask is bool[,,] m3:
                    return Backward3D(g3, m3);
                default:
                    throw new NotSupportedException("Unsupported input type for ReLULayer."); ;
            }
        }

        private object Backward3D(double[,,] gradOutput, bool[,,] mask)
        {
            double[,,] gradsActivation = new double[gradOutput.GetLength(0), gradOutput.GetLength(1), gradOutput.GetLength(2)];

            for (int channel = 0; channel < gradOutput.GetLength(0); channel++)
            {
                for(int row = 0; row < gradOutput.GetLength(1); row++)
                {
                    for (int col = 0; col < gradOutput.GetLength(2); col++)
                    {
                        if (mask[channel,row,col]) gradsActivation[channel, row, col] = gradOutput[channel, row, col];
                        else gradsActivation[channel, row, col] = 0.0;
                    }
                }
                
            }

            return gradsActivation;
        }

        private object Backward1D(double[] gradOutput, bool[] mask)
        {
            double[] gradsActivation = new double[gradOutput.Length];

            for (int index = 0; index < gradOutput.Length; index++)
            {
                if (mask[index])    gradsActivation[index] = gradOutput[index];
                else                gradsActivation[index] = 0.0;
            }

            return gradsActivation;
        }

        public object Forward(object input)
        {
            switch (input)
            {
                case double[] x1:
                    return Forward1D(x1);
                case double[,,] x3:
                    return Forward3D(x3);
                default:
                    throw new NotSupportedException("Unsupported input type for ReLULayer."); ;
            }
           
        }

        private object Forward1D(double[] preActivations)
        {
            double[]    activation = new double[preActivations.Length];
            bool[]      mask       = new bool[preActivations.Length];

            for (int index = 0; index < preActivations.Length; index++)
            {
                if (preActivations[index] > 0)
                {
                    activation[index]   = preActivations[index];
                    mask[index]         = true;
                }
                else
                {
                    activation[index]   = 0.0;
                    mask[index]         = false;
                }
            }

            this.mask = mask;
            return activation;
        }

        private object Forward3D(double[,,] preActivations)
        {
            double[,,]  activation  = new double[preActivations.GetLength(0),preActivations.GetLength(1),preActivations.GetLength(2)];
            bool[,,]    mask        = new bool[preActivations.GetLength(0), preActivations.GetLength(1), preActivations.GetLength(2)];

            for (int channel = 0; channel < preActivations.GetLength(0); channel++)
            {
                for (int row = 0; row < preActivations.GetLength(1); row++)
                {
                    for (int col = 0; col < preActivations.GetLength(2); col++)
                    {
                        if (preActivations[channel,row,col] > 0)
                        {
                            activation[channel, row, col]   = preActivations[channel, row, col];
                            mask[channel, row, col]         = true;
                        }
                        else
                        {
                            activation[channel, row, col]   = 0.0;
                            mask[channel, row, col]         = false;
                        }
                    }
                }
            }

            this.mask = mask;
            return activation;
        }

        public void Update(double learningRate)
        {
            // ReLU has no trainable parameters
        }

        public LayerModel ToModel()
        {
            return new LayerModel
            {
                Type = "ReLU"
            };
        }
    }
}
