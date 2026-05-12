using NeuralNet.Core.Models;

namespace NeuralNet.Core.Layers
{
    public class FlattenLayer : ILayer
    {
        private int  channels, rows, cols;
        public object Backward(object gradOutput)
        {
            if (this.channels == 0)
                throw new InvalidOperationException("Backward called before Forward.");

            if (gradOutput is not double[] gradOutput1D)
                throw new NotSupportedException("Flatten backward expects 1D gradient.");

            if (gradOutput1D.Length != channels * rows * cols)
                throw new InvalidOperationException("Gradient size does not match flatten shape.");

            double[,,]  gradsActivation = new double[channels, rows, cols];

            int index = 0;

            for (int channel = 0; channel < this.channels; channel++)
            {
                for (int row = 0; row < this.rows; row++)
                {
                    for (int col = 0; col < this.cols; col++)
                    {
                        gradsActivation[channel, row, col] = gradOutput1D[index];
                        index++;
                    }
                }
            }
            
            return gradsActivation;
        }

        public object Forward(object input)
        {

            if (input is not double[,,] input3D)
                throw new NotSupportedException("FlattenLayer expects a 3D input.");

            this.channels   = input3D.GetLength(0);
            this.rows       = input3D.GetLength(1);
            this.cols       = input3D.GetLength(2);

            double[] output = new double[channels * rows * cols];
            int      index  = 0;

            for (int channel = 0; channel < this.channels; channel++)
            {
                for (int row = 0; row < this.rows; row++)
                {
                    for (int col = 0; col < this.cols; col++)
                    {
                        output[index] = input3D[channel, row, col];
                        index++;
                    }
                }
            }

            return output;
        }

        public LayerModel ToModel()
        {
            return new LayerModel
            {
                Type = "Flatten"
            };
        }

        public void Update(double learningRate)
        {
            // Flatten has no trainable parameters
        }
    }
}
