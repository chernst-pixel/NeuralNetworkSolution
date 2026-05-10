namespace NeuralNet.Core.Serialization
{
    public static class TensorSerialization
    {
        public static double[][] ToJagged(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            var jagged = new double[rows][];

            for (int row = 0; row < rows; row++)
            {
                jagged[row] = new double[cols];
                for (int col = 0; col < cols; col++)
                    jagged[row][col] = matrix[row, col];
            }

            return jagged;
        }

        public static double[,] ToRect(double[][] jagged)
        {
            int rows = jagged.Length;
            int cols = jagged[0].Length;

            var rect = new double[rows, cols];

            for (int row = 0; row < rows; row++)
                for (int col = 0; col < cols; col++)
                    rect[row, col] = jagged[row][col];

            return rect;
        }

    }
}
