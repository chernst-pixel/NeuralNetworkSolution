using System;
using System.Drawing;

namespace NeuralNet.Training
{
    public class Data
    {
        public double[] input;
        public double[] lable;
        public string   path;
        public Data(string path, string value)
        {
            this.input = GetInput(path);
            this.lable = GetValue(int.Parse(value));
            this.path  = path;
        }
        private double[] GetInput(string path)
        {
            Bitmap image = new Bitmap(path);

            int index = 0;
            double[] input = new double[28 * 28];

            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    Color pixel  = image.GetPixel(x, y);
                    input[index] = (double)pixel.R / 255.0;
                    index++;
                }
            }
            return input;
        }

        private double[] GetValue(int value)
        {
            double[] value_array = new double[10];
            for (int i = 0; i < 10; i++)
            {
                if (i == value)
                {
                    value_array[i] = 1;
                }
                else
                {
                    value_array[i] = 0;
                }
            }
            return value_array;
        }

        public static List<Data> LoadData(string path, IProgress<int> progress)
        {
            int count = 0;
            List<Data> liste_data = new List<Data>();

            foreach (string folder in Directory.GetDirectories(path))
            {
                string folder_name = System.IO.Path.GetFileName(folder);
                foreach (string file in Directory.GetFiles(folder))
                {
                    liste_data.Add(new Data(file, folder_name));
                    count++;
                    if (count % 100 == 0) progress.Report(count);
                }
            }

            return liste_data;
        }
    }
}
