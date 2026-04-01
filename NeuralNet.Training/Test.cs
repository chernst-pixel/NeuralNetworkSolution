using NeuralNet.Core;
using NeuralNet.Persistence;
using System;

namespace NeuralNet.Training
{
    public class Test
    {
        private List<Data>      test_data;
        private double[]        prediction;
        private string          default_path = @"MNIST\test";

        public  int             correct_prediction, false_prediction;
        public  double          average_certainty;
        public void RunTest(IProgress<int[]> progress, NeuralNetwork network)
        {
            int     count       = 0;
            int     list_length = test_data.Count;
            int     answer; 
            double  certainty; 

            this.correct_prediction = 0;
            this.false_prediction   = 0;
            this.average_certainty  = 0;

            foreach (Data data in test_data)
            {
                prediction  = network.Activation(data.input);
                answer      = GetAnswer(prediction);
                certainty   = prediction[answer];

                if (data.lable[answer] == 1)
                {
                    correct_prediction++;
                    this.average_certainty += certainty * 100;
                }
                else 
                {
                    false_prediction++;
                }
                count++;
                if (count % 100 == 0) progress.Report([count, list_length]);
            }
            
            this.average_certainty = this.average_certainty / correct_prediction;
        }

        private int GetAnswer(double[] prediction)
        {
            int     answer = 0;
            double  value  = 0;

            for (int index = 0; index < prediction.Length; index++) 
            {
                if (prediction[index] > value)
                {
                    value  = prediction[index];
                    answer = index;
                }
            }

            return answer;
        }

        public void LoadData(IProgress<int> progress, string path = null)
        {
            if (path != null) test_data = Data.LoadData(path, progress);
            else test_data = Data.LoadData(Serialization.GetPathToFolder(default_path), progress);
        }
    }
}
