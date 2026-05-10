using NeuralNet.Core;
using NeuralNet.Core.Serialization;

namespace NeuralNet.Training
{
    public class Epoch
    {
        private List<Data>      trainings_data;
        private string          default_path = @"MNIST\train";
        private static Random   random = new Random();

        public double[] RunEpoch(int epochs, IProgress<int[]> progress, NeuralNetwork network)
        {
            int      count;
            int      list_length = trainings_data.Count;
            double   epoch_loss  = 0.0;
            double[] losses = new double[epochs];
            
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                Shuffle<Data>(trainings_data);
                count = 0;
                foreach (Data data in trainings_data)
                {
                    epoch_loss +=  network.TrainSample(data.input, data.lable);
                    count++;
                    if (count % 100 == 0) progress.Report([epoch, count, list_length]);
                }
                epoch_loss     /= trainings_data.Count;
                losses[epoch]   = epoch_loss;
                epoch_loss      = 0.0;
                Serialization.Serialize(network);
            }
            
            return losses;
        }

        public void LoadData(IProgress<int> progress, string path = null)
        {
            if(path != null) trainings_data = Data.LoadData(path, progress);
            else             trainings_data = Data.LoadData(Serialization.GetPathToFolder(default_path), progress);
        }

        private static void Shuffle<T>(IList<T> list)
        {
            int    index  = list.Count;
            while (index > 1)
            {
                index--;
                int j = random.Next(index + 1);
                (list[j], list[index]) = (list[index], list[j]);
            }
        }
    }
}
