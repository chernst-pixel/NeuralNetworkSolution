using NeuralNet.Core;
using NeuralNet.Core.Serialization;
using System.Windows;

namespace NeuralNet.Training.WPF
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private bool    network_loaded, training_loaded, test_loaded = false;
        NeuralNetwork   network;
        Epoch           epoch = new Epoch();
        Test            test  = new Test();
        private int     epochs = 1;
        private int[][] network_structure =  [
                                    [784, 128],
                                    [128,  64],
                                    [ 64,  10],
                                    ];
        public MainWindow()
        {
            InitializeComponent();
            StatusUpdate();
        }

        private void StatusUpdate()
        {
            string status       = $"KI geladen: \n{network_loaded}\n" +
                                  $"Training Daten geladen: \n{training_loaded}\n" +
                                  $"Test Daten geladen: \n{test_loaded}";
            
            Dispatcher.Invoke(() =>
            {
                lbl_status.Content = status;
                if(network_loaded && training_loaded)   btn_start_epoch.IsEnabled = true;
                else                                    btn_start_epoch.IsEnabled = false;
                if(network_loaded && test_loaded)       btn_start_test.IsEnabled  = true;
                else                                    btn_start_test.IsEnabled  = false;
            });
        }

        private void ErrorMessage(string e, string message)
        {
            string error         = $"{e} \n {message}";

            Dispatcher.Invoke(() =>
            {
                lbl_error.Content = error;
            });       
        }

        private async void btn_load_model_Click(object sender, RoutedEventArgs e)
        {
            await Task.Run(() =>
            {
                try
                {
                    network         = Serialization.Deserialize();
                    network_loaded  = true;
                }
                catch (Exception e)
                {
                    network         = new NeuralNetwork();
                    network_loaded  = true;
                    ErrorMessage(e.Message, "Erstelle neues netzwerk.");
                }
            });

            StatusUpdate();
        }

        private async void btn_load_training_Click(object sender, RoutedEventArgs e)
        {
            var progress = new Progress<int>(value =>
            {
                lbl_update.Content = value;
            });

            await Task.Run(() =>
            {
                try
                {
                    epoch.LoadData(progress);
                    training_loaded = true;
                }
                catch (Exception e) 
                {
                    ErrorMessage(e.Message, "Error beim Laden der Trainings Daten");
                }
            });

            lbl_update.Content = "Trainings Daten geladen!";
            StatusUpdate();
        }

        private async void btn_load_test_Click(object sender, RoutedEventArgs e)
        {
            var progress = new Progress<int>(value =>
            {
                lbl_update.Content = value;
            });

            await Task.Run(() =>
            {
                try
                {
                    test.LoadData(progress);
                    test_loaded = true;
                }
                catch (Exception e)
                {
                    ErrorMessage(e.Message, "Error beim Laden der Test Daten");
                }
            });

            lbl_update.Content = "Test Daten geladen!";
            StatusUpdate();
        }

        private async void btn_start_epoch_Click(object sender, RoutedEventArgs e)
        {
            var progress = new Progress<int[]>(value =>
            {
                lbl_update.Content = $"Epoch: {value[0] + 1} Daten {value[1]} von {value[2]}";
            });

            double[] losses = new double[epochs];
            string   report = "";

            await Task.Run(() =>
            {
                losses = epoch.RunEpoch(epochs, progress, network);
            });

            for(int epoch = 0; epoch < epochs; epoch++)
            {
                report += $"\nEpoch {epoch + 1} Loss : {losses[epoch]:0.0000}";
            }

            lbl_update.Content = "Epochs beendet" + report;
        }

        private async void btn_start_test_Click(object sender, RoutedEventArgs e)
        {
            var progress = new Progress<int[]>(value =>
            {
                lbl_update.Content = $"Daten {value[0]} von {value[1]} getestet";
            });

            await Task.Run(() =>
            {
                test.RunTest(progress, network);
                
            });

            lbl_result.Content = $"Richtige ergebnisse: {test.correct_prediction} \nFalsche ergebnisse: {test.false_prediction} \nDurchschnitts Sicherheit: {test.average_certainty:0.00}";
            lbl_update.Content = "Test beendet";
        }
    }
}