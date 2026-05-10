using NeuralNet.Core.Layers;
using NeuralNet.Core.Models;
using System.Text.Json;

namespace NeuralNet.Core.Serialization
{
    public static class Serialization
    {
        static string model_folder  = "Models";
        static string model_name    = "network.json";
        public static void Serialize(NeuralNetwork network, string path=null)
        {
            NeuralNetworkModel model = network.ExportModel();
            string json = JsonSerializer.Serialize(model, new JsonSerializerOptions { WriteIndented = true });

            if(path != null)  File.WriteAllText(path, json);
            else              File.WriteAllText(GetPathToFile(model_folder, model_name), json);
        }

        public static NeuralNetwork Deserialize(string path=null) 
        {
            NeuralNetwork network = new NeuralNetwork();
            network.ClearLayers();

            string json;
            if (path != null) json    = File.ReadAllText(path);
            else              json    = File.ReadAllText(GetPathToFile(model_folder, model_name));
            NeuralNetworkModel model  = JsonSerializer.Deserialize<NeuralNetworkModel>(json);


            foreach (var layer in model.Layers)
            {
                switch (layer.Type)
                {
                    case "Dense":
                        network.AddLayer(DenseLayer.FromModel(layer));
                        break;

                    case "ReLU":
                        network.AddLayer(new ReLULayer());
                        break;

                    default:
                        throw new NotSupportedException(
                            $"Unknown layer type: {layer.Type}");
                }
            }

            return network;
        }

        public static string GetPathToFile(string model_folder, string model_name)
        {
            string model_directory      = GetPathToFolder(model_folder);
            string model_path           = Path.Combine(model_directory, model_name);
        
            return model_path;
        }
        public static string GetPathToFolder(string model_folder)
        {
            string project_directory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
            string model_directory = Path.Combine(project_directory, model_folder);

            return model_directory;
        }
    }
}
