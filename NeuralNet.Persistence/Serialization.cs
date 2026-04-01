using System.Text.Json;
using NeuralNet.Core;

namespace NeuralNet.Persistence
{
    public static class Serialization
    {
        static string model_folder  = "Models";
        static string model_name    = "network.json";
        public static void Serialize(NeuralNetwork network, string path=null)
        {
            string json = JsonSerializer.Serialize(network, new JsonSerializerOptions { WriteIndented = true });

            if(path != null)  File.WriteAllText(path, json);
            else              File.WriteAllText(GetPathToFile(model_folder, model_name), json);
        }

        public static NeuralNetwork Deserialize(string path=null) 
        {
            string json;
            if (path != null) json    = File.ReadAllText(path);
            else              json    = File.ReadAllText(GetPathToFile(model_folder, model_name));
            NeuralNetwork     network = JsonSerializer.Deserialize<NeuralNetwork>(json);
            return            network;
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
