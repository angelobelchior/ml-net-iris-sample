using Microsoft.ML;
using System;
using System.IO;

namespace MLNetEstudos.Iris
{
    public class Prediction
    {
        public Prediction()
        {
            if (!File.Exists(Constants.PATH))
                throw new FileNotFoundException("Dataset not found");
        }

        public void Execute(Models.Data data, string modelPath)
        {
            var context = new MLContext();
            var model = context.Model.Load(modelPath, out DataViewSchema schema);
            var predictionEngine = context.Model.CreatePredictionEngine<Models.Data, Models.Prediction>(model);
            var prediction = predictionEngine.Predict(data);
            Console.WriteLine("===============================================");
            Console.WriteLine("+================Prediction===================+");
            Console.WriteLine("===============================================");
            for (int i = 0; i < prediction.Score.Length; i++)
                Console.WriteLine($"Class {i} => {prediction.Score[i]}");
            Console.WriteLine("===============================================");
            Console.WriteLine();
        }

        private void Predict(string modelPath, Models.Data data)
        {

        }
    }
}
