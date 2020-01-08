using Microsoft.ML;
using System;
using System.IO;
using MLNetEstudos.Helpers;
using System.Collections.Generic;

namespace MLNetEstudos.Iris
{
    public class Training
    {
        public Training()
        {
            if (!File.Exists(Constants.PATH))
                throw new FileNotFoundException("Dataset not found");
        }

        public void Execute()
        {
            var context = new MLContext(seed: 1234);
            var trainTestData = this.LoadData(context);

            var pipeline = context.Transforms
                                  .Concatenate("Features",
                                               nameof(Models.Data.SepalWidth),
                                               nameof(Models.Data.SepalLength),
                                               nameof(Models.Data.PetalWidth),
                                               nameof(Models.Data.PetalLength))
                                  .Append(context.Transforms
                                                 .Conversion
                                                 .MapValueToKey(outputColumnName: "Label",
                                                                inputColumnName: "Label"))
                                  .AppendCacheCheckpoint(context);

            var trainer = context.MulticlassClassification.Trainers;
            var algorithms = new Dictionary<string, IEstimator<ITransformer>>
            {
                ["SdcaNonCalibrated"] = trainer.SdcaNonCalibrated(labelColumnName: "Label", featureColumnName: "Features"),
                ["NaiveBayes"] = trainer.NaiveBayes(labelColumnName: "Label", featureColumnName: "Features"),
                ["LbfgsMaximumEntropy"] = trainer.LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"),
                ["SdcaMaximumEntropy"] = trainer.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"),
            };

            foreach (var algorithm in algorithms)
            {
                var model = pipeline.Append(algorithm.Value)
                                    .Fit(trainTestData.TrainSet);

                var predictions = model.Transform(trainTestData.TestSet);
                var metrics = context.MulticlassClassification.Evaluate(predictions, "Label", "Score");
                Print.MulticlassClassificationMetrics(algorithm.Key, metrics);

                var modelPath = Path.Combine(Constants.MODELS_PATH, $"{algorithm.Key}.zip");
                context.Model.Save(model, trainTestData.TrainSet.Schema, modelPath);
            }
        }

        private DataOperationsCatalog.TrainTestData LoadData(MLContext context)
        {
            var dataView = context.Data.LoadFromTextFile<Models.Data>(Constants.PATH, ';');
            return context.Data.TrainTestSplit(dataView, testFraction: 0.30);
        }
    }
}
