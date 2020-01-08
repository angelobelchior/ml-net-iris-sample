namespace MLNetEstudos
{
    public static class Program
    {
        static void Main(string[] args)
        {
            //var training = new Iris.Training();
            //training.Execute();

            //Iris Setosa 0
            //Iris Versicolour 1
            //Iris Virginica 2

            #region
            
            //"LbfgsMaximumEntropy.zip"
            //NaiveBayes.zip
            //SdcaMaximumEntropy.zip
            //SdcaNonCalibrated.zip
            var modelPath = System.IO.Path.Combine(Constants.MODELS_PATH, "LbfgsMaximumEntropy.zip");
            var irisSetosa = new Iris.Models.Data(5.7F, 3.8F, 1.7F, 0.3F); //0 Iris Setosa
            var irisVersicolour = new Iris.Models.Data(6.3F, 2.5F, 4.9F, 1.5F); //1 Iris Versicolour
            var irisVirginica = new Iris.Models.Data(6.9F, 3.1F, 5.1F, 2.3F);//2 Iris Virginica
            var prediction = new Iris.Prediction();
            prediction.Execute(irisSetosa, modelPath);
            prediction.Execute(irisVersicolour, modelPath);
            prediction.Execute(irisVirginica, modelPath);
            
            #endregion
        }
    }
}
