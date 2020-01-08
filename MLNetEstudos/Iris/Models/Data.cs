using Microsoft.ML.Data;

namespace MLNetEstudos.Iris.Models
{
    public class Data
    {
        [LoadColumn(0)]
        public float SepalLength;
        [LoadColumn(1)]
        public float SepalWidth;
        [LoadColumn(2)]
        public float PetalLength;
        [LoadColumn(3)]
        public float PetalWidth;
        [LoadColumn(4)]
        public uint Label;

        public Data()
        {

        }

        public Data(float sepalLength, float sepalWidth, float petalLength, float petalWidth)
        {
            this.SepalLength = sepalLength;
            this.SepalWidth = sepalWidth;
            this.PetalLength = petalLength;
            this.PetalWidth = petalWidth;
        }
    }
}