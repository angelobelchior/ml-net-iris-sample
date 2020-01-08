using Microsoft.ML.Data;
using System;
using System.Collections.Generic;

namespace MLNetEstudos.Helpers
{
    public static class Print
    {
        public static void MulticlassClassificationMetrics(string algorithm, MulticlassClassificationMetrics metrics)
        {
            var separator = "+=======================================================+";
            Console.WriteLine(separator);
            Console.WriteLine($"|=========== {algorithm} ===========|");
            Console.WriteLine(separator);
            Console.WriteLine($"| LogLoss          => {metrics.LogLoss}");
            Console.WriteLine($"| LogLossReduction => {metrics.LogLossReduction}");
            Console.WriteLine($"| MacroAccuracy    => {metrics.MacroAccuracy}");
            Console.WriteLine($"| MicroAccuracy    => {metrics.MicroAccuracy}");
            Console.WriteLine($"| TopKAccuracy     => {metrics.TopKPredictionCount}");
            Console.WriteLine($"| PerClassLogLoss");
            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
                Console.WriteLine($"|  - Class {i} => {metrics.PerClassLogLoss[i]}");
            Console.WriteLine($"| ConfusionMatrix");
            Console.WriteLine($"|  - NumberOfClasses   => {metrics.ConfusionMatrix.NumberOfClasses}");
            Console.WriteLine($"|  - PerClassPrecision => {string.Join('|', metrics.ConfusionMatrix.PerClassPrecision)}");
            Console.WriteLine($"|  - PerClassRecall    => {string.Join('|', metrics.ConfusionMatrix.PerClassRecall)}");
            Console.WriteLine($"|  - Counts");
            PrintMatrix(metrics.ConfusionMatrix.Counts);
            Console.WriteLine(separator);
            Console.WriteLine("");
            Console.WriteLine("");
        }


        static void PrintMatrix(IReadOnlyList<IReadOnlyList<double>> matrix)
        {
            Console.WriteLine();
            Console.Write("        |");
            var header = "";
            for (int i = 0; i < matrix.Count; i++)
            {
                header += new string(' ', 5 - i.ToString().Length - 1) + $"P{i} |";
            }
            Console.WriteLine(header);
            Console.WriteLine("     " + new string('-', header.Length + 4));
            for (int i = 0; i < matrix.Count; i++)
            {
                var obs = new string(' ', (5 - i.ToString().Length + 1)) + $"O{i}";
                Console.Write($"{obs} |");

                for (int j = 0; j < matrix[i].Count; j++)
                {
                    var text = new string(' ', (5 - matrix[i][j].ToString().Length)) + matrix[i][j];
                    Console.Write($"{text} |");
                }
                Console.WriteLine();
            }
        }
    }
}