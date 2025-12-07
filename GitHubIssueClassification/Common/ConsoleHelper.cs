using Microsoft.ML;
using Microsoft.ML.Data;
using System.Diagnostics;
using static Microsoft.ML.TrainCatalogBase;

namespace GitHubIssueClassification.Common;

/// <summary>Provides helper methods for console output and metrics display.</summary>
public static class ConsoleHelper
{
    /// <summary>Prints the prediction result to the console.</summary>
    /// <param name="prediction">The predicted value.</param>
    public static void PrintPrediction(string prediction)
    {
        Console.WriteLine("*************************************************");
        Console.WriteLine($"Predicted : {prediction}");
        Console.WriteLine("*************************************************");
    }

    /// <summary>Prints the regression prediction versus the observed value.</summary>
    /// <param name="predictionCount">The predicted count.</param>
    /// <param name="observedCount">The actual observed count.</param>
    public static void PrintRegressionPredictionVersusObserved(string predictionCount, string observedCount)
    {
        Console.WriteLine("-------------------------------------------------");
        Console.WriteLine($"Predicted : {predictionCount}");
        Console.WriteLine($"Actual:     {observedCount}");
        Console.WriteLine("-------------------------------------------------");
    }

    /// <summary>Prints regression metrics to the console.</summary>
    /// <param name="name">The name of the model.</param>
    /// <param name="metrics">The regression metrics.</param>
    public static void PrintRegressionMetrics(string name, RegressionMetrics metrics)
    {
        Console.WriteLine("*************************************************");
        Console.WriteLine($"*       Metrics for {name} regression model      ");
        Console.WriteLine("*------------------------------------------------");
        Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
        Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
        Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
        Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
        Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
        Console.WriteLine("*************************************************");
    }

    /// <summary>Prints binary classification metrics to the console.</summary>
    /// <param name="name">The name of the model.</param>
    /// <param name="metrics">The binary classification metrics.</param>
    public static void PrintBinaryClassificationMetrics(string name, CalibratedBinaryClassificationMetrics metrics)
    {
        Console.WriteLine("************************************************************");
        Console.WriteLine($"*       Metrics for {name} binary classification model      ");
        Console.WriteLine("*-----------------------------------------------------------");
        Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"*       Area Under Curve:      {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"*       Area under Precision recall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");
        Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
        Console.WriteLine($"*       LogLoss:  {metrics.LogLoss:#.##}");
        Console.WriteLine($"*       LogLossReduction:  {metrics.LogLossReduction:#.##}");
        Console.WriteLine($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");
        Console.WriteLine($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");
        Console.WriteLine($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");
        Console.WriteLine($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");
        Console.WriteLine("************************************************************");
    }

    /// <summary>Prints anomaly detection metrics to the console.</summary>
    /// <param name="name">The name of the model.</param>
    /// <param name="metrics">The anomaly detection metrics.</param>
    public static void PrintAnomalyDetectionMetrics(string name, AnomalyDetectionMetrics metrics)
    {
        Console.WriteLine("************************************************************");
        Console.WriteLine($"*       Metrics for {name} anomaly detection model      ");
        Console.WriteLine("*-----------------------------------------------------------");
        Console.WriteLine($"*       Area Under ROC Curve:                       {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine(
            $"*       Detection rate at false positive count: {metrics.DetectionRateAtFalsePositiveCount}"
        );
        Console.WriteLine("************************************************************");
    }

    /// <summary>Prints multi-class classification metrics to the console.</summary>
    /// <param name="name">The name of the model.</param>
    /// <param name="metrics">The multi-class classification metrics.</param>
    public static void PrintMultiClassClassificationMetrics(string name, MulticlassClassificationMetrics metrics)
    {
        Console.WriteLine("************************************************************");
        Console.WriteLine($"*    Metrics for {name} multi-class classification model   ");
        Console.WriteLine("*-----------------------------------------------------------");
        Console.WriteLine(
            $"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better"
        );
        Console.WriteLine(
            $"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better"
        );
        Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");

        for (var i = 0; i < metrics.PerClassLogLoss.Count; i++)
        {
            Console.WriteLine(
                $"    LogLoss for class {i + 1} = {metrics.PerClassLogLoss[i]:0.####}, the closer to 0, the better"
            );
        }

        Console.WriteLine("************************************************************");
    }

    /// <summary>Prints regression folds average metrics.</summary>
    /// <param name="algorithmName">The name of the algorithm.</param>
    /// <param name="crossValidationResults">The cross-validation results.</param>
    public static void PrintRegressionFoldsAverageMetrics(
        string algorithmName,
        IReadOnlyList<CrossValidationResult<RegressionMetrics>> crossValidationResults
    )
    {
        var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
        var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
        var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
        var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
        var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);

        Console.WriteLine(
            "*************************************************************************************************************"
        );
        Console.WriteLine($"*       Metrics for {algorithmName} Regression model      ");
        Console.WriteLine(
            "*------------------------------------------------------------------------------------------------------------"
        );
        Console.WriteLine($"*       Average L1 Loss:    {L1.Average():0.###} ");
        Console.WriteLine($"*       Average L2 Loss:    {L2.Average():0.###}  ");
        Console.WriteLine($"*       Average RMS:          {RMS.Average():0.###}  ");
        Console.WriteLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
        Console.WriteLine($"*       Average R-squared: {R2.Average():0.###}  ");
        Console.WriteLine(
            "*************************************************************************************************************"
        );
    }

    /// <summary>Prints multi-class classification folds average metrics.</summary>
    /// <param name="algorithmName">The name of the algorithm.</param>
    /// <param name="crossValResults">The cross-validation results.</param>
    public static void PrintMulticlassClassificationFoldsAverageMetrics(
        string algorithmName,
        IReadOnlyList<CrossValidationResult<MulticlassClassificationMetrics>> crossValResults
    )
    {
        var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

        var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
        var microAccuracyAverage = microAccuracyValues.Average();
        var microAccuraciesStdDeviation = CalculateStandardDeviation(microAccuracyValues);
        var microAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(microAccuracyValues);

        var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
        var macroAccuracyAverage = macroAccuracyValues.Average();
        var macroAccuraciesStdDeviation = CalculateStandardDeviation(macroAccuracyValues);
        var macroAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(macroAccuracyValues);

        var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
        var logLossAverage = logLossValues.Average();
        var logLossStdDeviation = CalculateStandardDeviation(logLossValues);
        var logLossConfidenceInterval95 = CalculateConfidenceInterval95(logLossValues);

        var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
        var logLossReductionAverage = logLossReductionValues.Average();
        var logLossReductionStdDeviation = CalculateStandardDeviation(logLossReductionValues);
        var logLossReductionConfidenceInterval95 = CalculateConfidenceInterval95(logLossReductionValues);

        Console.WriteLine(
            "*************************************************************************************************************"
        );
        Console.WriteLine($"*       Metrics for {algorithmName} Multi-class Classification model      ");
        Console.WriteLine(
            "*------------------------------------------------------------------------------------------------------------"
        );
        Console.WriteLine(
            $"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})"
        );
        Console.WriteLine(
            $"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})"
        );
        Console.WriteLine(
            $"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})"
        );
        Console.WriteLine(
            $"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})"
        );
        Console.WriteLine(
            "*************************************************************************************************************"
        );
    }

    /// <summary>Calculates the standard deviation of a sequence of values.</summary>
    /// <param name="values">The sequence of values.</param>
    /// <returns>The standard deviation.</returns>
    public static double CalculateStandardDeviation(IEnumerable<double> values)
    {
        var average = values.Average();
        var sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
        var standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
        return standardDeviation;
    }

    /// <summary>Calculates the 95% confidence interval of a sequence of values.</summary>
    /// <param name="values">The sequence of values.</param>
    /// <returns>The 95% confidence interval.</returns>
    public static double CalculateConfidenceInterval95(IEnumerable<double> values)
    {
        var confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt(values.Count() - 1);
        return confidenceInterval95;
    }

    /// <summary>Prints clustering metrics to the console.</summary>
    /// <param name="name">The name of the model.</param>
    /// <param name="metrics">The clustering metrics.</param>
    public static void PrintClusteringMetrics(string name, ClusteringMetrics metrics)
    {
        Console.WriteLine("*************************************************");
        Console.WriteLine($"*       Metrics for {name} clustering model      ");
        Console.WriteLine("*------------------------------------------------");
        Console.WriteLine($"*       Average Distance: {metrics.AverageDistance}");
        Console.WriteLine($"*       Davies Bouldin Index is: {metrics.DaviesBouldinIndex}");
        Console.WriteLine("*************************************************");
    }

    /// <summary>Shows the data in the DataView in the console.</summary>
    /// <param name="mlContext">The ML context.</param>
    /// <param name="dataView">The data view.</param>
    /// <param name="numberOfRows">The number of rows to show.</param>
    public static void ShowDataViewInConsole(MLContext mlContext, IDataView dataView, int numberOfRows = 4)
    {
        var msg = string.Format("Show data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());
        ConsoleWriteHeader(msg);

        var preViewTransformedData = dataView.Preview(numberOfRows);

        foreach (var row in preViewTransformedData.RowView)
        {
            var ColumnCollection = row.Values;
            var lineToPrint = "Row--> ";
            foreach (KeyValuePair<string, object> column in ColumnCollection)
            {
                lineToPrint += $"| {column.Key}:{column.Value}";
            }

            Console.WriteLine(lineToPrint + "\n");
        }
    }

    /// <summary>Peeks at the data in the DataView in the console.</summary>
    /// <param name="mlContext">The ML context.</param>
    /// <param name="dataView">The data view.</param>
    /// <param name="pipeline">The pipeline.</param>
    /// <param name="numberOfRows">The number of rows to show.</param>
    [Conditional("DEBUG")]
    // This method using 'DebuggerExtensions.Preview()' should only be used when debugging/developing, not for release/production trainings
    public static void PeekDataViewInConsole(
        MLContext mlContext,
        IDataView dataView,
        IEstimator<ITransformer> pipeline,
        int numberOfRows = 4
    )
    {
        var msg = string.Format("Peek data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());
        ConsoleWriteHeader(msg);

        //https://github.com/dotnet/machinelearning/blob/main/docs/code/MlNetCookBook.md#how-do-i-look-at-the-intermediate-data
        var transformer = pipeline.Fit(dataView);
        var transformedData = transformer.Transform(dataView);

        // 'transformedData' is a 'promise' of data, lazy-loading. call Preview
        //and iterate through the returned collection from preview.

        var preViewTransformedData = transformedData.Preview(numberOfRows);

        foreach (var row in preViewTransformedData.RowView)
        {
            var ColumnCollection = row.Values;
            var lineToPrint = "Row--> ";
            foreach (KeyValuePair<string, object> column in ColumnCollection)
            {
                lineToPrint += $"| {column.Key}:{column.Value}";
            }

            Console.WriteLine(lineToPrint + "\n");
        }
    }

    /// <summary>Peeks at the vector column data in the DataView in the console.</summary>
    /// <param name="mlContext">The ML context.</param>
    /// <param name="columnName">The column name.</param>
    /// <param name="dataView">The data view.</param>
    /// <param name="pipeline">The pipeline.</param>
    /// <param name="numberOfRows">The number of rows to show.</param>
    [Conditional("DEBUG")]
    // This method using 'DebuggerExtensions.Preview()' should only be used when debugging/developing, not for release/production trainings
    public static void PeekVectorColumnDataInConsole(
        MLContext mlContext,
        string columnName,
        IDataView dataView,
        IEstimator<ITransformer> pipeline,
        int numberOfRows = 4
    )
    {
        var msg = string.Format(
            "Peek data in DataView: : Show {0} rows with just the '{1}' column",
            numberOfRows,
            columnName
        );
        ConsoleWriteHeader(msg);

        var transformer = pipeline.Fit(dataView);
        var transformedData = transformer.Transform(dataView);

        // Extract the 'Features' column.
        var someColumnData = transformedData.GetColumn<float[]>(columnName).Take(numberOfRows).ToList();

        // print to console the peeked rows

        var currentRow = 0;
        someColumnData.ForEach(row =>
            {
                currentRow++;
                var concatColumn = string.Empty;
                foreach (var f in row)
                {
                    concatColumn += f.ToString();
                }

                Console.WriteLine();
                var rowMsg = string.Format("**** Row {0} with '{1}' field value ****", currentRow, columnName);
                Console.WriteLine(rowMsg);
                Console.WriteLine(concatColumn);
                Console.WriteLine();
            }
        );
    }

    /// <summary>Writes a header to the console.</summary>
    /// <param name="lines">The lines to write.</param>
    public static void ConsoleWriteHeader(params string[] lines)
    {
        var defaultColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine(" ");
        foreach (var line in lines)
        {
            Console.WriteLine(line);
        }

        var maxLength = lines.Select(x => x.Length).Max();
        Console.WriteLine(new string('#', maxLength));
        Console.ForegroundColor = defaultColor;
    }

    /// <summary>Writes a section header to the console.</summary>
    /// <param name="lines">The lines to write.</param>
    public static void ConsoleWriterSection(params string[] lines)
    {
        var defaultColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Blue;
        Console.WriteLine(" ");
        foreach (var line in lines)
        {
            Console.WriteLine(line);
        }

        var maxLength = lines.Select(x => x.Length).Max();
        Console.WriteLine(new string('-', maxLength));
        Console.ForegroundColor = defaultColor;
    }

    /// <summary>Waits for the user to press any key.</summary>
    public static void ConsolePressAnyKey()
    {
        var defaultColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine(" ");
        Console.WriteLine("Press any key to finish.");
        Console.ReadKey();
    }

    /// <summary>Writes an exception to the console.</summary>
    /// <param name="lines">The lines to write.</param>
    public static void ConsoleWriteException(params string[] lines)
    {
        var defaultColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Red;
        const string exceptionTitle = "EXCEPTION";
        Console.WriteLine(" ");
        Console.WriteLine(exceptionTitle);
        Console.WriteLine(new string('#', exceptionTitle.Length));
        Console.ForegroundColor = defaultColor;
        foreach (var line in lines)
        {
            Console.WriteLine(line);
        }
    }

    /// <summary>Writes a warning to the console.</summary>
    /// <param name="lines">The lines to write.</param>
    public static void ConsoleWriteWarning(params string[] lines)
    {
        var defaultColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.DarkMagenta;
        const string warningTitle = "WARNING";
        Console.WriteLine(" ");
        Console.WriteLine(warningTitle);
        Console.WriteLine(new string('#', warningTitle.Length));
        Console.ForegroundColor = defaultColor;
        foreach (var line in lines)
        {
            Console.WriteLine(line);
        }
    }
}
