using GitHubIssueClassification;
using GitHubIssueClassification.Common;
using Microsoft.ML;

var applicationPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]) ?? ".";
var trainingDataPath = Path.Combine(applicationPath, "..", "..", "..", "Data", "issues_train.tsv");
var testDataPath = Path.Combine(applicationPath, "..", "..", "..", "Data", "issues_test.tsv");
var modelPath = Path.Combine(applicationPath, "..", "..", "..", "Models", "model.zip");

// Create MLContext to be shared across the model creation workflow objects.
// Set a random seed for repeatable/deterministic results across multiple trainings.
var mlContext = new MLContext(0);
try
{
    // 1. Load Data
    var (trainingDataView, testDataView) = LoadData(mlContext, trainingDataPath, testDataPath);
    // 2. Process Data
    var pipeline = ProcessData(mlContext);
    // 3. Train Model
    var trainingPipeline = TrainModel(mlContext, pipeline, trainingDataView);
    var model = trainingPipeline.Fit(trainingDataView);
    // 4. Evaluate Model
    EvaluateModel(mlContext, model, testDataView);
    // 5. Save Model
    SaveModel(mlContext, model, trainingDataView.Schema, modelPath);
    // 6. Predict
    PredictIssue(mlContext, model);
    PredictIssueFromFile(mlContext, modelPath);
}
catch (Exception ex)
{
    ConsoleHelper.ConsoleWriteException(ex.Message);
}

ConsoleHelper.ConsolePressAnyKey();

return 0;

(IDataView trainingData, IDataView testData) LoadData(MLContext mlCtx, string trainingPath, string testPath)
{
    ConsoleHelper.ConsoleWriteHeader("Loading Dataset");
    if (!File.Exists(trainingPath))
    {
        throw new FileNotFoundException($"Training data not found at {trainingPath}");
    }

    if (!File.Exists(testPath))
    {
        throw new FileNotFoundException($"Test data not found at {testPath}");
    }

    var trainingDataView = mlCtx.Data.LoadFromTextFile<GitHubIssue>(trainingPath, hasHeader: true);
    var testDataView = mlCtx.Data.LoadFromTextFile<GitHubIssue>(testPath, hasHeader: true);
    ConsoleHelper.ConsoleWriteHeader("Finished Loading Dataset");

    return (trainingDataView, testDataView);
}

IEstimator<ITransformer> ProcessData(MLContext mlCtx)
{
    ConsoleHelper.ConsoleWriteHeader("Processing Data");
    var pipeline = mlCtx.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
        .Append(mlCtx.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
        .Append(
            mlCtx.Transforms.Text.FeaturizeText(
                inputColumnName: "Description",
                outputColumnName: "DescriptionFeaturized"
            )
        )
        .Append(mlCtx.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
        .AppendCacheCheckpoint(mlCtx);
    ConsoleHelper.ConsoleWriteHeader("Finished Processing Data");

    return pipeline;
}

IEstimator<ITransformer> TrainModel(MLContext mlCtx, IEstimator<ITransformer> pipeline, IDataView trainingData)
{
    ConsoleHelper.ConsoleWriteHeader("Creating and Training the Model");
    IEstimator<ITransformer> trainer;
    const int selectedStrategy = 0; // 0: SDCA Maximum Entropy, 1: OVA with Averaged Perceptron
    switch (selectedStrategy)
    {
        case 0:
            trainer = mlCtx.MulticlassClassification.Trainers.SdcaMaximumEntropy();
            break;
        case 1:
            {
                // Create a binary classification trainer.
                var averagedPerceptronBinaryTrainer = mlCtx.BinaryClassification.Trainers.AveragedPerceptron();
                // Compose an OVA (One-Versus-All) trainer with the BinaryTrainer.
                // In this strategy, a binary classification algorithm is used to train one classifier for each class,
                // which distinguishes that class from all other classes. Prediction is then performed by running these binary classifiers,
                // and choosing the prediction with the highest confidence score.
                trainer = mlCtx.MulticlassClassification.Trainers.OneVersusAll(averagedPerceptronBinaryTrainer);

                break;
            }
    }

    var trainingPipeline = pipeline.Append(trainer).Append(mlCtx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
    ConsoleHelper.ConsoleWriteHeader("Cross-validating to get model's accuracy metrics");
    var crossValidationResults = mlCtx.MulticlassClassification.CrossValidate(trainingData, trainingPipeline);
    ConsoleHelper.PrintMulticlassClassificationFoldsAverageMetrics(trainer.ToString(), crossValidationResults);
    ConsoleHelper.ConsoleWriteHeader("Finished Cross-validating to get model's accuracy metrics");
    ConsoleHelper.ConsoleWriteHeader("Finished Creating and Training the Model");

    return trainingPipeline;
}

void EvaluateModel(MLContext mlCtx, ITransformer model, IDataView testData)
{
    ConsoleHelper.ConsoleWriteHeader("Evaluating Model");
    var predictions = model.Transform(testData);
    var metrics = mlCtx.MulticlassClassification.Evaluate(predictions);
    ConsoleHelper.PrintMultiClassClassificationMetrics(model.ToString(), metrics);
    ConsoleHelper.ConsoleWriteHeader("Finished Evaluating Model");
}

void SaveModel(MLContext mlCtx, ITransformer model, DataViewSchema schema, string path)
{
    ConsoleHelper.ConsoleWriteHeader($"Saving Model to {path}");
    var directory = Path.GetDirectoryName(path);
    if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
    {
        Directory.CreateDirectory(directory);
    }

    mlCtx.Model.Save(model, schema, path);
    ConsoleHelper.ConsoleWriteHeader("Finished Saving Model");
}

void PredictIssue(MLContext mlCtx, ITransformer model)
{
    ConsoleHelper.ConsoleWriteHeader("Single Prediction");
    var predictionEngine = mlCtx.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(model);
    var issue = new GitHubIssue
    {
        Title = "WebSockets communication is slow in my machine",
        Description =
            "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine..",
    };
    var prediction = predictionEngine.Predict(issue);
    ConsoleHelper.ConsoleWriteHeader($"Single Prediction - Result: {prediction.Area}");
}

void PredictIssueFromFile(MLContext mlCtx, string path)
{
    ConsoleHelper.ConsoleWriteHeader($"Loading Model from {path} for Prediction");
    if (!File.Exists(path))
    {
        ConsoleHelper.ConsoleWriteWarning($"Model file not found at {path}");
        return;
    }

    var loadedModel = mlCtx.Model.Load(path, out _);
    var predictionEngine = mlCtx.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
    var issue = new GitHubIssue
    {
        Title = "Entity Framework crashes",
        Description = "When connecting to the database, EF is crashing",
    };
    var prediction = predictionEngine.Predict(issue);
    ConsoleHelper.ConsoleWriteHeader($"Single Prediction - Result: {prediction.Area}");
}
