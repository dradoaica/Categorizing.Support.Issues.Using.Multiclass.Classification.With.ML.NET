using GitHubIssueClassification;
using GitHubIssueClassification.Common;
using Microsoft.ML;

var appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]) ?? ".";
var trainingDataPath = Path.Combine(appPath, "..", "..", "..", "Data", "issues_train.tsv");
var testDataPath = Path.Combine(appPath, "..", "..", "..", "Data", "issues_test.tsv");
var modelPath = Path.Combine(appPath, "..", "..", "..", "Models", "model.zip");

// Create MLContext to be shared across the model creation workflow objects.
// Set a random seed for repeatable/deterministic results across multiple trainings.
var mlContext = new MLContext(0);

// STEP 1: Common data loading configuration
// CreateTextReader<GitHubIssue>(hasHeader: true) - Creates a TextLoader by inferencing the dataset schema from the GitHubIssue data model type.
// .Read(trainingDataPath) - Loads the training text file into an IDataView (trainingDataView) and maps from input columns to IDataView columns.
Console.WriteLine("=============== Loading Dataset  ===============");
var trainingDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(trainingDataPath, hasHeader: true);
var testDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(testDataPath, hasHeader: true);
Console.WriteLine("=============== Finished Loading Dataset  ===============");

// STEP 2: Common data process configuration with pipeline data transformations
Console.WriteLine("=============== Processing Data ===============");
var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
    .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
    .Append(
        mlContext.Transforms.Text.FeaturizeText(
            inputColumnName: "Description",
            outputColumnName: "DescriptionFeaturized"
        )
    )
    .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
    // Sample Caching the IDataView so estimators iterating over the data multiple times, instead of always reading from file, using the cache might get better performance.
    .AppendCacheCheckpoint(mlContext);
// (OPTIONAL) Peek data (such as 2 records) in training IDataView after applying the ProcessPipeline's transformations into "Features".
ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, pipeline, 2);
Console.WriteLine("=============== Finished Processing Data ===============");

// STEP 3: Create the training algorithm/trainer
// Use the multi-class SDCA algorithm to predict the label using features.
// Set the trainer/algorithm and map label sto value (original readable state).
Console.WriteLine("=============== Create the training algorithm/trainer  ===============");
var trainingPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
// (OPTIONAL) Cross-Validate with training dataset in order to evaluate and get the model's accuracy metrics.
Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
var crossValidationResults = mlContext.MulticlassClassification.CrossValidate(trainingDataView, trainingPipeline, 6);
ConsoleHelper.PrintMulticlassClassificationFoldsAverageMetrics(trainingPipeline.ToString(), crossValidationResults);
Console.WriteLine("=============== Finished Create the training algorithm/trainer  ===============");

// STEP 4: Train the model by fitting to the dataset
Console.WriteLine("=============== Training the model  ===============");
ITransformer model = trainingPipeline.Fit(trainingDataView);
// (OPTIONAL) Try/test a single prediction with the "just-trained model" (before saving the model).
Console.WriteLine("=============== Single Prediction just-trained-model ===============");
// Create prediction engine related to the loaded trained model.
var predictionEngine = mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(model);
var issue = new GitHubIssue
{
    Title = "WebSockets communication is slow in my machine",
    Description =
        "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine..",
};
var prediction = predictionEngine.Predict(issue);
Console.WriteLine(
    $"=============== Finished Single Prediction just-trained-model - Result: {prediction.Area} ==============="
);
Console.WriteLine(
    $"=============== Finished Training the model Ending time: {DateTime.Now.ToString()} ==============="
);

// STEP 5:  Evaluate the model in order to get the model's accuracy metrics
Console.WriteLine(
    $"=============== Evaluating to get model's accuracy metrics - Starting time: {DateTime.Now.ToString()} ==============="
);
// Evaluate the model on a test dataset and calculate metrics of the model on the test data.
var testMetrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testDataView));
ConsoleHelper.PrintMultiClassClassificationMetrics(model.ToString(), testMetrics);
Console.WriteLine(
    $"=============== Finished Evaluating to get model's accuracy metrics - Ending time: {DateTime.Now.ToString()} ==============="
);

// STEP 6: Save/persist the trained model to a .ZIP file
Console.WriteLine($"=============== Saving the model to {modelPath} ===============");
mlContext.Model.Save(model, trainingDataView.Schema, modelPath);
Console.WriteLine($"=============== Finished Saving the model to {modelPath} ===============");

// STEP 7: Load the model from .ZIP file, and try a single prediction
Console.WriteLine($"=============== Loading the model from {modelPath}, and try a single prediction ===============");
var loadedModel = mlContext.Model.Load(modelPath, out _);
Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
var singleIssue = new GitHubIssue
{
    Title = "Entity Framework crashes",
    Description = "When connecting to the database, EF is crashing",
};
// Predict label for single hard-coded issue.
predictionEngine = mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
prediction = predictionEngine.Predict(singleIssue);
Console.WriteLine($"=============== Finished Single Prediction - Result: {prediction.Area} ===============");
Console.WriteLine(
    $"=============== Finished Loading the model from {modelPath}, and try a single prediction ==============="
);

return 0;
