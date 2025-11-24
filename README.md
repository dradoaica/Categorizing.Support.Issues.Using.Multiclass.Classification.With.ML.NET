# GitHub Issue Classification with ML.NET

This project demonstrates how to use ML.NET to categorize GitHub issues into different areas (labels) using a
multi-class classification model.

## Overview

The application trains a machine learning model using the **SDCA Maximum Entropy** algorithm to classify issues based on
their **Title** and **Description**.

| ML.NET version | API type    | Status     | App Type    | Data sources                | Scenario              | ML Task                    | Algorithms                  |
|----------------|-------------|------------|-------------|-----------------------------|-----------------------|----------------------------|-----------------------------|
| v5.0.0         | Dynamic API | Up-to-date | Console app | .tsv file and GitHub issues | Issues classification | Multi-class classification | SDCA multi-class classifier |

## Prerequisites

- [.NET 10.0 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) or later.
- Visual Studio 2026 or VS Code.

## Project Structure

- **GitHubIssueClassification**: The main console application.
    - **Data**: Contains training (`issues_train.tsv`) and test (`issues_test.tsv`) datasets.
    - **Models**: Stores the trained model (`model.zip`).
    - **Common**: Helper classes for console output.
    - `Program.cs`: The entry point and main logic.
    - `GitHubIssueData.cs`: Data models for input and prediction.

## How to Run

1. Clone the repository.
2. Navigate to the `GitHubIssueClassification` directory.
3. Run the application using the .NET CLI:

   ```bash
   dotnet run
   ```

## Code Overview

The `Program.cs` file follows these steps:

1. **Load Data**: Loads training and test data from TSV files.
2. **Process Data**: Transforms text data (Title, Description) into numeric features and maps labels to keys.
3. **Train Model**: Trains the model using the SDCA Maximum Entropy algorithm.
4. **Evaluate Model**: Evaluates the model's performance using the test dataset.
5. **Save Model**: Saves the trained model to a `.zip` file.
6. **Predict**: Demonstrates single predictions using both the in-memory model and the loaded model.
