# GitHub Issue Classification with ML.NET

## Overview

The application `GitHubIssueClassification` trains a machine learning model using the **SDCA Maximum Entropy** or the
**OVA with Averaged Perceptron** algorithm to classify issues based on their **Title** and **Description**.

| ML.NET version | API type    | Status     | App Type    | Data sources                | Scenario              | ML Task                    | Algorithms                  |
|----------------|-------------|------------|-------------|-----------------------------|-----------------------|----------------------------|-----------------------------|
| v5.0.0         | Dynamic API | Up-to-date | Console app | .tsv file and GitHub issues | Issues classification | Multi-class classification | SDCA multi-class classifier |

For a detailed explanation of how to build this application, see the
accompanying [tutorial](https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/github-issue-classification)
on the Microsoft Docs site.

## Prerequisites

- [.NET 10.0 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) or later.
- Visual Studio 2026 or VS Code.

## Problem

This project demonstrates how to use ML.NET to categorize GitHub issues into different areas (labels) using a
multi-class classification model.

## Project Structure

- **GitHubIssueClassification**: The main console application.
    - **Common**: Helper methods for formatting console output.
    - **Data**: Contains training (`issues_train.tsv`) and test (`issues_test.tsv`) datasets.
    - **Models**: Stores the trained model (`model.zip`).
    - `GitHubIssueData.cs`: Data models for input and prediction.
    - `Program.cs`: The main entry point.

## How to Run

1. **Open the Solution**: Open `GitHubIssueClassification.slnx` or the project folder in your IDE.
2. **Restore Dependencies**: Run `dotnet restore` to install the required NuGet packages.
3. **Run the Application**: Run the application using your IDE or `dotnet run`.

## How it Works

1. **Load Data**: Loads training and test data from TSV files.
2. **Process Data**: Transforms text data (Title, Description) into numeric features and maps labels to keys.
3. **Train Model**: Trains the model using the SDCA Maximum Entropy algorithm.
4. **Evaluate Model**: Evaluates the model's performance using the test dataset.
5. **Save Model**: Saves the trained model to a `.zip` file.
6. **Predict**: Demonstrates single predictions using both the in-memory model and the loaded model.
