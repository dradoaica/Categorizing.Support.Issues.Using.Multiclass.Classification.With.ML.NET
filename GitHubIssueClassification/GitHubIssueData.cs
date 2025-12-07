using Microsoft.ML.Data;

namespace GitHubIssueClassification;

/// <summary>Represents a GitHub issue with its properties.</summary>
public class GitHubIssue
{
    /// <summary>Gets or sets the ID of the issue.</summary>
    [LoadColumn(0)]
    public string? ID { get; set; }

    /// <summary>Gets or sets the Area (label) of the issue.</summary>
    [LoadColumn(1)]
    public string? Area { get; set; }

    /// <summary>Gets or sets the Title of the issue.</summary>
    [LoadColumn(2)]
    public required string Title { get; set; }

    /// <summary>Gets or sets the Description of the issue.</summary>
    [LoadColumn(3)]
    public required string Description { get; set; }
}

/// <summary>Represents the prediction result for a GitHub issue.</summary>
public class IssuePrediction
{
    /// <summary>Gets or sets the predicted Area (label) for the issue.</summary>
    [ColumnName("PredictedLabel")]
    public string? Area { get; set; }
}
