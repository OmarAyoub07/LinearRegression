// Import Packages (Libraries)
using Microsoft.ML;
using LinearRegression;

/* Declaring Required Variables */
List<DataSet.LinearRegression> dataset = new List<DataSet.LinearRegression>();
MLContext mlContext = new MLContext(seed: 7985); // Seed for Random Number Generator
Methods.data _data = new Methods.data();
Methods.LR _lr = new Methods.LR();

/* Preparing the data */

// Get the data from NoSQL database by calling Map-Reduce using API
dataset = await _data.Get();

// Makking the values between 0 and 1, to predict visitors rate
_data.NormalizeBinning(ref dataset);

// Load dataset
IDataView dataView = mlContext.Data.LoadFromEnumerable<DataSet.LinearRegression>(dataset);

// split dataset into train, and test with 20% for testSet
var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

// Data Visualization represent visits in Times
_data.display<Single, float>(split, "Rate Of Visitors/Time", "time", "visits_num", "markers");

// Data Visualization represent visits on dates
//_data.display<string, float>(split, "Rate Of Visitors/Date", "date", "visits_num", "markers");


/* Processes of Machine Learning Model */
// Call Training Method
var model = _lr.Train(mlContext, split);

// Call Evaluation Method
var metrics = _lr.Evaluate(mlContext, model, split);

// Print (R^2) and (RMS)
_lr.display(metrics);

// Save the Model to reuse it
mlContext.Model.Save(model, dataView.Schema, Path.Combine(Environment.CurrentDirectory, "LinearRegressionModel", "Model.zip"));

// Predict One Sample
_lr.TestSinglePrediction(mlContext,
                         Path.Combine(Environment.CurrentDirectory, "LinearRegressionModel", "Model.zip"),
                         new string[] { "b279738fb9a444e49c69173a9379c137", "Friday - March", "21" });

// Data Visualization display the relation between Actual and desired results
_data.display_visitsPerTime(mlContext, split, Path.Combine(Environment.CurrentDirectory, "LinearRegressionModel", "Model.zip"));
