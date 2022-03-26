using Newtonsoft.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using XPlot.Plotly;


namespace LinearRegression
{
    public class Methods
    {
        public class data
        {
            // Get Linear Regression Dataset from api
            public async Task<List<DataSet.LinearRegression>> Get()
            {
                // API uri
                string uri = "http://api.dalilak.pro/Query/LR_DataSet_";
                using (var client = new HttpClient())
                {
                    // initialize Get Request
                    var response = await client.GetAsync(uri);
                    var content = response.Content.ReadAsStringAsync().Result;

                    // Convert the content of the request from string to DataSet.LinearRegression
                    return JsonConvert.DeserializeObject<List<DataSet.LinearRegression>>(content);
                }
            }

            // function to make visitors numbers between 0 and 1
            public void NormalizeBinning(ref List<DataSet.LinearRegression> ds)
            {
                // Get maximum and minimu
                float max = ds.Select(x => x.visits_num).ToList().Max();
                float min = ds.Select(x => x.visits_num).ToList().Min();

                // Compress the numbers between 0 and 1
                foreach (var item in ds)
                {
                    item.visits_num = (item.visits_num - min)/(max - min);
                }
            }
           
            // data visualizer
            public void display<T1, T2>(DataOperationsCatalog.TrainTestData split, string title, string dependent, string independent, string mod)
            {
                // define dependent and independent values
                var x = split.TrainSet.GetColumn<T1>(dependent).ToArray();
                var y = split.TrainSet.GetColumn<T2>(independent).ToArray();

                // Options of chart
                var yearsChart = Chart.Plot(new Graph.Scatter
                {
                    x = x,
                    y = y,
                    mode = mod//"markers"
                });

                // set title and display the graph
                yearsChart.WithTitle(title);
                yearsChart.Show();
            }

            public void display_visitsPerTime(MLContext ml, DataOperationsCatalog.TrainTestData split, string modelPath)
            {
                // Loading model
                DataViewSchema modelSchema;
                ITransformer model = ml.Model.Load(modelPath, out modelSchema);

                // prediction Engine from loaded model
                var predictionFunction = ml.Model.CreatePredictionEngine<DataSet.LinearRegression, visitsDataset_Predictions>(model);

                // Desired data (20% trainSet), this load only times from dataSet
                var enumerableTestSet = ml.Data.CreateEnumerable<DataSet.LinearRegression>(split.TestSet, reuseRowObject: false)
                        .Select(ts => new DataSet.LinearRegression() { time = ts.time });

                // Engine will predict all samples within testSet
                var preductionResults = enumerableTestSet.Select(ts => predictionFunction.Predict(ts));

                // x-axis contains of all times
                var testHour = enumerableTestSet.Select(ts => ts.time).ToArray();

                // y-axis
                // Contain of desired and predicted visits rate 
                var desired = ml.Data.CreateEnumerable<DataSet.LinearRegression>(split.TestSet, reuseRowObject: false).Select(ts => ts.visits_num).ToArray();
                var actual = preductionResults.Select(r => r.visits_num).ToArray();

                // Graph options
                var actual_hv = new Graph.Scatter()
                {
                    x = testHour,
                    y = desired,
                    name = "Desired",
                    mode = "markers"

                };

                var predicted_hv = new Graph.Scatter()
                {
                    x = testHour,
                    y = actual,
                    name = "Predicted",
                    mode = "Line"

                };

                // Settingup and Display
                var chart = Chart.Plot(new[] { actual_hv, predicted_hv });
                var layout = new Layout.Layout() { title = "Visits Rate/Time" };
                chart.WithLayout(layout);
                chart.WithXTitle("Hours");
                chart.WithYTitle("Visits");
                chart.Show();
            }
        }

        public class LR
        {
            // Train the Model
            public ITransformer Train(MLContext ml, DataOperationsCatalog.TrainTestData d)
            {
                // Setting Options of the trainer algorithm
                var options = new SdcaRegressionTrainer.Options
                {
                    LabelColumnName = "Label",
                    FeatureColumnName = "Features",

                    // Make the convergence tolerance tighter. It effectively leads to
                    // more training iterations.
                    ConvergenceTolerance = 0.5f, // underwent several experiments, 0.5, 0.6, and 0.7 give us best (R^2)

                    // Increase the maximum number of passes over training data. Similar
                    // to ConvergenceTolerance, this value specifics the hard iteration
                    // limit on the training algorithm.
                    MaximumNumberOfIterations = 100000, // underwent several experiments, No improvement after 100000

                    // Increase learning rate for bias.
                    BiasLearningRate = 0.7f,
                };
                // End options

                /* Data Preparation for trainer algorithm */
                // to predict visists as output label
                var pipeline = ml.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "visits_num")

                // transform the categorical data into numbers - bcs the algorithm require numeric features
                // -- (Customize and edit the model or it can be called as *data preparation*)
                .Append(ml.Transforms.Categorical.OneHotEncoding(outputColumnName: "place_id_encoded", inputColumnName: "place_id"))
                .Append(ml.Transforms.Categorical.OneHotEncoding(outputColumnName: "date_encoded", inputColumnName: "date"))

                //Combine each feature into the features column..
                .Append(ml.Transforms.Concatenate("Features", "place_id_encoded", "date_encoded", "time"))

                // Select learning Algorithm..
                .Append(ml.Regression.Trainers.Sdca(options));

                // Fit the data and train the model
                return pipeline.Fit(d.TrainSet);
            }

            // Test the Model
            public RegressionMetrics Evaluate(MLContext ml, ITransformer model, DataOperationsCatalog.TrainTestData d)
            {
                // transform test data
                // makes predictions for the testSet
                var predictions = model.Transform(d.TestSet);

                //computes the quality metrics for the PredictionModel.
                return ml.Regression.Evaluate(predictions, "Label", "Score");
            }

            // Display the model quality
            // the quality effected by many factors
            // our model affected negatively from our dataset 
            // 0.37 - 0.4, (Under-Fitting)
            public void display(RegressionMetrics metrics)
            {
                //produce the evaluation metrics
                Console.WriteLine();
                Console.WriteLine($"*************************************************");
                Console.WriteLine($"*       Model quality metrics evaluation         ");
                Console.WriteLine($"*------------------------------------------------");

                //RSquared takes values between 0 and 1. The closer its value is to 1, the better the model is
                //    R-Squared is good at facilitating comparisons between models. ... Generally, an R-Squared above 0.6 makes a model worth your attention
                Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");


                //RMS The lower it is, the better the model is
                //RMSE is a good measure of accuracy, but only to compare prediction errors of different models or model configurations
                Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:0.##}");
            }

            // Function to test one sample, that should be used in the applcation
            public void TestSinglePrediction(MLContext ml, string modelPath, string[] givenData)
            {
                // load Model
                DataViewSchema modelSchema;
                ITransformer model = ml.Model.Load(modelPath, out modelSchema);

                // Engine to prediction process
                var predictionFunction = ml.Model.CreatePredictionEngine<DataSet.LinearRegression, visitsDataset_Predictions>(model);

                // set a scenario to pedict it
                var visitSample = new DataSet.LinearRegression()
                {
                    place_id = givenData[0],
                    date = givenData[1],
                    time = int.Parse(givenData[2])
                };

                // predict the rate based on single instance
                var prediction = predictionFunction.Predict(visitSample);

                // display
                Console.WriteLine($"**********************************************************************");
                Console.WriteLine($"Visitors Rate: {prediction.visits_num}");
                Console.WriteLine($"**********************************************************************");

            }

        }
    }
}
