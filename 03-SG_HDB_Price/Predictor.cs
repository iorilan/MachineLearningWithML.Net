using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace _03_SG_HDB_Price
{
   public  static class Predictor
    {
		static readonly string _datapath = Path.Combine(Environment.CurrentDirectory, "train.csv");
		static readonly string _testdatapath = Path.Combine(Environment.CurrentDirectory, "test.csv");
		static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Model.zip");

		public class PriceData
		{
			[Column(ordinal: "0", name: "quarter")]
			public string quarter;

			[Column(ordinal: "1", name: "index")]
			public float index;
		}

		public class PricePrediction
		{
			[ColumnName("Score")]
			public float index;
		}

	    public static async Task<PricePrediction> Predict(PriceData tt)
	    {
		    var model = await Train();
		    Evaluate(model);

		    return model.Predict(tt);
	    }

	    private static async Task<PredictionModel<PriceData, PricePrediction>> Train()
	    {

		    var pipeline = new LearningPipeline
		    {
			    new TextLoader(_datapath).CreateFrom<PriceData>(useHeader: true, separator: ','),
			    new ColumnCopier(("index", "Label")),
			    new CategoricalOneHotVectorizer(
				    "quarter"),
				new ColumnConcatenator("Features","quarter","index"),
			    new FastTreeRegressor()
		    };
		    PredictionModel<PriceData, PricePrediction> model = pipeline.Train<PriceData, PricePrediction>();
		    await model.WriteAsync(_modelpath);
		    return model;
	    }
	    private static void Evaluate(PredictionModel<PriceData, PricePrediction> model)
	    {
		    var testData = new TextLoader(_testdatapath).CreateFrom<PriceData>(useHeader: true, separator: ',');
		    var evaluator = new RegressionEvaluator();
		    RegressionMetrics metrics = evaluator.Evaluate(model, testData);
		    Console.WriteLine($"Rms = {metrics.Rms}");
		    Console.WriteLine($"RSquared = {metrics.RSquared}");
	    }

	}
}
