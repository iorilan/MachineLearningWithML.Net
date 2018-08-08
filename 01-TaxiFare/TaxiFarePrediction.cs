using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace _01_TaxiFare
{
	public class TaxiFarePrediction
	{
		static readonly string _datapath = Path.Combine(Environment.CurrentDirectory,  "taxi-fare-train.csv");
		static readonly string _testdatapath = Path.Combine(Environment.CurrentDirectory,  "taxi-fare-test.csv");
		static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Model.zip");

		public static async Task<TaxiTripFarePrediction> Predict(TaxiTrip tt)
		{
			var model = await Train();
			Evaluate(model);

			return model.Predict(tt);
		}

		private static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
		{
			
			var pipeline = new LearningPipeline
			{
				new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','),
				new ColumnCopier(("FareAmount", "Label")),
				new CategoricalOneHotVectorizer(
					"VendorId",
					"RateCode",
					"PaymentType"),
				new ColumnConcatenator(
					"Features",
					"VendorId",
					"RateCode",
					"PassengerCount",
					"TripDistance",
					"PaymentType"),
				new FastTreeRegressor()
			};
			PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
			await model.WriteAsync(_modelpath);
			return model;
		}
		private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
		{
			var testData = new TextLoader(_testdatapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');
			var evaluator = new RegressionEvaluator();
			RegressionMetrics metrics = evaluator.Evaluate(model, testData);
			Console.WriteLine($"Rms = {metrics.Rms}");
			Console.WriteLine($"RSquared = {metrics.RSquared}");
		}
	}
}
