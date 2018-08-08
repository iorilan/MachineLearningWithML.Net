using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace _02_SentimentAnalysis
{
	public static class SentimentAnalysisExecutor
	{
		static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "wikipedia-detox-250-line-data.tsv");
		static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "wikipedia-detox-250-line-test.tsv");
		static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Model.zip");

		public class SentimentData
		{
			[Column(ordinal: "0", name: "Label")]
			public float Sentiment;
			[Column(ordinal: "1")]
			public string SentimentText;
		}

		public class SentimentPrediction
		{
			[ColumnName("PredictedLabel")]
			public bool Sentiment;


			public override string ToString()
			{
				return Sentiment ? "Positive" : "Negtive";
			}
		}

		public static async Task<IEnumerable<SentimentPrediction>> Run(IEnumerable<SentimentData> inputs)
		{
			var model = await Train();
			Evoluate(model);

			var results = model.Predict(inputs);
			return results;
		}

		private static void Evoluate(PredictionModel<SentimentData, SentimentPrediction> model)
		{
			var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();
			var evaluator = new BinaryClassificationEvaluator();
			BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);


			Console.WriteLine();
			Console.WriteLine("PredictionModel quality metrics evaluation");
			Console.WriteLine("------------------------------------------");
			Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
			Console.WriteLine($"Auc: {metrics.Auc:P2}");
			Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
		}

		private static async Task<PredictionModel<SentimentData, SentimentPrediction>> Train()
		{
			var pipeline = new LearningPipeline();
			pipeline.Add(new TextLoader(_dataPath).CreateFrom<SentimentData>());
			pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
			pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });
			PredictionModel<SentimentData, SentimentPrediction> model =
				pipeline.Train<SentimentData, SentimentPrediction>();
			await model.WriteAsync(_modelpath);

			return model;
		}


	}
}
