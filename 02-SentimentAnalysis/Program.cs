using System;

namespace _02_SentimentAnalysis
{
	class Program
	{
		static void Main(string[] args)
		{
			var ret = SentimentAnalysisExecutor.Run(new[]
			{
				new SentimentAnalysisExecutor.SentimentData
				{
					SentimentText = "Please refrain from adding nonsense to Wikipedia."
				},
				new SentimentAnalysisExecutor.SentimentData
				{
					SentimentText = "He is the best, and the article should say that."
				},
				new SentimentAnalysisExecutor.SentimentData
				{
					SentimentText = "I'm not sure If that is correct."
				},
			}).Result;

			foreach (var sentimentPrediction in ret)
			{
				Console.WriteLine(sentimentPrediction);
			}



			Console.ReadLine();

		}
	}
}
