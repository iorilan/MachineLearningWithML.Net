using System;

namespace _03_SG_HDB_Price
{
    class Program
    {
        static void Main(string[] args)
        {
			var prediction = Predictor.Predict(new Predictor.PriceData()
			{
				quarter = "2030-Q4",
				index = 0// predict it
			}).Result;

	        Console.WriteLine("Predicted SG hdb house price index: {0}", prediction.index);

	        Console.ReadLine();
        }
	}
}
