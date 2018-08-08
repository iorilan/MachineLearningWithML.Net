using System;

namespace _01_TaxiFare
{
    class Program
    {
        static void Main(string[] args)
        {
	        var prediction = TaxiFarePrediction.Predict(new TaxiTrip
	        {
		        VendorId = "VTS",
		        RateCode = "1",
		        PassengerCount = 1,
		        TripDistance = 10.33f,
		        PaymentType = "CSH",
		        FareAmount = 0 // predict it. actual = 29.5
	        }).Result;

	        Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.FareAmount);

	        Console.ReadLine();
        }
    }
}
