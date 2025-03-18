import React, { useState, useEffect } from 'react';
import { FaExclamationTriangle, FaCheckCircle, FaClock } from 'react-icons/fa';

const LiveFeed = () => {
  const [transactions, setTransactions] = useState([]);

  // Simulate new transactions being added dynamically
  useEffect(() => {
    const interval = setInterval(async () => {
      // Simulate a new transaction
      const newTransaction = {
        id: transactions.length + 1,
        description: `Transfer to User ${transactions.length + 1}`,
        amount: Math.floor(Math.random() * 10000),
        location: Math.random() > 0.5 ? 'Domestic' : 'International',
        timestamp: new Date().toLocaleTimeString(),
      };

      // Send the transaction to the Dockerized model for prediction
      try {
        const response = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            amount: newTransaction.amount,
            location: newTransaction.location,
            time: newTransaction.timestamp,
          }),
        });

        const result = await response.json();
        newTransaction.status = result.is_fraud ? 'Flagged' : 'Safe';
        newTransaction.confidence = result.confidence;
        newTransaction.reason = result.is_fraud ? 'Potential Fraud' : 'Safe';
      } catch (error) {
        console.error('Error predicting fraud:', error);
        newTransaction.status = 'Safe';  // Fallback to safe if the API fails
        newTransaction.confidence = 99;
      }

      // Add the new transaction to the list
      setTransactions((prev) => [newTransaction, ...prev.slice(0, 4)]);
    }, 2000);  // Add a new transaction every 2 seconds

    return () => clearInterval(interval);
  }, [transactions]);

  return (
    <section className="bg-gray-900 text-white py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Section Header */}
        <div className="text-center mb-10">
          <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight">
            Secure Internet Banking with GenAI
          </h2>
          <p className="mt-4 text-lg text-gray-400">
            Monitor transactions in real-time, detect anomalies, and reduce false positives with Generative AI-powered fraud detection.
          </p>
        </div>

        {/* Live Transaction Feed */}
        <div className="bg-gray-800 rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center">
            <FaClock className="mr-2 text-blue-400" /> Live Transaction Feed
          </h3>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {transactions.map((transaction) => (
              <div
                key={transaction.id}
                className={`p-4 rounded-lg flex justify-between items-center ${
                  transaction.status === 'Flagged' ? 'bg-red-900/50' : 'bg-green-900/50'
                }`}
              >
                <div className="flex items-center space-x-3">
                  {transaction.status === 'Flagged' ? (
                    <FaExclamationTriangle className="text-red-400" />
                  ) : (
                    <FaCheckCircle className="text-green-400" />
                  )}
                  <div>
                    <p className="text-sm font-medium">{transaction.description}</p>
                    <p className="text-xs text-gray-400">{transaction.timestamp}</p>
                    {transaction.status === 'Flagged' && (
                      <p className="text-xs text-red-300">Reason: {transaction.reason}</p>
                    )}
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-semibold">${transaction.amount}</p>
                  <p className="text-xs text-gray-400">
                    Confidence: {transaction.confidence}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default LiveFeed;