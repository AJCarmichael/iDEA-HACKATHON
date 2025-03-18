import React, { useState, useEffect } from 'react';
import { FaExclamationTriangle, FaCheckCircle, FaClock } from 'react-icons/fa';

const SecureInternetBanking = () => {
  // Simulated live transaction data
  const initialTransactions = [
    { id: 1, description: 'Transfer to John Doe', amount: '$500', status: 'Safe', timestamp: '10:30 AM', confidence: 98 },
    { id: 2, description: 'Cross-Border Transfer to Country X', amount: '$10,000', status: 'Flagged', timestamp: '10:32 AM', confidence: 92, reason: 'Potential Smurfing' },
    { id: 3, description: 'Bill Payment - Electricity', amount: '$150', status: 'Safe', timestamp: '10:35 AM', confidence: 99 },
    { id: 4, description: 'Transfer to Unknown Account', amount: '$8,000', status: 'Flagged', timestamp: '10:40 AM', confidence: 95, reason: 'Unusual Cross-Border Activity' },
  ];

  const [transactions, setTransactions] = useState(initialTransactions);

  // Simulate new transactions being added dynamically
 useEffect(() => {
  const interval = setInterval(() => {
    setTransactions((prev) => {
      const newTransaction = {
        id: prev.length + 1,
        description: `Transfer to User ${prev.length + 1}`,
        amount: `$${Math.floor(Math.random() * 10000)}`,
        status: Math.random() > 0.7 ? "Flagged" : "Safe",
        timestamp: new Date().toLocaleTimeString(),
        confidence: Math.floor(Math.random() * (100 - 80) + 80),
        reason: Math.random() > 0.7 ? "Potential Smurfing" : "Unusual Activity",
      };
      return [newTransaction, ...prev.slice(0, 4)];
    });
  }, 5000);

  return () => clearInterval(interval);
}, []); // ✅ Empty dependency array prevents unnecessary resets


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

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Live Transaction Feed */}
          <div className="lg:col-span-2 bg-gray-800 rounded-lg shadow-lg p-6">
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
                    <p className="text-sm font-semibold">{transaction.amount}</p>
                    <p className="text-xs text-gray-400">
                      Confidence: {transaction.confidence}%
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* GenAI Anomaly Detection Demo */}
          <div className="bg-gray-800 rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">GenAI Anomaly Detection</h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <p className="text-sm">Fraud Detection Accuracy</p>
                <p className="text-sm font-semibold text-green-400">98%</p>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2.5">
                <div
                  className="bg-green-500 h-2.5 rounded-full"
                  style={{ width: '98%' }}
                ></div>
              </div>

              <div className="flex justify-between items-center">
                <p className="text-sm">False Positive Rate</p>
                <p className="text-sm font-semibold text-red-400">5%</p>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2.5">
                <div
                  className="bg-red-500 h-2.5 rounded-full"
                  style={{ width: '5%' }}
                ></div>
              </div>

              <div className="mt-6">
                <h4 className="text-sm font-medium mb-2">Latest Flagged Transaction</h4>
                {transactions.find((t) => t.status === 'Flagged') ? (
                  <div className="p-4 bg-red-900/50 rounded-lg">
                    <p className="text-sm">
                      {transactions.find((t) => t.status === 'Flagged').description}
                    </p>
                    <p className="text-xs text-gray-400">
                      {transactions.find((t) => t.status === 'Flagged').timestamp}
                    </p>
                    <p className="text-xs text-red-300">
                      Reason: {transactions.find((t) => t.status === 'Flagged').reason}
                    </p>
                    <p className="text-xs text-gray-400">
                      Confidence: {transactions.find((t) => t.status === 'Flagged').confidence}%
                    </p>
                  </div>
                ) : (
                  <p className="text-sm text-gray-400">No flagged transactions yet.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default SecureInternetBanking;