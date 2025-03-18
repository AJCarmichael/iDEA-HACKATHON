import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FaExclamationTriangle, FaCheckCircle, FaClock } from 'react-icons/fa';
import Papa from 'papaparse';
import { useNavigate } from 'react-router-dom';
import Navbar from './Navbar';
import TransactionDetails from './TransactionDetails';

const FraudDetection = () => {
  const [transactions, setTransactions] = useState([]);
  const [results, setResults] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const navigate = useNavigate();

  // Handle CSV file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        complete: (result) => {
          setTransactions(result.data); // Set the parsed transactions
          setCurrentIndex(0); // Start from the first transaction
        },
      });
    }
  };

  // Process transactions one by one
  const processTransactions = () => {
    if (currentIndex < transactions.length) {
      const transaction = transactions[currentIndex];

      // Determine status based on the `is_fraud` column
      const status = transaction.is_fraud === 1 ? 'Flagged' : 'Safe';

      // Add the result to the results list
      setResults((prev) => [
        ...prev,
        {
          ...transaction,
          status: status,
          confidence: status === 'Flagged' ? 95 : 99, // Confidence can be hardcoded or dynamic
        },
      ]);

      // Move to the next transaction
      setCurrentIndex((prev) => prev + 1);
    }
  };

  // Automatically process transactions every 2 seconds
  React.useEffect(() => {
    const interval = setInterval(() => {
      if (currentIndex < transactions.length) {
        processTransactions();
      } else {
        clearInterval(interval); // Stop when all transactions are processed
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [currentIndex, transactions]);

  return (
    <section className="text-gray-900 py-12 bg-gray-100 px-4 sm:px-6 lg:px-8">
      <Navbar />
      <div className="max-w-7xl mx-auto mt-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-10"
        >
          <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-blue-900">
            Fraud Detection with GenAI
          </h2>
          <p className="mt-4 text-lg text-gray-600">
            Upload a CSV file to analyze transactions in real-time.
          </p>
        </motion.div>

        {/* File Upload */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-8"
        >
          <input
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            className="block w-full text-sm text-gray-600 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-600"
          />
        </motion.div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Live Transaction Feed */}
          <div className="lg:col-span-2 bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <FaClock className="mr-2 text-blue-400" /> Live Transaction Feed
            </h3>
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {results.map((transaction, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.6 + index * 0.1 }}
                  className={`p-4 rounded-lg flex justify-between items-center ${
                    transaction.status === 'Flagged' ? 'bg-red-100' : 'bg-green-100'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    {transaction.status === 'Flagged' ? (
                      <FaExclamationTriangle className="text-red-400" />
                    ) : (
                      <FaCheckCircle className="text-green-400" />
                    )}
                    <div>
                      <p className="text-sm font-medium">{transaction.Description}</p>
                      <p className="text-xs text-gray-500">
                        {transaction.Date} {transaction.Time}
                      </p>
                      {transaction.status === 'Flagged' && (
                        <p className="text-xs text-red-500">Reason: Potential Fraud</p>
                      )}
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-semibold">₹{transaction['Amount (INR)'].toLocaleString('en-IN')}</p>
                    <p className="text-xs text-gray-500">
                      Confidence: {transaction.confidence}%
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* GenAI Anomaly Detection Demo */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">GenAI Anomaly Detection</h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <p className="text-sm">Fraud Detection Accuracy</p>
                <p className="text-sm font-semibold text-green-500">98%</p>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-green-500 h-2.5 rounded-full"
                  style={{ width: '98%' }}
                ></div>
              </div>

              <div className="flex justify-between items-center">
                <p className="text-sm">False Positive Rate</p>
                <p className="text-sm font-semibold text-red-500">5%</p>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-red-500 h-2.5 rounded-full"
                  style={{ width: '5%' }}
                ></div>
              </div>

              <div className="mt-6">
                <h4 className="text-sm font-medium mb-2">Latest Flagged Transaction</h4>
                {results.find((t) => t.status === 'Flagged') ? (
                  <div className="p-4 bg-red-100 rounded-lg">
                    <p className="text-sm">
                      {results.find((t) => t.status === 'Flagged').Description}
                    </p>
                    <p className="text-xs text-gray-500">
                      {results.find((t) => t.status === 'Flagged').Date}{' '}
                      {results.find((t) => t.status === 'Flagged').Time}
                    </p>
                    <p className="text-xs text-red-500">Reason: Potential Fraud</p>
                    <p className="text-xs text-gray-500">
                      Confidence: {results.find((t) => t.status === 'Flagged').confidence}%
                    </p>
                    <button
                      onClick={() =>
                        navigate('/transactiondetails', {
                          state: { transaction: results.find((t) => t.status === 'Flagged') },
                        })
                      }
                      className="mt-2 px-3 py-1 text-xs bg-blue-500 text-white rounded-lg hover:bg-blue-600"
                    >
                      Know More
                    </button>
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">No flagged transactions yet.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default FraudDetection;