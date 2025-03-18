import React, { useState } from 'react';
import { motion } from 'framer-motion'; // Import Framer Motion
import { FaExclamationTriangle, FaCheckCircle } from 'react-icons/fa';
import Papa from 'papaparse';  // For parsing CSV files
import Navbar from './Navbar';

const FraudDetection = () => {
  const [transactions, setTransactions] = useState([]);
  const [results, setResults] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  // Handle CSV file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        complete: (result) => {
          setTransactions(result.data);  // Set the parsed transactions
          setCurrentIndex(0);  // Start from the first transaction
        },
      });
    }
  };

  // Check if a transaction is fraudulent
  const checkTransaction = async (transaction) => {
    try {
      const response = await fetch('http://192.168.1.100:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(transaction),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      return data;  // { is_fraud: boolean, confidence: number }
    } catch (error) {
      console.error('Error checking transaction:', error);
      return { is_fraud: false, confidence: 99 };  // Fallback response
    }
  };

  // Process transactions one by one
  const processTransactions = async () => {
    if (currentIndex < transactions.length) {
      const transaction = transactions[currentIndex];
      const result = await checkTransaction(transaction);

      // Add the result to the results list
      setResults((prev) => [
        ...prev,
        { ...transaction, status: result.is_fraud ? 'Flagged' : 'Safe', confidence: result.confidence },
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
        clearInterval(interval);  // Stop when all transactions are processed
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

        {/* Results Table */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="bg-gray-50 rounded-lg shadow-lg p-6"
        >
          <h3 className="text-xl font-semibold mb-4 text-blue-900">Transaction Analysis</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                    Description
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                    Amount
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                    Confidence
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {results.map((result, index) => (
                  <motion.tr
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.6 + index * 0.1 }}
                  >
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {result.description}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      ${result.amount}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {result.status === 'Flagged' ? (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                          <FaExclamationTriangle className="mr-1" /> Flagged
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                          <FaCheckCircle className="mr-1" /> Safe
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      {result.confidence}%
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default FraudDetection;