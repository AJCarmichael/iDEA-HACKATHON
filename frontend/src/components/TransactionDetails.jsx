import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import Navbar from './Navbar';

const TransactionDetails = () => {
  const location = useLocation();
  const { transaction } = location.state || {};

  if (!transaction) {
    return (
        
      <motion.div
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="p-6 text-center"
      >
        <p className="text-red-500">No transaction details found.</p>
        <Link
          to="/"
          className="mt-4 inline-block px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
        >
          Go back to the dashboard
        </Link>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
      className="p-6 bg-white rounded-lg shadow-lg max-w-2xl mx-auto mt-16"
    ><Navbar />
      <h2 className="text-2xl font-semibold mb-4 text-blue-900">Transaction Details</h2>
      <div className="space-y-4">
        <p><strong className="text-gray-700">Description:</strong> {transaction.Description}</p>
        <p><strong className="text-gray-700">Amount:</strong> ₹{transaction['Amount (INR)'].toLocaleString('en-IN')}</p>
        <p><strong className="text-gray-700">Date:</strong> {transaction.Date}</p>
        <p><strong className="text-gray-700">Time:</strong> {transaction.Time}</p>
        <p><strong className="text-gray-700">Recipient Account:</strong> {transaction['Recipient Account']}</p>
        <p><strong className="text-gray-700">Recipient Bank:</strong> {transaction['Recipient Bank']}</p>
        <p><strong className="text-gray-700">Recipient Country:</strong> {transaction['Recipient Country']}</p>
        <p><strong className="text-gray-700">Status:</strong> {transaction.status}</p>
        <p><strong className="text-gray-700">Confidence:</strong> {transaction.confidence}%</p>
        <p><strong className="text-gray-700">Reason:</strong> Potential Fraud</p>
      </div>
      <Link
        to="/"
        className="mt-6 inline-block px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
      >
        Go back to the dashboard
      </Link>
    </motion.div>
  );
};

export default TransactionDetails;