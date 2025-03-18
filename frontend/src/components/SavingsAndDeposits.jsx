import React from 'react';
import { motion } from 'framer-motion';
import Navbar from './Navbar';  // ✅ Import Navbar from your file

// Savings and Deposit Accounts Data
const accounts = [
  {
    id: 1,
    name: 'Savings Accounts',
    description: 'Regular savings accounts with interest earnings and easy access to funds.',
  },
  {
    id: 2,
    name: 'Fixed Deposits (FDs)',
    description: 'High-interest fixed-term deposits with flexible tenures.',
  },
  {
    id: 3,
    name: 'Recurring Deposits (RDs)',
    description: 'Regular savings with fixed monthly deposits and interest earnings.',
  },
  {
    id: 4,
    name: 'Current Accounts',
    description: 'For businesses and entrepreneurs to manage daily transactions.',
  },
  {
    id: 5,
    name: 'Senior Citizen Accounts',
    description: 'Special accounts with higher interest rates for senior citizens.',
  },
];

// Savings and Deposit Accounts Page Component
const SavingsAndDeposits = () => {
  return (
    <>
      {/* Navbar */}
      <Navbar />

      {/* Main Content */}
      <div className="min-h-screen bg-gray-100 text-center p-6 font-poppins mt-16">
        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-4xl font-bold text-blue-900 mb-4"
        >
          Savings and Deposit Accounts
        </motion.h1>

        {/* Subtitle */}
        <motion.h2
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-xl text-gray-600 mb-16"
        >
          Grow your savings with our range of deposit accounts
        </motion.h2>

        {/* Accounts Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {accounts.map((account) => (
            <motion.div
              key={account.id}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 + account.id * 0.1 }}
              className="bg-white p-6 rounded-lg shadow-lg hover:shadow-xl hover:cursor-pointer transition-shadow duration-300"
            >
              <h3 className="text-xl font-semibold text-blue-900 mb-2">{account.name}</h3>
              <p className="text-gray-600">{account.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </>
  );
};

export default SavingsAndDeposits;