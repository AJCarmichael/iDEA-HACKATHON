import React from "react";
import { motion } from "framer-motion"; // Import Framer Motion
import Navbar from "./Navbar";

// LoanCard Component with Animation
const LoanCard = ({ title, description, icon, index }) => (
  <motion.div
    initial={{ opacity: 0, y: 50 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay: 0.2 + index * 0.1 }}
    className="bg-white p-6 rounded-lg shadow-lg hover:shadow-xl transition-shadow duration-300"
  >
    <div className="text-4xl mb-4 text-blue-600">{icon}</div>
    <h3 className="text-xl font-bold mb-2">{title}</h3>
    <p className="text-gray-600">{description}</p>
  </motion.div>
);

const Loans = () => {
  const loans = [
    {
      title: "Personal Loans",
      description: "For personal expenses like weddings, travel, or medical emergencies.",
      icon: "💳",
    },
    {
      title: "Home Loans",
      description: "For purchasing, constructing, or renovating a house.",
      icon: "🏠",
    },
    {
      title: "Education Loans",
      description: "For funding higher education in India or abroad.",
      icon: "🎓",
    },
    {
      title: "Vehicle Loans",
      description: "For purchasing cars, two-wheelers, or commercial vehicles.",
      icon: "🚗",
    },
    {
      title: "Agricultural Loans",
      description: "For farmers to support farming activities, equipment purchase, or crop cultivation.",
      icon: "🚜",
    },
    {
      title: "Business Loans",
      description: "For small, medium, and large enterprises to meet working capital or expansion needs.",
      icon: "💼",
    },
  ];

  return (
    <div className="min-h-screen bg-gray-100 py-12">
      <Navbar />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Title Section */}
        <motion.h1
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-4xl font-bold text-center text-blue-900 m-8"
        >
          Loans and Advances
        </motion.h1>

        {/* Subtitle Section */}
        <motion.p
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-center text-gray-600 mb-12"
        >
          Union Bank of India offers a wide range of loan products to cater to your financial needs.
        </motion.p>

        {/* Loan Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {loans.map((loan, index) => (
            <LoanCard
              key={index}
              title={loan.title}
              description={loan.description}
              icon={loan.icon}
              index={index} // Pass index for staggered delay
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default Loans;