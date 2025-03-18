import React, { useState } from "react";
import { motion } from "framer-motion"; // Import Framer Motion
import Navbar from "../components/Navbar";

const Recommendations = () => {
  const [selectedFile, setSelectedFile] = useState(null);
<<<<<<< Updated upstream
  const [recommendedServices, setRecommendedServices] = useState([]); // Updated state to store both numbers and names
=======
  const [recommendedServiceNumbers, setRecommendedServiceNumbers] = useState([]);
>>>>>>> Stashed changes
  const [topTransactions, setTopTransactions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showRecommendations, setShowRecommendations] = useState(false);
  const [customPrompt, setCustomPrompt] = useState(""); // New state for custom prompt

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
<<<<<<< Updated upstream
      setRecommendedServices([]); // Updated state
=======
      setRecommendedServiceNumbers([]);
>>>>>>> Stashed changes
      setError("");
      setShowRecommendations(false);

      // Read the CSV file to extract top 5 transactions
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target.result;
        const rows = text.split("\n").slice(0, 6); // Get the first 5 rows (including header)
        setTopTransactions(rows);
      };
      reader.readAsText(file);
    }
  };

  const handleViewRecommendations = async () => {
    if (!selectedFile) {
      setError("Please select a CSV file first.");
      return;
    }

    setLoading(true);
    setError("");

    // Create a FormData object to send the file to the backend.
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("custom_prompt", customPrompt); // Append custom prompt

    try {
      const response = await fetch("http://127.0.0.1:5000/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        setError(`Error processing data: ${errData.error || "Unknown error"}`);
        setLoading(false);
        return;
      }

      const resData = await response.json();
      setRecommendedServices(resData.recommended_services.map((num, index) => ({
        number: num,
        name: resData.service_names[index]
      })));
      setShowRecommendations(true);
    } catch (err) {
      setError(`An error occurred while processing the data: ${err.message}`);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-r bg-gray-100 text-gray-900 font-poppins">
      <Navbar />
      <div className="pt-20 flex flex-col items-center text-center px-5">
        {/* Title Section */}
        <motion.h1
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-3xl sm:text-4xl font-extrabold tracking-tight text-blue-900"
        >
          Bank Service Recommender
        </motion.h1>
        <motion.h2
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="m-4 text-lg text-gray-600"
        >
          Upload your transaction file to get suggestions!
        </motion.h2>

        {/* Main Content */}
         {/* File Upload and Button */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.2 }}
          className="w-full max-w-2xl bg-white rounded-xl shadow-lg p-8 mb-10"
        >
          <div className="mb-6">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
<<<<<<< Updated upstream
          <div className="mb-6">
            <textarea
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
              placeholder="Enter your custom request here..."
              className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
=======
>>>>>>> Stashed changes
          <button
            onClick={handleViewRecommendations}
            disabled={loading}
            className="w-full px-5 py-3 bg-indigo-600 text-white font-bold rounded-lg shadow-md hover:bg-indigo-700 transition duration-300 disabled:bg-indigo-400"
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <svg
                  className="animate-spin h-5 w-5 mr-3 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Processing...
              </div>
            ) : (
              "Get Recommendations"
            )}
          </button>
          {error && (
            <p className="mt-4 text-sm text-red-600 bg-red-50 p-2 rounded-lg">
              {error}
            </p>
          )}
        </motion.div>
        <div className="w-full max-w-7xl flex flex-col lg:flex-row gap-8">
          
          {/* Left Side: Top 5 Transactions Table */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="w-full lg:w-1/2 bg-white rounded-xl shadow-lg p-8 max-h-fit
"
          >
            <h3 className="text-xl font-semibold mb-4 text-blue-900">
              Top 5 Transactions
            </h3>
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead>
                  <tr>
                    {topTransactions[0] &&
                      topTransactions[0]
                        .split(",")
                        .map((header, index) => (
                          <th
                            key={index}
                            className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                          >
                            {header}
                          </th>
                        ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {topTransactions.slice(1).map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      {row.split(",").map((cell, cellIndex) => (
                        <td
                          key={cellIndex}
                          className="px-6 py-4 whitespace-nowrap text-sm text-gray-700"
                        >
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
<<<<<<< Updated upstream

          {/* Right Side: Recommendations Card and Chatbot */}
          <div className="w-full lg:w-1/2 flex flex-col gap-8">
            {/* Recommendations Card */}
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.6 }}
              className="bg-white rounded-xl shadow-lg p-8"
            >
              <h3 className="text-xl font-semibold mb-4 text-blue-900">
                Recommended Services
              </h3>
              <ul className="space-y-3">
                {recommendedServices.map((service, index) => (
                  <motion.li
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.8 + index * 0.1 }}
                    className="p-4 bg-gray-50 rounded-lg text-gray-700 font-medium"
                  >
                    Service #{service.number}: {service.name}
                  </motion.li>
                ))}
              </ul>
            </motion.div>

            {/* Chatbot */}
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 1 }}
              className="bg-white rounded-xl shadow-lg p-8"
            >
              <h3 className="text-xl font-semibold mb-4 text-blue-900">
                Chatbot Assistance
              </h3>
              <div className="h-64 bg-gray-50 rounded-lg p-4">
                <p className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                  Ask me anything about the recommendations!
                </p>
                <textarea
                  value={customPrompt}
                  onChange={(e) => setCustomPrompt(e.target.value)}
                  placeholder="Enter your custom request here..."
                  className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </motion.div>
          </div>
        </div>
=======

          {/* Right Side: Recommendations Card and Chatbot */}
          <div className="w-full lg:w-1/2 flex flex-col gap-8">
            {/* Recommendations Card */}
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.6 }}
              className="bg-white rounded-xl shadow-lg p-8"
            >
              <h3 className="text-xl font-semibold mb-4 text-blue-900">
                Recommended Service Numbers
              </h3>
              <ul className="space-y-3">
                {recommendedServiceNumbers.map((service, index) => (
                  <motion.li
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.8 + index * 0.1 }}
                    className="p-4 bg-gray-50 rounded-lg text-gray-700 font-medium"
                  >
                    Service #{service}
                  </motion.li>
                ))}
              </ul>
            </motion.div>

            {/* Chatbot */}
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 1 }}
              className="bg-white rounded-xl shadow-lg p-8"
            >
              <h3 className="text-xl font-semibold mb-4 text-blue-900">
                Chatbot Assistance
              </h3>
              <div className="h-64 bg-gray-50 rounded-lg p-4">
                <p className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                  Ask me anything about the recommendations!
                </p>
                {/* Add your chatbot component or integration here */}
              </div>
            </motion.div>
          </div>
        </div>

       
        
>>>>>>> Stashed changes
      </div>
    </div>
  );
};

export default Recommendations;