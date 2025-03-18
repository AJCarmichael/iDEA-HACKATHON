import React from "react";
import Navbar from "../components/Navbar";

const transactions = [
  { id: 1, accountId: "ACC12345", dateTime: "2024-03-15 14:23", description: "Online Purchase", amount: 5000, flagged: false, reason: "" },
  { id: 2, accountId: "ACC67890", dateTime: "2024-03-16 09:45", description: "Wire Transfer", amount: 150000, flagged: true, reason: "Large international transfer from unknown source" },
  { id: 3, accountId: "ACC11223", dateTime: "2024-03-16 18:12", description: "Bill Payment", amount: 250, flagged: false, reason: "" },
  { id: 4, accountId: "ACC33445", dateTime: "2024-03-17 10:30", description: "International Transfer", amount: 99999, flagged: true, reason: "Smurfing pattern detected (multiple high-value transactions)" },
  { id: 5, accountId: "ACC55667", dateTime: "2024-03-18 13:05", description: "ATM Withdrawal", amount: 12000, flagged: true, reason: "Unusual high amount compared to past transactions" }
];

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-gradient-to-r from-indigo-400 to-purple-600 text-white font-poppins">
      <Navbar />
      <div className="pt-28 flex flex-col items-center text-center px-5">
        <h1 className="text-5xl font-bold mb-5">Dashboard</h1>
        <h2 className="text-2xl mb-10">Transaction Monitoring</h2>

        {/* Transactions Table */}
        <div className="w-full max-w-5xl overflow-x-auto">
          <table className="w-full border-collapse border border-gray-300 text-black bg-white rounded-lg overflow-hidden">
            <thead className="bg-gray-200">
              <tr>
                <th className="border border-gray-300 p-3 text-lg">Transaction ID</th>
                <th className="border border-gray-300 p-3 text-lg">Account No. ID</th>
                <th className="border border-gray-300 p-3 text-lg">Date / Time</th>
                <th className="border border-gray-300 p-3 text-lg">Description</th>
                <th className="border border-gray-300 p-3 text-lg">Amount ($)</th>
                <th className="border border-gray-300 p-3 text-lg">Fraud Status</th>
                <th className="border border-gray-300 p-3 text-lg">Reason</th>
              </tr>
            </thead>
            <tbody>
              {transactions.map((tx) => (
                <tr key={tx.id} className="text-center">
                  <td className="border border-gray-300 p-3">{tx.id}</td>
                  <td className="border border-gray-300 p-3">{tx.accountId}</td>
                  <td className="border border-gray-300 p-3">{tx.dateTime}</td>
                  <td className="border border-gray-300 p-3">{tx.description}</td>
                  <td className="border border-gray-300 p-3">${tx.amount.toLocaleString()}</td>
                  <td
                    className={`border border-gray-300 p-3 font-semibold ${
                      tx.flagged ? "text-red-600" : "text-green-600"
                    }`}
                  >
                    {tx.flagged ? "⚠️ Fraudulent" : "✅ Safe"}
                  </td>
                  <td className="border border-gray-300 p-3">{tx.flagged ? tx.reason : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
