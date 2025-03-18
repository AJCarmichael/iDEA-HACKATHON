import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './components/HomePage';
import Recommendations from './components/Recommendations';
import Dashboard from './components/Dashboard';
import Login from './components/Login'; // Import Login Component
import Register from './components/Register'; // Import Register Component
import SecureInternetBanking from './components/SecureInternetBanking';
// import MobileBanking from './components/MobileBanking';
import Loans from './components/Loans';
import SavingsAndDeposits from './components/SavingsAndDeposits';
import FraudDetection from './components/FraudDetection';
import LiveFeed from './components/LiveFeed';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/recommendations" element={<Recommendations />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/secureinternetbanking" element={<SecureInternetBanking />} />
        {/* <Route path="/mobilebanking" element={<MobileBanking />} /> */}
        <Route path="/loans" element={<Loans />} />
        <Route path="/savingsanddeposits" element={<SavingsAndDeposits />} />
        <Route path="/frauddetection" element={<FraudDetection />} />
        <Route path="/livefeed" element={<LiveFeed />} />
      </Routes>
    </Router>
  );
}

export default App;
