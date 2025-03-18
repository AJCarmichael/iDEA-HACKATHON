import React from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex justify-between items-center p-1 bg-black bg-opacity-20 font-poppins">
      <img src="/logo1.png" alt="Logo 1" className="h-14 mx-5" />
      <div className="flex gap-5">
        <Link to="/" className="text-white text-lg hover:underline">
          Home
        </Link>
        <Link to="/recommendations" className="text-white text-lg hover:underline">
          Recommendations
        </Link>
        <Link to="/frauddetection" className="text-white text-lg hover:underline">
          Dashboard
        </Link>
        <Link to="/register" className="text-white text-lg hover:underline">
          Register
        </Link>
        <Link to="/login" className="text-white text-lg hover:underline">
          Login
        </Link>
      </div>
      <img src="/logo2.png" alt="Logo 2" className="h-10 mx-5" />
    </nav>
  );
};

export default Navbar;
