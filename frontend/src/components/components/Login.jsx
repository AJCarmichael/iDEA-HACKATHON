// src/components/Login.js
import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Navbar from "./Navbar";

const Login = () => {
    const navigate = useNavigate();
    const [credentials, setCredentials] = useState({ email: "", password: "" });

    const handleChange = (e) => {
        setCredentials({ ...credentials, [e.target.name]: e.target.value });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        const storedUser = JSON.parse(localStorage.getItem("user"));

        if (storedUser && storedUser.email === credentials.email && storedUser.password === credentials.password) {
            localStorage.setItem("loggedIn", "true");
            alert("Login Successful!");
            navigate("/");
        } else {
            alert("Invalid email or password");
        }
    };

    return (
        <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-br from-blue-600 to-purple-700 text-white text-center p-5">
            <Navbar />
            <h2 className="text-4xl font-bold mb-5">Login</h2>
            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                <input
                    type="email"
                    name="email"
                    placeholder="Email"
                    value={credentials.email}
                    onChange={handleChange}
                    required
                    className="p-2 text-black rounded"
                />
                <input
                    type="password"
                    name="password"
                    placeholder="Password"
                    value={credentials.password}
                    onChange={handleChange}
                    required
                    className="p-2 text-black rounded"
                />
                <button type="submit" className="bg-blue-500 px-4 py-2 rounded">
                    Login
                </button>
            </form>
            <p className="mt-4">
                Don't have an account?{" "}
                <Link to="/register" className="underline">
                    Register
                </Link>
            </p>
        </div>
    );
};

export default Login;
