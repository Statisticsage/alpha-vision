import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "../styles/auth.css"; // Import CSS file

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await axios.post(
        "http://localhost:8000/auth/login",
        { email, password },
        { headers: { "Content-Type": "application/json" } }
      );

      console.log("✅ Login successful:", response.data);

      // Store token in localStorage or sessionStorage (more secure)
      localStorage.setItem("token", response.data.access_token);

      // Redirect to home after successful login
      navigate("/home", { replace: true });
    } catch (error) {
      console.error("❌ Login failed:", error.response?.data || error.message);

      // Handle different error cases
      if (!error.response) {
        setError("⚠️ Network error. Please check your connection.");
      } else if (error.response.status === 401) {
        setError("❌ Incorrect email or password. Try again.");
      } else if (error.response.status === 404) {
        setError("❌ User not found. Please check your email or sign up.");
      } else {
        setError("⚠️ Something went wrong. Please try again later.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-box">
        <h2>Welcome Back</h2>
        {error && <p style={{ color: "red", fontWeight: "bold" }}>{error}</p>}
        <form onSubmit={handleLogin}>
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <button type="submit" disabled={loading}>
            {loading ? "Logging in..." : "Login"}
          </button>
        </form>
        <p>
          Don't have an account? <a href="/signup">Sign up</a>
        </p>
      </div>
    </div>
  );
};

export default Login;
