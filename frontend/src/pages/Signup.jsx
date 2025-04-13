import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "../styles/auth.css"; // Import CSS file

const Signup = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSignup = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    // 🔍 Validate password length before sending request
    if (password.length < 8) {
      setError("⚠️ Password must be at least 8 characters long.");
      setLoading(false);
      return;
    }

    try {
      const response = await axios.post(
        "http://localhost:8000/auth/signup",
        { email, password },
        { headers: { "Content-Type": "application/json" } }
      );

      console.log("✅ Signup success:", response.data);

      // 🔄 Redirect to login after successful signup
      navigate("/login", { replace: true });
    } catch (error) {
      console.error("❌ Signup failed:", error.response?.data || error.message);

      if (!error.response) {
        setError("⚠️ Network error. Please check your connection.");
      } else if (error.response.status === 400) {
        setError("❌ Email is already registered.");
      } else {
        setError("⚠️ Signup failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-box">
        <h2>Create an Account</h2>
        {error && <p style={{ color: "red", fontWeight: "bold" }}>{error}</p>}
        <form onSubmit={handleSignup}>
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Password (min. 8 chars)"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <button type="submit" disabled={loading}>
            {loading ? "Signing up..." : "Sign Up"}
          </button>
        </form>
        <p>
          Already have an account? <a href="/login">Login here</a>
        </p>
      </div>
    </div>
  );
};

export default Signup;
