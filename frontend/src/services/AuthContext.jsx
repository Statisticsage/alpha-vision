import { createContext, useContext, useState, useEffect } from "react";
import { login as apiLogin, signup as apiSignup } from "./api";

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem("token") || "");

  useEffect(() => {
    if (token) {
      localStorage.setItem("token", token);
    } else {
      localStorage.removeItem("token");
    }
  }, [token]);

  const login = async (email, password) => {
    try {
      const response = await apiLogin(email, password);
      const accessToken = response.data.access_token;

      setToken(accessToken);
      setUser({ email });

      return { success: true };
    } catch (error) {
      const errorMessage = error.response?.data?.detail || "Invalid email or password.";
      return { success: false, message: errorMessage };
    }
  };

  const signup = async (email, password) => {
    try {
      await apiSignup(email, password);
      return { success: true };
    } catch (error) {
      const errorMessage = error.response?.data?.detail || "Signup failed. Please try again.";
      return { success: false, message: errorMessage };
    }
  };

  const logout = () => {
    setToken("");
    setUser(null);
    localStorage.removeItem("token");
    window.location.href = "/login"; // ✅ Ensure redirection on logout
  };
  
  return (
    <AuthContext.Provider value={{ user, token, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
