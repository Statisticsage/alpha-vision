import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000"; // Ensure this matches your backend

export const login = async (email, password) => {
  return axios.post(`${API_BASE_URL}/auth/login`, { email, password });
};

export const signup = async (email, password) => {
  return axios.post(`${API_BASE_URL}/auth/signup`, { email, password });
};
