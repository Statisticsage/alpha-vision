import { Navigate } from "react-router-dom";

const PrivateRoute = ({ children }) => {
  const token = localStorage.getItem("token"); // Check authentication
  return token ? children : <Navigate to="/login" replace />;
};

export default PrivateRoute;
