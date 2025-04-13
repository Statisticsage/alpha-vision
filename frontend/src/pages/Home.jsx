import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import "../styles/index.css";
import videoSrc from '../assets/Video_1.mp4'; // Import the video file
import { 
    FaCloudUploadAlt, 
    FaFileAlt, 
    FaSignOutAlt, 
    FaUser, 
    FaCog, 
    FaMoon, 
    FaSun, 
    FaUsers 
} from "react-icons/fa";

const HomePage = () => {
    const [file, setFile] = useState(null);
    const [task, setTask] = useState("prediction");
    const [message, setMessage] = useState("");
    const [loading, setLoading] = useState(false);
    const [showLogoutConfirmation, setShowLogoutConfirmation] = useState(false);
    const [landingDarkMode, setLandingDarkMode] = useState(true);
    const navigate = useNavigate();

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        const allowedTypes = [
            "text/csv",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ];
        if (selectedFile && !allowedTypes.includes(selectedFile.type)) {
            setMessage("⚠️ Invalid file type. Please upload a CSV or Excel file.");
            setFile(null);
            return;
        }
        setFile(selectedFile);
    };

    const handleFileUpload = async () => {
        if (!file) {
            setMessage("⚠️ Please select a file to upload.");
            return;
        }
        const formData = new FormData();
        formData.append("file", file);
        formData.append("task", task);
        try {
            setLoading(true);
            setMessage("");
            const response = await axios.post("http://localhost:8000/process", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                    Authorization: `Bearer ${localStorage.getItem("token")}`,
                },
            });
            const fileId = response.data?.file_id;
            if (!fileId) {
                throw new Error("❌ No file_id returned from backend!");
            }
            setMessage("✅ File processed successfully!");
            setTimeout(() => {
                navigate(`/dashboard?fileId=${fileId}&task=${task}`);
            }, 1500);
        } catch (error) {
            setMessage(
                error.response?.data?.detail || "❌ Error uploading file. Please try again."
            );
        } finally {
            setLoading(false);
        }
    };

    const handleLogout = () => {
        localStorage.removeItem("token");
        navigate("/login", { replace: true });
    };

    const toggleLandingDarkMode = () => {
        setLandingDarkMode(!landingDarkMode);
    };

    return (
        <div className={`landing-home-container ${landingDarkMode ? "landing-dark-mode" : ""}`}>
            <div className="landing-sidebar">
                <h2>Alpha</h2>
                <ul>
                    <li onClick={() => navigate("/dashboard")}>
                        <FaUser /> Dashboard
                    </li>
                    <li onClick={() => navigate("/user-dashboard")}>
                        <FaUsers /> User Dashboard
                    </li>
{/* <li onClick={() => navigate("/predictions")}>
    <FaFileAlt /> Predictions
</li> */}
{/* <li onClick={() => navigate("/analysis")}>
    <FaCloudUploadAlt /> Analysis
</li> */}
{/* <li onClick={() => navigate("/settings")}>
    <FaCog /> Settings
</li> */}

                    <li onClick={() => setShowLogoutConfirmation(true)}>
                        <FaSignOutAlt /> Logout
                    </li>
                </ul>
                <div className="landing-dark-mode-toggle" onClick={toggleLandingDarkMode}>
                    {landingDarkMode ? <FaSun /> : <FaMoon />}
                </div>
                {showLogoutConfirmation && (
                    <div className="landing-logout-confirmation">
                        <p>Are you sure you want to log out?</p>
                        <button onClick={handleLogout}>Yes</button>
                        <button onClick={() => setShowLogoutConfirmation(false)}>No</button>
                    </div>
                )}
            </div>
            <div className="landing-main-content">
                <div className="landing-left-section">
                    <h1>AI-Powered Business Intelligence</h1>
                    <p>Upload your data and let Alpha generate insights.</p>
                    <video autoPlay loop muted className="landing-background-video">
                        <source src={videoSrc} type="video/mp4" />
                        Your browser does not support the video tag.
                    </video>
                </div>
                <div className="landing-right-section">
                    <h2>Upload Your Data</h2>
                    <select
                        onChange={(e) => setTask(e.target.value)}
                        value={task}
                        className="landing-task-dropdown"
                    >
                        <option value="prediction">Make Prediction</option>
                        <option value="analysis">Perform Analysis</option>
                    </select>
                    <input
                        type="file"
                        accept=".csv,.xls,.xlsx"
                        onChange={handleFileChange}
                        className="landing-file-input"
                    />
                    <button
                        onClick={handleFileUpload}
                        disabled={loading || !file}
                        className="landing-upload-btn"
                    >
                        {loading ? <span className="landing-spinner"></span> : "Upload & Process"}
                    </button>
                    {message && <p className="landing-message-box">{message}</p>}
                </div>
            </div>
        </div>
    );
};

export default HomePage;
