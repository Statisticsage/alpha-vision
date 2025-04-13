import React, { useState, useEffect, useRef } from "react";
import { askAI } from "../api/chatbotService"; // Ensure this path is correct
import "../styles/Chatbot.css";

const MAX_RESPONSE_LENGTH = 500;

const Chatbot = ({ dashboardData }) => {
    const [messages, setMessages] = useState([]);
    const [userInput, setUserInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [isOpen, setIsOpen] = useState(false);
    const chatRef = useRef(null);

    // 🔄 Auto-scroll chat to bottom if user is near the bottom
    useEffect(() => {
        if (chatRef.current) {
            const shouldScroll =
                chatRef.current.scrollHeight - chatRef.current.scrollTop <=
                chatRef.current.clientHeight + 50;
            if (shouldScroll) {
                chatRef.current.scrollTop = chatRef.current.scrollHeight;
            }
        }
    }, [messages]);

    const handleSend = async () => {
        const trimmedInput = userInput.trim();
        if (!trimmedInput || loading) return; // Prevent empty & duplicate sends

        // Log the dashboard data before sending
        console.log("📊 dashboardData being sent to AI:", dashboardData);

        // Ensure dashboardData is valid
        if (!Array.isArray(dashboardData) || dashboardData.length === 0) {
            setMessages(prev => [...prev, { text: "⚠️ No dashboard data available for analysis.", sender: "bot" }]);
            return;
        }

        const newMessage = { text: trimmedInput, sender: "user" };
        setMessages(prev => [...prev, newMessage]);
        setUserInput(""); // Clear input
        setLoading(true);

        try {
            const response = await askAI(trimmedInput, dashboardData);
            setMessages(prev => [...prev, response]);
        } catch (error) {
            console.error("AI Response Error:", error);
            setMessages(prev => [...prev, { text: "⚠️ AI failed to respond. Please try again.", sender: "bot" }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className={`chat-container ${isOpen ? "open" : ""}`}>
            {/* 💬 Chat Toggle Button */}
            <div
                className="chat-icon"
                onClick={() => setIsOpen(!isOpen)}
                aria-label="Toggle chatbot"
                role="button"
            >
                💬
            </div>
            {isOpen && (
                <div className="chat-box">
                    {/* 📩 Messages */}
                    <div className="chat-messages" ref={chatRef}>
                        {messages.map((msg, index) => (
                            <div key={index} className={`message ${msg.sender}`}>
                                {msg.text}
                            </div>
                        ))}
                    </div>
                    {/* ✍️ Input Section */}
                    <div className="chat-input">
                        <input
                            type="text"
                            value={userInput}
                            onChange={(e) => setUserInput(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === "Enter" && !loading && userInput.trim()) {
                                    handleSend();
                                }
                            }}
                            placeholder="Ask about the dashboard..."
                            disabled={loading}
                            aria-label="Chat input"
                        />
                        <button onClick={handleSend} disabled={loading || !userInput.trim()}>
                            {loading ? "⏳ Sending..." : "Send"}
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Chatbot;
