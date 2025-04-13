export const askAI = async (userInput, dashboardData) => {
  try {
    if (!dashboardData || !Array.isArray(dashboardData) || dashboardData.length === 0) {
      throw new Error("Invalid dashboard data: must be a non-empty array");
    }

    const response = await fetch("http://localhost:8000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userInput, data: dashboardData }),
    });

    if (!response.ok) {
      const errorDetail = await response.json();

      // 👇 Check for meaningful error message from backend
      const errorMessage = errorDetail?.detail || "Unexpected AI API error.";
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return { text: data.response, sender: "bot" };
  } catch (error) {
    console.error("❌ AI Chatbot Error:", error);

    // 👇 Return the actual error message instead of a generic one
    return {
      text: error.message || "⚠️ AI is offline. Try again later.",
      sender: "bot",
    };
  }
};
