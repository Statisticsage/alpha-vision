const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

export const fetchRealTimeData = async () => {
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
            const response = await fetch("/api/realtime-data"); // Vite proxy handles this in dev
            if (!response.ok) {
                throw new Error(`Attempt ${attempt}: Server responded with ${response.status}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.warn(`⚠️ Retry ${attempt}/${MAX_RETRIES} failed: ${error.message}`);
            if (attempt < MAX_RETRIES) {
                await new Promise((res) => setTimeout(res, RETRY_DELAY));
            } else {
                console.error("❌ Final attempt failed. Error fetching real-time data:", error);
                return []; // Graceful fallback (return empty array instead of null)
            }
        }
    }
};
