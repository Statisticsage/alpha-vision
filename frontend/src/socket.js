import { useEffect } from "react";
import { io } from "socket.io-client";

const socket = io("http://localhost:8000"); // Replace with your WebSocket server URL

export const useSocket = (event, callback) => {
  useEffect(() => {
    socket.on(event, (data) => {
      callback(data);
    });

    return () => {
      socket.off(event);
    };
  }, [event, callback]);
};
