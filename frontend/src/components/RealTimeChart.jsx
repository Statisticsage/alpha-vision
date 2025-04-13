import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";

const RealTimeChart = ({ wsUrl }) => {
  const [data, setData] = useState([]);
  const [layout, setLayout] = useState({
    title: "📈 Live Data Stream",
    xaxis: { title: "Time", type: "date" },
    yaxis: { title: "Prediction Value" },
    autosize: true,
    paper_bgcolor: "#1a1a1a",
    plot_bgcolor: "#1a1a1a",
    font: { color: "#e0e0e0" },
  });

  useEffect(() => {
    const ws = new WebSocket(wsUrl);
    ws.onmessage = (event) => {
      const newPoint = JSON.parse(event.data);
      setData((prevData) => [...prevData, newPoint]);
    };
    return () => ws.close();
  }, [wsUrl]);

  return (
    <Plot
      data={[
        {
          x: data.map((d) => d.timestamp),
          y: data.map((d) => d.value),
          type: "scatter",
          mode: "lines+markers",
          marker: { color: "cyan" },
        },
      ]}
      layout={layout}
      useResizeHandler
      style={{ width: "100%", height: "100%", minHeight: "500px" }}
    />
  );
};

export default RealTimeChart;
