import React, { useState, useEffect, useRef, memo, useMemo } from "react";
import {
  Line, Bar, Pie, Scatter, Doughnut, Radar, Bubble, PolarArea
} from "react-chartjs-2";
import { Chart as ChartJS, registerables } from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";
import * as XLSX from "xlsx";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import regression from "regression";
import "../styles/UserDashboard.css";
import Chatbot from "./Chatbot";
import { ClipLoader } from "react-spinners";

// Register chart components and plugins
ChartJS.register(...registerables, zoomPlugin);

const chartTypes = {
  bar: Bar,
  line: Line,
  pie: Pie,
  scatter: Scatter,
  doughnut: Doughnut,
  radar: Radar,
  bubble: Bubble,
  polar: PolarArea,
};

const palette = [
  "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
  "#66a61e", "#e6ab02", "#a6761d", "#666666"
];

const UserDashboard = memo(() => {
  const [data, setData] = useState([]);
  const [chartType, setChartType] = useState("bar");
  const [xAxis, setXAxis] = useState("");
  const [yAxis, setYAxis] = useState("");
  const [secondaryYAxis, setSecondaryYAxis] = useState("");
  const [filteredData, setFilteredData] = useState([]);
  const [stats, setStats] = useState({});
  const [regressionResult, setRegressionResult] = useState({});
  const [dataUploaded, setDataUploaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showHistogram, setShowHistogram] = useState(true);
  const [showKDE, setShowKDE] = useState(true);
  const chartRef = useRef(null);

  const chartData = useMemo(() => {
    const datasets = [];

    if (yAxis) {
      datasets.push({
        label: `${xAxis} vs ${yAxis}`,
        data: filteredData.map(item => ({
          x: item[xAxis],
          y: parseFloat(item[yAxis]) || 0
        })),
        borderColor: palette[0],
        backgroundColor: palette[0] + "88",
        borderWidth: 2,
      });
    }

    if (secondaryYAxis) {
      datasets.push({
        label: `${xAxis} vs ${secondaryYAxis}`,
        data: filteredData.map(item => ({
          x: item[xAxis],
          y: parseFloat(item[secondaryYAxis]) || 0
        })),
        borderColor: palette[1],
        backgroundColor: palette[1] + "88",
        borderWidth: 2,
        yAxisID: "y1",
      });
    }

    if (showHistogram) {
      datasets.push({
        label: "Histogram Data",
        data: filteredData.map(item => item[yAxis]),
        backgroundColor: palette[2] + "88",
        type: "bar",
        yAxisID: "y2",
      });
    }
    if ((chartType === "line" || chartType === "scatter") && filteredData.length > 1) {
        const regressionData = filteredData
          .map(item => [parseFloat(item[xAxis]), parseFloat(item[yAxis])])
          .filter(pair => pair.every(Number.isFinite));
      
        const result = regression.linear(regressionData);
        const points = result.points;
      
        datasets.push({
          label: "Regression Line",
          data: points.map(([x, y]) => ({ x, y })),
          borderColor: "#ff6384",
          borderWidth: 2,
          type: "line",
          fill: false,
          borderDash: [5, 5],
        });
      }
      
    if (showKDE) {
      datasets.push({
        label: "KDE Data",
        data: filteredData.map(item => item[yAxis]),
        backgroundColor: palette[3] + "88",
        type: "line",
        yAxisID: "y3",
      });
    }

    return {
      labels: filteredData.map(item => item[xAxis]),
      datasets,
    };
  }, [filteredData, xAxis, yAxis, secondaryYAxis, showHistogram, showKDE, chartType]);

  useEffect(() => {
    if (xAxis && yAxis && data.length > 0) filterData();
  }, [xAxis, yAxis, data]);

  useEffect(() => {
    if (filteredData.length) {
      updateStatistics();
      updateRegression();
    }
  }, [filteredData]);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    if (file.size > 5 * 1024 * 1024) {
      alert("⚠️ File too large! Max: 5MB.");
      return;
    }

    setLoading(true);
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { type: "array" });
        const sheet = workbook.Sheets[workbook.SheetNames[0]];
        const jsonData = XLSX.utils.sheet_to_json(sheet);
        setData(jsonData);
        setFilteredData(jsonData);
        setDataUploaded(true);
      } catch (err) {
        console.error(" Error reading file:", err);
        alert("Failed to load file. Ensure it's a valid format.");
      } finally {
        setLoading(false);
      }
    };
    reader.readAsArrayBuffer(file);
  };

  const filterData = () => {
    const valid = data.filter(
      item =>
        item[xAxis] !== undefined &&
        item[yAxis] !== undefined &&
        !isNaN(parseFloat(item[yAxis]))
    );
    setFilteredData(valid);
  };

  const updateStatistics = () => {
    const values = filteredData.map(item => parseFloat(item[yAxis])).filter(Number.isFinite);
    if (!values.length) return;
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    const stdDev = Math.sqrt(values.reduce((sum, num) => sum + Math.pow(num - avg, 2), 0) / values.length);
    const sorted = [...values].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];

    setStats({
      avg: avg.toFixed(4),
      min: Math.min(...values).toFixed(4),
      max: Math.max(...values).toFixed(4),
      median,
      stdDev: stdDev.toFixed(4),
    });
  };

  const updateRegression = () => {
    const dataset = filteredData
      .map(item => [parseFloat(item[xAxis]), parseFloat(item[yAxis])])
      .filter(pair => pair.every(Number.isFinite));
    if (!dataset.length) return;

    const result = regression.linear(dataset);
    setRegressionResult({
      slope: result.equation[0].toFixed(4),
      intercept: result.equation[1].toFixed(4),
      r2: result.r2.toFixed(4),
      trend: result.equation[0] > 0 ? "📈 Upward Trend" : "📉 Downward Trend",
    });
  };

  const getChartOptions = () => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: {
      tooltip: {
        callbacks: {
          label: function(context) {
            const x = context.label;
            const y = context.raw;
            return `${context.dataset.label}: (${x}, ${y})`;
          }
        }
      },
      zoom: {
        pan: {
          enabled: true,
          mode: "xy",
        },
        zoom: {
          wheel: { enabled: true },
          pinch: { enabled: true },
          mode: "xy",
        },
      },
    },
    scales: {
      x: { title: { display: true, text: xAxis } },
      y: {
        type: "linear",
        position: "left",
        title: { display: true, text: yAxis },
      },
      y1: {
        type: "linear",
        position: "right",
        title: { display: true, text: secondaryYAxis || "Secondary Y" },
        grid: { drawOnChartArea: false },
      },
      y2: {
        type: "linear",
        position: "right",
        title: { display: true, text: "Histogram" },
        grid: { drawOnChartArea: false },
      },
      y3: {
        type: "linear",
        position: "right",
        title: { display: true, text: "KDE" },
        grid: { drawOnChartArea: false },
      },
    },
  });

  const handleDownload = async () => {
    const canvas = chartRef.current?.canvas;
    if (!canvas) return;
    const image = await html2canvas(canvas);
    const pdf = new jsPDF();
    pdf.addImage(image.toDataURL("image/png"), "PNG", 10, 10, 190, 100);
    pdf.save("chart-report.pdf");
  };
  const handleCSVDownload = () => {
    const ws = XLSX.utils.json_to_sheet(filteredData);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "FilteredData");
    XLSX.writeFile(wb, "filtered_data.xlsx");
  };
  
  return (
    <div className="dashboard-container">
      <h1 className="dashboard-title">📊 User Analytics Dashboard</h1>

      <input
        type="file"
        accept=".csv, .xlsx"
        onChange={handleFileUpload}
        className="dashboard-input-file"
        aria-label="Upload data file"
      />

      <div className="dashboard-controls">
        <select onChange={e => setChartType(e.target.value)} className="dashboard-select">
          {Object.keys(chartTypes).map(type => (
            <option key={type} value={type}>{type.toUpperCase()}</option>
          ))}
        </select>

        <select onChange={e => setXAxis(e.target.value)} className="dashboard-select" disabled={!data.length}>
          <option value="">Select X-Axis</option>
          {data.length && Object.keys(data[0]).map(col => (
            <option key={col} value={col}>{col}</option>
          ))}
        </select>

        <select onChange={e => setYAxis(e.target.value)} className="dashboard-select" disabled={!data.length}>
          <option value="">Select Y-Axis</option>
          {data.length && Object.keys(data[0]).map(col => (
            <option key={col} value={col}>{col}</option>
          ))}
        </select>

        <select onChange={e => setSecondaryYAxis(e.target.value)} className="dashboard-select" disabled={!data.length}>
          <option value="">Select Secondary Y-Axis</option>
          {data.length && Object.keys(data[0]).map(col => (
            <option key={col} value={col}>{col}</option>
          ))}
        </select>

        <label>
          <input type="checkbox" checked={showHistogram} onChange={() => setShowHistogram(!showHistogram)} />
          Show Histogram
        </label>

        <label>
          <input type="checkbox" checked={showKDE} onChange={() => setShowKDE(!showKDE)} />
          Show KDE
        </label>
      </div>

      {loading ? (
  <div className="spinner-container">
    <ClipLoader size={50} color={"#123abc"} loading={loading} />
    <p>Loading data, please wait...</p>
  </div>
) :(
        dataUploaded && xAxis && yAxis && (
          <div>
            <div className="chart-container">
              {filteredData.length ? (
                React.createElement(chartTypes[chartType], {
                  ref: chartRef,
                  data: chartData,
                  options: getChartOptions(),
                })
              ) : (
                <p>No valid data to display.</p>
              )}
            </div>

            <div className="stats-container">
              <h3>📈 Regression Analysis</h3>
              <p>y = {regressionResult.slope}x + {regressionResult.intercept} | R²: {regressionResult.r2} | {regressionResult.trend}</p>

              <h3>📉 Statistics</h3>
              <p>
                Avg: {stats.avg} | Median: {stats.median} | Min: {stats.min} | Max: {stats.max} | Std Dev: {stats.stdDev}
              </p>
            </div>

            <button onClick={handleDownload} className="download-button">📥 Download PDF</button>
            <button onClick={() => chartRef.current?.resetZoom()} className="reset-button">🔄 Reset Zoom</button>
            <button onClick={handleCSVDownload} className="download-button">📤 Export to Excel</button>
          </div>
        )
      )}

      <Chatbot dashboardData={filteredData} />
    </div>
  );
});
export default UserDashboard;
