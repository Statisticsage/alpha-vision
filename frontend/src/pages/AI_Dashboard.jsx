import React, { useEffect, useState, useCallback, useRef } from "react";
import axios from "axios";
import { useLocation } from "react-router-dom";
import Plot from "react-plotly.js";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import "../styles/dashboard.css";

const Dashboard = () => {
  const location = useLocation();
  const params = new URLSearchParams(location.search);
  const fileId = params.get("fileId");
  const task = params.get("task");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [isPolling, setIsPolling] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);
  
  const dashboardRef = useRef(null);

  const fetchResults = useCallback(async () => {
    if (!fileId || !task) {
        setError("Please upload data.");
        setLoading(true);
        return;
    }
    try {
        console.log(`📡 Fetching ${task} results for file ID: ${fileId}`);
        const endpoint = task === "analysis" 
            ? `/analysis/result/${fileId}` 
            : `/result/${fileId}`;

        const response = await axios.get(`http://localhost:8000${endpoint}`);
        const data = response.data;
        console.log("✅ API Response:", data);

        if (data.status === "processing") {
            setIsPolling(true); // Keep polling
            setError("Analysis is still processing, please wait..."); // Inform user to wait
        } else if (data.status === "not_found") {
            setError("Analysis result not found. Please check the file ID and try again.");
            setIsPolling(false);
        } else {
            setResult(data);
            setIsPolling(false); // Result ready
            setError(null); // Clear any previous errors
        }
    } catch (err) {
        console.error("❌ API Error:", err.response ? err.response.data : err.message);
        if (err.response?.data?.detail) {
            setError(err.response.data.detail);
        } else {
            setError("Failed to fetch results. Please try again.");
        }
        setIsPolling(false);
    } finally {
        setLoading(false);
    }
}, [fileId, task]);


  useEffect(() => {
    fetchResults();
    let interval;
    if (isPolling) {
      interval = setInterval(fetchResults, 10000); // Poll every 10 seconds
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [fetchResults, isPolling]);

  const downloadPDF = async () => {
    if (!dashboardRef.current) {
      alert("No content to download. Please wait for results to load.");
      return;
    }
    try {
      setPdfLoading(true);
      console.log("Starting PDF generation...");
      
      setTimeout(async () => {
        const input = dashboardRef.current;
        const canvas = await html2canvas(input, {
          scale: 2,
          useCORS: true,
          allowTaint: true,
          logging: true,
          letterRendering: true,
          foreignObjectRendering: false
        });
        console.log("Canvas generated successfully");
        const imgData = canvas.toDataURL("image/png");
        const pdf = new jsPDF({
          orientation: "portrait",
          unit: "mm",
          format: "a4"
        });
        
        const imgWidth = 210; // A4 width in mm
        const imgHeight = (canvas.height * imgWidth) / canvas.width;
        
        pdf.setFont("helvetica", "bold");
        pdf.text("📊 AI Business Intelligence Report", 10, 10);
        pdf.addImage(imgData, "PNG", 10, 20, imgWidth - 20, imgHeight);
        pdf.save(`AI_Report_${fileId}.pdf`);
        console.log("PDF saved successfully");
        setPdfLoading(false);
      }, 1000);
      
    } catch (error) {
      console.error("PDF Generation Error:", error);
      alert("Failed to generate PDF. Please try again.");
      setPdfLoading(false);
    }
  };

  // Helper function to safely parse plot JSON
  const safelyParsePlotJson = (plotJson) => {
    if (!plotJson) return null;
    
    try {
      // If it's already an object, return it
      if (typeof plotJson === 'object') return plotJson;
      
      // Otherwise parse the JSON string
      return JSON.parse(plotJson);
    } catch (e) {
      console.error("❌ Failed to parse plot data:", e);
      return null;
    }
  };

  // Get a nice display name from the plot key
  const getDisplayName = (key) => {
    return key
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Render a single plot with error handling
  const renderPlot = (plotData, key) => {
    const parsedData = safelyParsePlotJson(plotData);
    
    if (!parsedData) {
      return (
        <div key={key} className="analytics-plot-error">
          <p>Error rendering plot. Invalid data format.</p>
        </div>
      );
    }
    
    if (!parsedData.data || !parsedData.layout) {
      console.error("Plot is missing data or layout:", parsedData);
      return (
        <div key={key} className="analytics-plot-error">
          <p>Error rendering plot. Missing data or layout.</p>
        </div>
      );
    }
    
    // Enhanced layout for better visualization
    const enhancedLayout = {
      ...parsedData.layout,
      autosize: true,
      height: 500,
      width: undefined,
      margin: { t: 50, b: 80, l: 80, r: 40 },
      font: { family: "Arial, sans-serif", size: 12, color: "#333" },
      paper_bgcolor: "#f8f9fa",
      plot_bgcolor: "#f8f9fa",
      title: {
        text: parsedData.layout.title?.text || getDisplayName(key),
        font: { size: 18, color: "#333", family: "Arial, sans-serif" }
      },
      xaxis: {
        ...parsedData.layout.xaxis,
        title: {
          ...parsedData.layout.xaxis?.title,
          font: { size: 14, color: "#555" }
        },
        gridcolor: "#e1e5e9"
      },
      yaxis: {
        ...parsedData.layout.yaxis,
        title: {
          ...parsedData.layout.yaxis?.title,
          font: { size: 14, color: "#555" }
        },
        gridcolor: "#e1e5e9"
      },
      legend: {
        ...parsedData.layout.legend,
        bgcolor: "rgba(255,255,255,0.5)",
        bordercolor: "#e1e5e9",
        borderwidth: 1
      }
    };
    
    return (
      <div key={key} className="analytics-plotly-chart">
        <h4 className="analytics-plotly-chart-title">
          {getDisplayName(key)}
        </h4>
        <div className="analytics-plotly-chart-container">
          <Plot
            data={parsedData.data}
            layout={enhancedLayout}
            useResizeHandler={true}
            style={{ width: "100%", height: "100%" }}
            config={{
              displayModeBar: true,
              displaylogo: false,
              responsive: true,
              toImageButtonOptions: {
                format: 'png',
                filename: `plot_${key}`,
                height: 800,
                width: 1200,
                scale: 2
              },
              modeBarButtonsToRemove: [
                'sendDataToCloud',
                'autoScale2d',
                'hoverClosestCartesian',
                'hoverCompareCartesian',
                'lasso2d',
                'select2d'
              ]
            }}
            className="analytics-plotly-chart-plot"
          />
        </div>
      </div>
    );
  };

  return (
    <div className="analytics-dashboard-wrapper">
      <div className="analytics-dashboard-container">
        <header className="analytics-dashboard-header">
          <h1 className="analytics-dashboard-title">📊 AI Business Intelligence Dashboard</h1>
          <button
            onClick={downloadPDF}
            className="analytics-report-download-btn"
            disabled={pdfLoading || loading}
          >
            {pdfLoading ? "⏳ Generating PDF..." : "📄 Download Report"}
          </button>
        </header>
        
        {loading && (
          <div className="analytics-loading-container">
            <div className="analytics-loading-spinner"></div>
            <p className="analytics-dashboard-loading-text">⏳ Loading results...</p>
          </div>
        )}
        
        {error && (
          <div className="analytics-error-container">
            <p className="analytics-dashboard-error-text">❌ {error}</p>
          </div>
        )}
        
        <div ref={dashboardRef} className="analytics-dashboard-content">

                    {/* Analysis Task Rendering */}
{result && task === "analysis" && (
  <div className="analytics-result-container">
    <section className="analytics-summary-section">
      <h2 className="analytics-section-title">🔍 Analysis Summary</h2>
      <div className="analytics-summary-card">
        <p className="analytics-summary-text">{result.ai_summary}</p>
      </div>

      {result.insights && (
        <div className="analytics-insights-card">
          <h3 className="analytics-insights-title">💡 Key Insights</h3>
          <p className="analytics-insights-text">{result.insights}</p>
        </div>
      )}
    </section>

    {result.plots && Object.keys(result.plots).length > 0 && (
      <section className="analytics-plots-section">
        <h2 className="analytics-section-title">📊 Interactive Visualizations</h2>
        <div className="analytics-plotly-charts-grid">
          {Object.entries(result.plots).map(([key, plotData]) => (
            renderPlot(plotData, key)
          ))}
        </div>
      </section>
    )}

    {result.dynamic_visuals && Object.keys(result.dynamic_visuals).length > 0 && (
      <section className="analytics-dynamic-plots-section">
        <h2 className="analytics-section-title">🎨 Visual Insights</h2>
        <div className="analytics-dynamic-plots-grid">
          {Object.entries(result.dynamic_visuals).map(([key, base64Img]) => (
            <div key={key} className="analytics-dynamic-plot-card">
              <h4 className="analytics-dynamic-plot-title">{getDisplayName(key)}</h4>
              <img 
                src={base64Img} 
                alt={`${getDisplayName(key)}`} 
                className="analytics-dynamic-plot-image"
              />
            </div>
          ))}
        </div>
      </section>
    )}
  </div>
)}
          {/* Prediction Task Rendering */}
          {result && task === "prediction" && (
            <div className="analytics-prediction-section">
              <h2 className="analytics-section-title">🔮 Prediction Results</h2>
              
              {result.status === "error" && result.error_message && (
                <div className="analytics-error-container">
                  <p className="analytics-dashboard-error-text">{result.error_message}</p>
                </div>
              )}
              
              {result.prediction_result && result.status !== "error" && (
                <>
                  <div className="analytics-prediction-card">
                    <h3 className="analytics-prediction-title">📈 Model Performance</h3>
                    <div className="analytics-prediction-details">
                      <div className="analytics-prediction-metric">
                        <span className="analytics-prediction-label">Model Type:</span>
                        <span className="analytics-prediction-value">{result.prediction_result.model_type}</span>
                      </div>
                      <div className="analytics-prediction-metric">
                        <span className="analytics-prediction-label">Target Column:</span>
                        <span className="analytics-prediction-value">{result.prediction_result.target_column}</span>
                      </div>
                      <div className="analytics-prediction-metric">
                        <span className="analytics-prediction-label">Average Prediction:</span>
                        <span className="analytics-prediction-value">{result.prediction_result.avg_prediction.toFixed(2)}</span>
                      </div>
                      
                      {result.prediction_result.MAE !== undefined && (
                        <div className="analytics-prediction-metric">
                          <span className="analytics-prediction-label">MAE:</span>
                          <span className="analytics-prediction-value">{result.prediction_result.MAE.toFixed(4)}</span>
                        </div>
                      )}
                      
                      {result.prediction_result.RMSE !== undefined && (
                        <div className="analytics-prediction-metric">
                          <span className="analytics-prediction-label">RMSE:</span>
                          <span className="analytics-prediction-value">{result.prediction_result.RMSE.toFixed(4)}</span>
                        </div>
                      )}
                      
                      {result.prediction_result.R2 !== undefined && (
                        <div className="analytics-prediction-metric">
                          <span className="analytics-prediction-label">R²:</span>
                          <span className="analytics-prediction-value">{result.prediction_result.R2.toFixed(4)}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {result.prediction_result.plots && result.prediction_result.plots.length > 0 && (
                    <div className="analytics-chart-container">
                      <h3 className="analytics-chart-title">📊 Prediction Visualizations</h3>
                      <div className="analytics-plotly-charts-grid">
                        {result.prediction_result.plots.map((plotJson, index) => (
                          renderPlot(plotJson, `prediction_plot_${index}`)
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
          
          {/* Queued Message for Analysis */}
          {result && task === "analysis" && result.message === "Queued for processing" && (
            <div className="analytics-queued-message">
              <div className="analytics-loading-spinner"></div>
              <p>🔄 Your analysis is being processed. This page will automatically update when results are ready.</p>
                        </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
