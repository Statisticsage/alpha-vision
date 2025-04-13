import React, { useRef } from "react";
import { jsPDF } from "jspdf";
import html2canvas from "html2canvas";

const PDFExport = ({ elementId, fileName = "dashboard-report.pdf" }) => {
  const exportToPDF = async () => {
    const input = document.getElementById(elementId);
    if (!input) {
      console.error("❌ Element not found:", elementId);
      return;
    }

    try {
      const canvas = await html2canvas(input);
      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF("p", "mm", "a4");
      pdf.addImage(imgData, "PNG", 10, 10, 190, 0);
      pdf.save(fileName);
    } catch (err) {
      console.error("❌ PDF Export Error:", err);
    }
  };

  return (
    <button className="export-btn" onClick={exportToPDF}>
      📄 Download PDF
    </button>
  );
};

export default PDFExport;
