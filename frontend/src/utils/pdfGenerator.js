import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';

export const generatePDFReport = (reportData, patientName = "Unknown") => {
  try {
    console.log("Starting PDF generation for:", patientName, reportData);
    const doc = new jsPDF();
    
    // Header
    doc.setFontSize(22);
    doc.setTextColor(40, 40, 40);
    doc.text('BrainScan AI - Medical Report', 14, 22);
    
    doc.setFontSize(10);
    doc.setTextColor(100, 100, 100);
    doc.text(`Generated on: ${new Date().toLocaleString()}`, 14, 30);
    
    doc.setLineWidth(0.5);
    doc.line(14, 34, 196, 34);
    
    // Patient Details
    doc.setFontSize(14);
    doc.setTextColor(0, 0, 0);
    doc.text('Patient Information', 14, 45);
    
    autoTable(doc, {
      startY: 50,
      head: [['Field', 'Details']],
      body: [
        ['Patient Name', patientName || 'N/A'],
        ['Scan Date', reportData.date ? new Date(reportData.date).toLocaleString() : new Date().toLocaleString()],
        ['Report ID', reportData.id || 'N/A']
      ],
      theme: 'grid',
      headStyles: { fillColor: [41, 128, 185] },
      margin: { left: 14 }
    });
    
    // Diagnostic Result
    const finalY = (doc.lastAutoTable?.finalY || 80) + 10;
    doc.text('AI Diagnostic Result', 14, finalY);
    
    let predictionColor = [0, 229, 160]; // Green for notumor
    if (reportData.prediction !== 'notumor') {
       predictionColor = [255, 79, 109]; // Red for tumors
    }
    
    autoTable(doc, {
      startY: finalY + 5,
      head: [['Prediction', 'Confidence']],
      body: [
        [(reportData.prediction || "UNKNOWN").toUpperCase(), `${((reportData.confidence || 0) * 100).toFixed(2)}%`]
      ],
      theme: 'grid',
      headStyles: { fillColor: [41, 128, 185] },
      bodyStyles: { fontStyle: 'bold' },
      didParseCell: function(data) {
        if (data.section === 'body' && data.column.index === 0) {
          data.cell.styles.textColor = predictionColor;
        }
      },
      margin: { left: 14 }
    });
    
    // Heatmap Visualization (if available)
    if (reportData.overlay_b64) {
      const nextY = (doc.lastAutoTable?.finalY || 120) + 15;
      doc.text('Grad-CAM Tumor Localization', 14, nextY);
      
      // Add image
      try {
        if (reportData.original_b64) {
            doc.addImage(`data:image/png;base64,${reportData.original_b64}`, 'PNG', 14, nextY + 5, 75, 75);
        }
        doc.addImage(`data:image/png;base64,${reportData.overlay_b64}`, 'PNG', 100, nextY + 5, 75, 75);
        
        doc.setFontSize(10);
        doc.setTextColor(150, 150, 150);
        doc.text('(Left: Original MRI, Right: AI Attention Heatmap)', 14, nextY + 85);
      } catch(err) {
        console.warn("Could not embed image to PDF", err);
      }
    }
    
    // Footer
    doc.setFontSize(10);
    doc.setTextColor(150, 150, 150);
    doc.text('Note: This is an AI generated summary. Please review with an oncologist.', 14, 280);
    
    const fileName = `Report_${(patientName || 'Unknown').replace(/ /g, '_')}_${reportData.id || 'Current'}.pdf`;
    doc.save(fileName);
    console.log("PDF saved successfully:", fileName);
  } catch (error) {
    console.error("PDF Generation Error:", error);
    alert("Critical failure during PDF generation. Check console for details.");
  }
};
