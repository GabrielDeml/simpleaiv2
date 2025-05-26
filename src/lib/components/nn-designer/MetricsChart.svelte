<script lang="ts">
  /**
   * MetricsChart Component
   * 
   * Purpose: Renders a real-time line chart of training metrics using
   * HTML5 Canvas. Displays both training and validation loss over epochs
   * with automatic scaling and grid lines.
   * 
   * Key features:
   * - Canvas-based rendering for performance
   * - Auto-scaling Y-axis based on data range
   * - Grid lines for better readability
   * - Legend and axis labels
   * - Responsive to data updates via reactive statements
   */
  
  import { trainingHistory } from '$lib/nn-designer/stores';
  
  // Reference to the canvas element
  let canvasElement: HTMLCanvasElement;
  
  // Reactive: Redraw chart whenever training history updates
  $: if (canvasElement && $trainingHistory.loss.length > 0) {
    drawChart();
  }
  
  /**
   * Main chart drawing function
   * Renders the entire chart including grid, lines, legend, and labels
   */
  function drawChart() {
    const ctx = canvasElement.getContext('2d')!;
    const width = canvasElement.width;
    const height = canvasElement.height;
    
    // Chart margins for better layout - much larger left margin for Y-axis labels
    const margin = { top: 20, right: 20, bottom: 40, left: 80 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    // Clear canvas with gradient background
    const bgGradient = ctx.createLinearGradient(0, 0, 0, height);
    bgGradient.addColorStop(0, '#1a1a1a');
    bgGradient.addColorStop(1, '#0a0a0a');
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, width, height);
    
    // Save context for clipping
    ctx.save();
    ctx.translate(margin.left, margin.top);
    
    // Draw subtle grid lines
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 0.5;
    ctx.setLineDash([2, 4]);
    
    // Draw horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(chartWidth, y);
      ctx.stroke();
    }
    
    // Draw vertical grid lines
    const epochs = $trainingHistory.loss.length;
    const stepX = chartWidth / Math.max(epochs - 1, 1);
    const verticalLines = Math.min(epochs, 8);
    for (let i = 0; i <= verticalLines; i++) {
      const x = (i / verticalLines) * chartWidth;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, chartHeight);
      ctx.stroke();
    }
    
    ctx.setLineDash([]); // Reset line dash
    
    // Calculate data range for Y-axis scaling
    const allValues = [
      ...$trainingHistory.loss,
      ...$trainingHistory.valLoss
    ].filter(v => v !== null && !isNaN(v));
    
    const maxValue = Math.max(...allValues, 0.1);
    const minValue = Math.min(...allValues, 0);
    const range = maxValue - minValue || 1;
    
    // Scale functions with proper margins
    const scaleY = (value: number) => {
      return chartHeight - ((value - minValue) / range) * chartHeight;
    };
    
    const scaleX = (index: number) => {
      return (index / Math.max(epochs - 1, 1)) * chartWidth;
    };
    
    // Helper function to draw smooth curves
    function drawSmoothLine(data: number[], color: string, shadowColor: string) {
      if (data.length < 2) return;
      
      // Create gradient for the line
      const lineGradient = ctx.createLinearGradient(0, 0, chartWidth, 0);
      lineGradient.addColorStop(0, color + '80'); // Semi-transparent start
      lineGradient.addColorStop(1, color); // Full opacity end
      
      // Draw area fill
      ctx.fillStyle = ctx.createLinearGradient(0, 0, 0, chartHeight);
      (ctx.fillStyle as CanvasGradient).addColorStop(0, color + '20');
      (ctx.fillStyle as CanvasGradient).addColorStop(1, color + '05');
      
      ctx.beginPath();
      ctx.moveTo(scaleX(0), chartHeight);
      
      for (let i = 0; i < data.length; i++) {
        const x = scaleX(i);
        const y = scaleY(data[i]);
        
        if (i === 0) {
          ctx.lineTo(x, y);
        } else {
          // Smooth curve using quadratic curves
          const prevX = scaleX(i - 1);
          const prevY = scaleY(data[i - 1]);
          const cpX = (prevX + x) / 2;
          
          ctx.quadraticCurveTo(cpX, prevY, x, y);
        }
      }
      
      ctx.lineTo(scaleX(data.length - 1), chartHeight);
      ctx.closePath();
      ctx.fill();
      
      // Draw the line with shadow
      ctx.shadowColor = shadowColor;
      ctx.shadowBlur = 6;
      ctx.shadowOffsetY = 2;
      
      ctx.strokeStyle = lineGradient;
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const x = scaleX(i);
        const y = scaleY(data[i]);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          const prevX = scaleX(i - 1);
          const prevY = scaleY(data[i - 1]);
          const cpX = (prevX + x) / 2;
          
          ctx.quadraticCurveTo(cpX, prevY, x, y);
        }
      }
      ctx.stroke();
      
      // Reset shadow
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;
      
      // Draw data points
      ctx.fillStyle = color;
      for (let i = 0; i < data.length; i++) {
        const x = scaleX(i);
        const y = scaleY(data[i]);
        
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
        
        // White center for points
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(x, y, 1.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = color;
      }
    }
    
    // Draw training loss line (red/orange gradient)
    drawSmoothLine($trainingHistory.loss, '#ff6b35', '#ff6b3550');
    
    // Draw validation loss line (blue gradient)  
    drawSmoothLine($trainingHistory.valLoss, '#4f9eff', '#4f9eff50');
    
    // Restore context for legend and labels
    ctx.restore();
    
    // Draw improved legend with background
    const legendX = 15;
    const legendY = 15;
    const legendWidth = 120;
    const legendHeight = 50;
    
    // Legend background with rounded corners
    ctx.fillStyle = '#1f1f1f';
    ctx.fillRect(legendX, legendY, legendWidth, legendHeight);
    ctx.strokeStyle = '#404040';
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);
    
    // Legend items
    ctx.font = '11px -apple-system, system-ui, sans-serif';
    
    // Training Loss
    ctx.fillStyle = '#ff6b35';
    ctx.fillRect(legendX + 10, legendY + 12, 12, 3);
    ctx.fillStyle = '#ffffff';
    ctx.fillText('Training Loss', legendX + 28, legendY + 17);
    
    // Validation Loss
    ctx.fillStyle = '#4f9eff';
    ctx.fillRect(legendX + 10, legendY + 28, 12, 3);
    ctx.fillStyle = '#ffffff';
    ctx.fillText('Validation Loss', legendX + 28, legendY + 33);
    
    // Draw axis labels with better positioning
    ctx.fillStyle = '#a3a3a3';
    ctx.font = '10px -apple-system, system-ui, sans-serif';
    
    // Y-axis labels - positioned outside chart area
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const labelSteps = 5;
    for (let i = 0; i <= labelSteps; i++) {
      const value = minValue + (range * i) / labelSteps;
      const y = height - margin.bottom - (i / labelSteps) * chartHeight;
      
      ctx.fillText(value.toFixed(3), margin.left - 15, y);
    }
    ctx.textAlign = 'left'; // Reset text alignment
    
    // X-axis labels - centered under chart
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xLabelSteps = Math.min(epochs - 1, 6);
    for (let i = 0; i <= xLabelSteps; i++) {
      const epochNum = Math.round((i / xLabelSteps) * (epochs - 1));
      const x = margin.left + (i / xLabelSteps) * chartWidth;
      
      ctx.fillText(epochNum.toString(), x, height - margin.bottom + 8);
    }
    ctx.textAlign = 'left'; // Reset text alignment
    
    // Axis titles
    ctx.fillStyle = '#d4d4d4';
    ctx.font = '12px -apple-system, system-ui, sans-serif';
    
    // Y-axis title
    ctx.save();
    ctx.translate(20, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Loss', 0, 0);
    ctx.restore();
    
    // X-axis title
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', width / 2, height - 8);
  }
</script>

<!-- 
  Canvas element for chart rendering
  - Fixed resolution for crisp rendering
  - CSS scales to container size
-->
<canvas 
  bind:this={canvasElement}
  width="560"
  height="180"
></canvas>

<style>
  canvas {
    width: 100%;
    height: 100%;
    display: block;
    border-radius: 8px;
    border: 1px solid #333333;
    background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
    box-shadow: 
      0 4px 6px rgba(0, 0, 0, 0.3),
      inset 0 1px 0 rgba(255, 255, 255, 0.05);
  }
</style>