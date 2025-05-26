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
    
    // Clear canvas with dark background
    ctx.fillStyle = '#0f0f0f';
    ctx.fillRect(0, 0, width, height);
    
    // Configure grid style
    ctx.strokeStyle = '#262626';
    ctx.lineWidth = 1;
    
    // Draw horizontal grid lines (5 lines total)
    for (let i = 0; i <= 4; i++) {
      const y = (height / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    
    // Draw vertical grid lines (up to 10 lines based on epochs)
    const epochs = $trainingHistory.loss.length;
    const stepX = width / Math.max(epochs - 1, 1);
    for (let i = 0; i < epochs; i += Math.ceil(epochs / 10)) {
      const x = i * stepX;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    // Calculate data range for Y-axis scaling
    // Combine all loss values to find global min/max
    const allValues = [
      ...$trainingHistory.loss,
      ...$trainingHistory.valLoss
    ].filter(v => v !== null && !isNaN(v));
    
    const maxValue = Math.max(...allValues, 0.1);  // Ensure minimum scale
    const minValue = Math.min(...allValues, 0);
    const range = maxValue - minValue || 1;  // Prevent division by zero
    
    /**
     * Scales a value to canvas Y coordinate
     * @param value - The data value to scale
     * @returns Y coordinate on canvas (inverted, with padding)
     * 
     * - Canvas Y coordinates are inverted (0 at top)
     * - 5% padding on top and bottom for visual clarity
     */
    const scaleY = (value: number) => {
      return height - ((value - minValue) / range) * height * 0.9 - height * 0.05;
    };
    
    // Draw training loss line (red)
    if ($trainingHistory.loss.length > 1) {
      ctx.strokeStyle = '#ef4444';  // Red color
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      for (let i = 0; i < $trainingHistory.loss.length; i++) {
        const x = i * stepX;
        const y = scaleY($trainingHistory.loss[i]);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }
    
    // Draw validation loss line (blue)
    if ($trainingHistory.valLoss.length > 1) {
      ctx.strokeStyle = '#3b82f6';  // Blue color
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      for (let i = 0; i < $trainingHistory.valLoss.length; i++) {
        const x = i * stepX;
        const y = scaleY($trainingHistory.valLoss[i]);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }
    
    // Draw legend in top-left corner
    ctx.font = '12px system-ui';
    
    // Loss legend item
    ctx.fillStyle = '#ef4444';
    ctx.fillRect(10, 10, 20, 2);  // Color indicator
    ctx.fillStyle = '#ffffff';
    ctx.fillText('Loss', 35, 15);
    
    // Validation loss legend item
    ctx.fillStyle = '#3b82f6';
    ctx.fillRect(10, 25, 20, 2);  // Color indicator
    ctx.fillStyle = '#ffffff';
    ctx.fillText('Val Loss', 35, 30);
    
    // Draw axis labels
    ctx.fillStyle = '#737373';
    ctx.font = '10px system-ui';
    
    // Y-axis labels (min and max values)
    ctx.fillText('0', 5, height - 5);
    ctx.fillText(maxValue.toFixed(3), 5, 15);
    
    // X-axis label (epoch count)
    ctx.fillText(`Epoch ${epochs}`, width - 50, height - 5);
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
    /* Scale canvas to fill container while maintaining aspect ratio */
    width: 100%;
    height: 100%;
    display: block;
  }
</style>