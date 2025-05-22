import React, { useRef, useImperativeHandle, forwardRef } from 'react';
import { DATASET_CONFIG, UI_CONFIG } from '../constants/mnistConfig.js';

/**
 * Canvas component for displaying MNIST images
 * @param {Object} props - Component props
 * @param {Object} ref - Forwarded ref for canvas access
 */
const ImageCanvas = forwardRef((props, ref) => {
  const canvasRef = useRef(null);

  // Expose canvas methods to parent component
  useImperativeHandle(ref, () => ({
    /**
     * Draw image data to canvas
     * @param {Float32Array} imageData - Flattened image pixel data (784 values)
     */
    drawImage: (imageData) => {
      if (!canvasRef.current || !imageData) return;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      
      // Create RGBA image array
      const imageArray = new Uint8ClampedArray(
        DATASET_CONFIG.IMAGE_WIDTH * DATASET_CONFIG.IMAGE_HEIGHT * 4
      );
      
      // Convert grayscale data to RGBA
      for (let i = 0; i < DATASET_CONFIG.IMAGE_WIDTH * DATASET_CONFIG.IMAGE_HEIGHT; i++) {
        const pixelValue = Math.round(imageData[i] * 255);
        const pixelIndex = i * 4;
        
        imageArray[pixelIndex] = pixelValue;     // Red
        imageArray[pixelIndex + 1] = pixelValue; // Green  
        imageArray[pixelIndex + 2] = pixelValue; // Blue
        imageArray[pixelIndex + 3] = 255;       // Alpha
      }

      // Create and draw image data
      const imgData = new ImageData(
        imageArray, 
        DATASET_CONFIG.IMAGE_WIDTH, 
        DATASET_CONFIG.IMAGE_HEIGHT
      );
      
      ctx.putImageData(imgData, 0, 0);
    },

    /**
     * Clear the canvas
     */
    clear: () => {
      if (!canvasRef.current) return;
      
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Fill with light background
      ctx.fillStyle = '#f7fafc';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    },

    /**
     * Get canvas element
     */
    getCanvas: () => canvasRef.current
  }));

  return (
    <div className="image-section">
      <h2 className="section-title">Test Image</h2>
      <div className="canvas-container">
        <canvas 
          ref={canvasRef} 
          width={DATASET_CONFIG.IMAGE_WIDTH} 
          height={DATASET_CONFIG.IMAGE_HEIGHT} 
          className="mnist-canvas"
          style={{ 
            width: `${UI_CONFIG.CANVAS_DISPLAY_SIZE}px`,
            height: `${UI_CONFIG.CANVAS_DISPLAY_SIZE}px`
          }}
          aria-label="MNIST digit image display"
        />
      </div>
    </div>
  );
});

ImageCanvas.displayName = 'ImageCanvas';

export default ImageCanvas;