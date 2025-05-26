<!-- 
  This is a Svelte component for training an MNIST digit recognition model.
  Svelte components have three sections:
  1. <script> - JavaScript/TypeScript logic
  2. HTML template - The component's markup
  3. <style> - Component-scoped CSS
-->

<script lang="ts">
  // External dependencies
  import * as tf from '@tensorflow/tfjs';
  import { onMount } from 'svelte';
  
  // Local imports
  import { MnistData } from '../mnist/dataLoader';
  import { createCNNModel } from '../mnist/model';

  // ========== Component State ==========
  // All these variables are reactive - UI updates automatically when they change
  
  // Model and data
  let model: tf.Sequential | null = null;
  let data: MnistData | null = null;
  
  // Training state
  let isTraining = false;
  let isLoading = false;
  let currentEpoch = 0;
  let loss = 0;
  let accuracy = 0;
  
  // Training parameters
  let epochs = 10;
  let batchSize = 512;
  
  // UI elements
  let logs: string[] = [];
  let canvas: HTMLCanvasElement;

  // ========== Lifecycle ==========
  onMount(() => {
    loadData();
    // Clear canvas after it's rendered
    clearCanvas();
  });

  // ========== Data Loading ==========
  async function loadData() {
    isLoading = true;
    // In Svelte, we create a new array to trigger reactivity
    // This is why we use [...logs, 'new message'] instead of logs.push()
    logs = [...logs, 'Loading MNIST dataset...'];
    
    data = new MnistData();
    await data.load(); // Download and prepare the dataset
    
    logs = [...logs, 'Dataset loaded successfully!'];
    isLoading = false;
  }

  // ========== Training Functions ==========
  async function startTraining() {
    if (!data) {
      logs = [...logs, 'Error: Data not loaded'];
      return;
    }

    isTraining = true;
    currentEpoch = 0;
    
    // Create a new CNN model
    model = createCNNModel();
    logs = [...logs, 'Model created'];
    
    // Get training data (images and labels)
    const { xs: xTrain, ys: yTrain } = data.getTrainData();
    
    logs = [...logs, `Starting training for ${epochs} epochs...`];
    
    // Train the model using TensorFlow.js fit() method
    await model.fit(xTrain, yTrain, {
      batchSize,
      epochs,
      shuffle: true, // Shuffle data each epoch for better training
      validationSplit: 0.1, // Use 10% of data for validation
      callbacks: {
        // This callback runs after each epoch
        onEpochEnd: (epoch, log) => {
          currentEpoch = epoch + 1;
          loss = log?.loss || 0;
          accuracy = log?.acc || 0;
          // Update logs with training progress
          logs = [...logs, `Epoch ${epoch + 1}: loss = ${loss.toFixed(4)}, accuracy = ${(accuracy * 100).toFixed(2)}%`];
        }
      }
    });
    
    // Clean up tensors to free memory
    xTrain.dispose();
    yTrain.dispose();
    
    logs = [...logs, 'Training completed!'];
    isTraining = false;
    
    // Test the model after training
    await testModel();
  }

  // Evaluate model on test dataset
  async function testModel() {
    if (!model || !data) return;
    
    logs = [...logs, 'Evaluating on test set...'];
    
    // Get test data
    const { xs: xTest, ys: yTest } = data.getTestData();
    // Evaluate returns [loss, accuracy] tensors
    const result = model.evaluate(xTest, yTest) as tf.Scalar[];
    
    // Extract values from tensors
    const testLoss = await result[0].data();
    const testAccuracy = await result[1].data();
    
    logs = [...logs, `Test loss: ${testLoss[0].toFixed(4)}, Test accuracy: ${(testAccuracy[0] * 100).toFixed(2)}%`];
    
    // Clean up tensors
    xTest.dispose();
    yTest.dispose();
    result.forEach(t => t.dispose());
  }

  // ========== Canvas and Prediction Functions ==========
  async function predictDigit() {
    if (!model || !canvas) return;
    
    // Get canvas 2D context for reading pixel data
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.getImageData(0, 0, 280, 280);
    
    // Convert canvas image to tensor and preprocess
    // Pipeline: Canvas (280x280) → Grayscale → Resize (28x28) → Normalize → Add batch dim
    const input = tf.browser.fromPixels(imageData, 1) // 1 channel = grayscale
      .resizeNearestNeighbor([28, 28]) // Resize to MNIST dimensions
      .toFloat()
      .div(255.0) // Normalize pixel values from [0,255] to [0,1]
      .expandDims(0); // Add batch dimension: [28,28,1] → [1,28,28,1]
    
    // Run prediction - returns probabilities for each digit (0-9)
    const prediction = model.predict(input) as tf.Tensor;
    const probabilities = await prediction.data();
    
    // Find the digit with highest probability
    const predictedClass = probabilities.indexOf(Math.max(...probabilities));
    const confidence = Math.max(...probabilities) * 100;
    
    logs = [...logs, `Predicted digit: ${predictedClass} (confidence: ${confidence.toFixed(1)}%)`];
    
    // Clean up tensors
    input.dispose();
    prediction.dispose();
  }

  function clearCanvas() {
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = 'black'; // MNIST dataset uses white digits on black background
    ctx.fillRect(0, 0, 280, 280);
  }

  // ========== Drawing Functions ==========
  function startDrawing(e: MouseEvent) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    
    // Setup drawing style to match MNIST
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 20; // Thick line simulates pen/marker width
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round'; // Smooth line joints
    ctx.beginPath();
    
    // Calculate mouse position relative to canvas
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    ctx.moveTo(x, y);
    
    // Handle continuous drawing
    function draw(e: MouseEvent) {
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      ctx.lineTo(x, y);
      ctx.stroke();
    }
    
    // Cleanup function to remove event listeners
    function stopDrawing() {
      canvas.removeEventListener('mousemove', draw);
      canvas.removeEventListener('mouseup', stopDrawing);
      canvas.removeEventListener('mouseleave', stopDrawing);
    }
    
    // Attach drawing event listeners
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing); // Stop if mouse leaves canvas
  }

</script>

<!-- HTML Template Section -->
<!-- This is where we define the component's structure -->
<div class="mnist-trainer">
  <h2>MNIST CNN Trainer</h2>
  
  <div class="controls">
    <div class="training-params">
      <label>
        Epochs:
        <!-- bind:value creates two-way data binding with the epochs variable -->
        <!-- When input changes, epochs variable updates automatically -->
        <input type="number" bind:value={epochs} min="1" max="50" disabled={isTraining}>
      </label>
      <label>
        Batch Size:
        <input type="number" bind:value={batchSize} min="32" max="1024" step="32" disabled={isTraining}>
      </label>
    </div>
    
    <!-- on:click attaches event handler -->
    <!-- disabled attribute is reactive - updates when conditions change -->
    <button on:click={startTraining} disabled={isTraining || isLoading || !data}>
      <!-- Conditional rendering with ternary operator -->
      {isTraining ? `Training... (${currentEpoch}/${epochs})` : 'Start Training'}
    </button>
  </div>
  
  <!-- {#if} is Svelte's conditional rendering syntax -->
  <!-- This section only renders if model exists and training is complete -->
  {#if model && !isTraining}
    <div class="prediction-area">
      <h3>Draw a digit to predict:</h3>
      <!-- bind:this gets a reference to the DOM element -->
      <canvas 
        bind:this={canvas}
        width="280" 
        height="280"
        on:mousedown={startDrawing}
      ></canvas>
      <div class="canvas-controls">
        <button on:click={clearCanvas}>Clear</button>
        <button on:click={predictDigit}>Predict</button>
      </div>
    </div>
  {/if}
  
  <div class="status">
    {#if isTraining}
      <p>Loss: {loss.toFixed(4)} | Accuracy: {(accuracy * 100).toFixed(2)}%</p>
    {/if}
  </div>
  
  <div class="logs">
    <h3>Logs:</h3>
    <div class="log-container">
      <!-- {#each} is Svelte's loop syntax -->
      <!-- It iterates over the logs array -->
      {#each logs as log}
        <div class="log-entry">{log}</div>
      {/each}
    </div>
  </div>
</div>

<!-- Style Section -->
<!-- Styles are scoped to this component by default -->
<!-- They won't affect other components -->
<style>
  :global(body) {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  }

  .mnist-trainer {
    max-width: 1000px;
    margin: 0 auto;
    padding: 40px 20px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    min-height: 100vh;
  }

  h2 {
    text-align: center;
    font-size: 2.5rem;
    color: #2c3e50;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
  }

  .controls {
    background: white;
    border-radius: 16px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    margin-bottom: 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 30px;
    flex-wrap: wrap;
  }

  .training-params {
    display: flex;
    gap: 25px;
  }

  label {
    display: flex;
    flex-direction: column;
    gap: 8px;
    color: #5a6c7d;
    font-weight: 500;
    font-size: 0.9rem;
  }

  input {
    padding: 12px 16px;
    border: 2px solid #e1e8ed;
    border-radius: 8px;
    font-size: 16px;
    transition: all 0.3s ease;
    background: #f8f9fa;
    min-width: 120px;
  }

  input:focus {
    outline: none;
    border-color: #667eea;
    background: white;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
  }

  input:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    background: #e9ecef;
  }

  button {
    padding: 14px 28px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.35);
    position: relative;
    overflow: hidden;
  }

  button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
  }

  button:hover::before {
    width: 300px;
    height: 300px;
  }

  button:disabled {
    background: linear-gradient(135deg, #868e96 0%, #495057 100%);
    cursor: not-allowed;
    box-shadow: none;
  }

  button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
  }

  button:active:not(:disabled) {
    transform: translateY(0);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.35);
  }

  .prediction-area {
    background: white;
    border-radius: 16px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    margin-bottom: 30px;
    opacity: 0;
    animation: fadeIn 0.5s ease forwards;
  }

  .prediction-area h3 {
    color: #2c3e50;
    margin-bottom: 20px;
    font-size: 1.3rem;
  }

  canvas {
    border: 3px solid #e1e8ed;
    border-radius: 12px;
    cursor: crosshair;
    display: block;
    margin: 20px auto;
    background: #1a1a1a;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
  }

  canvas:hover {
    border-color: #667eea;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
  }

  .canvas-controls {
    display: flex;
    gap: 15px;
    justify-content: center;
  }

  .canvas-controls button {
    padding: 10px 24px;
    font-size: 14px;
  }

  .canvas-controls button:first-child {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    box-shadow: 0 4px 15px rgba(245, 87, 108, 0.35);
  }

  .canvas-controls button:first-child:hover:not(:disabled) {
    box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4);
  }

  .status {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    margin-bottom: 30px;
    text-align: center;
    min-height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .status p {
    margin: 0;
    font-size: 1.1rem;
    color: #2c3e50;
    font-weight: 500;
  }

  .logs {
    background: white;
    border-radius: 16px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
  }

  .logs h3 {
    color: #2c3e50;
    margin-bottom: 20px;
    font-size: 1.3rem;
  }

  .log-container {
    background: #f8f9fa;
    border: 1px solid #e1e8ed;
    border-radius: 12px;
    padding: 20px;
    max-height: 250px;
    overflow-y: auto;
    font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
    scrollbar-width: thin;
    scrollbar-color: #667eea #f8f9fa;
  }

  .log-container::-webkit-scrollbar {
    width: 8px;
  }

  .log-container::-webkit-scrollbar-track {
    background: #f8f9fa;
    border-radius: 4px;
  }

  .log-container::-webkit-scrollbar-thumb {
    background: #667eea;
    border-radius: 4px;
  }

  .log-container::-webkit-scrollbar-thumb:hover {
    background: #764ba2;
  }

  .log-entry {
    padding: 8px 12px;
    font-size: 14px;
    color: #495057;
    border-left: 3px solid transparent;
    margin-bottom: 4px;
    transition: all 0.2s ease;
    opacity: 0;
    animation: slideIn 0.3s ease forwards;
  }

  .log-entry:hover {
    background: #e9ecef;
    border-left-color: #667eea;
    border-radius: 4px;
  }

  @keyframes fadeIn {
    to {
      opacity: 1;
    }
  }

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateX(-20px);
    }
    to {
      opacity: 1;
      transform: translateX(0);
    }
  }

  @media (max-width: 768px) {
    .controls {
      flex-direction: column;
      align-items: stretch;
    }

    .training-params {
      flex-direction: column;
      gap: 15px;
      width: 100%;
    }

    .training-params label {
      width: 100%;
    }

    button {
      width: 100%;
    }

    canvas {
      width: 100%;
      max-width: 280px;
      height: auto;
      aspect-ratio: 1;
    }
  }
</style>