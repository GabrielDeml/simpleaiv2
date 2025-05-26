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
  .mnist-trainer {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
  }

  .controls {
    margin: 20px 0;
    display: flex;
    gap: 20px;
    align-items: center;
  }

  .training-params {
    display: flex;
    gap: 15px;
  }

  label {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  input {
    padding: 5px;
    border: 1px solid #ccc;
    border-radius: 4px;
  }

  button {
    padding: 10px 20px;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
  }

  /* :disabled pseudo-class for disabled state */
  button:disabled {
    background-color: #6c757d;
    cursor: not-allowed;
  }

  /* :not() pseudo-class to exclude disabled buttons from hover */
  button:hover:not(:disabled) {
    background-color: #0056b3;
  }

  .prediction-area {
    margin: 30px 0;
  }

  canvas {
    border: 2px solid #333;
    cursor: crosshair;
    display: block;
    margin: 10px 0;
  }

  .canvas-controls {
    display: flex;
    gap: 10px;
  }

  .status {
    margin: 20px 0;
    font-weight: bold;
  }

  .logs {
    margin-top: 30px;
  }

  .log-container {
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 10px;
    max-height: 200px;
    overflow-y: auto; /* Scrollable when content exceeds height */
  }

  .log-entry {
    padding: 2px 0;
    font-family: monospace;
    font-size: 14px;
  }
</style>