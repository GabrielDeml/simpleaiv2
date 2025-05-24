<!-- 
  This is a Svelte component for training an MNIST digit recognition model.
  Svelte components have three sections:
  1. <script> - JavaScript/TypeScript logic
  2. HTML template - The component's markup
  3. <style> - Component-scoped CSS
-->

<script lang="ts">
  import * as tf from '@tensorflow/tfjs';
  import { onMount } from 'svelte'; // Svelte lifecycle function that runs after component is mounted to DOM
  import { MnistData } from '../mnist/dataLoader';
  import { createCNNModel } from '../mnist/model';

  // State variables - In Svelte, these are reactive by default
  // When these change, the UI automatically updates
  let model: tf.Sequential | null = null; // The trained TensorFlow.js model
  let data: MnistData | null = null; // The MNIST dataset
  let isTraining = false; // Flag to track if model is currently training
  let isLoading = false; // Flag to track if data is loading
  let epochs = 10; // Number of training epochs (full passes through the dataset)
  let batchSize = 512; // Number of samples to process at once during training
  let currentEpoch = 0; // Current training epoch for progress display
  let loss = 0; // Current training loss value
  let accuracy = 0; // Current training accuracy
  let logs: string[] = []; // Array to store log messages
  let canvas: HTMLCanvasElement; // Reference to the drawing canvas element

  // onMount runs after the component is added to the DOM
  // It's like useEffect(() => {}, []) in React
  onMount(() => {
    loadData(); // Load MNIST data when component mounts
  });

  // Async function to load the MNIST dataset
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

  // Main training function
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

  // Predict digit from canvas drawing
  async function predictDigit() {
    if (!model || !canvas) return;
    
    // Get canvas 2D context for reading pixel data
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.getImageData(0, 0, 280, 280);
    
    // Convert canvas image to tensor and preprocess
    const input = tf.browser.fromPixels(imageData, 1) // 1 = grayscale
      .resizeNearestNeighbor([28, 28]) // Resize to MNIST size
      .toFloat()
      .div(255.0) // Normalize pixel values to 0-1
      .expandDims(0); // Add batch dimension [1, 28, 28, 1]
    
    // Run prediction
    const prediction = model.predict(input) as tf.Tensor;
    const probabilities = await prediction.data();
    // Find index of highest probability
    const predictedClass = probabilities.indexOf(Math.max(...probabilities));
    
    logs = [...logs, `Predicted digit: ${predictedClass}`];
    
    // Clean up tensors
    input.dispose();
    prediction.dispose();
  }

  // Clear the drawing canvas
  function clearCanvas() {
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = 'black'; // MNIST uses white on black
    ctx.fillRect(0, 0, 280, 280);
  }

  // Handle mouse drawing on canvas
  function startDrawing(e: MouseEvent) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    ctx.strokeStyle = 'white'; // Draw in white
    ctx.lineWidth = 20; // Thick line for digit drawing
    ctx.lineCap = 'round'; // Round line endings
    ctx.beginPath();
    
    // Get mouse position relative to canvas
    const rect = canvas.getBoundingClientRect();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
    
    // Inner function to handle mouse movement
    function draw(e: MouseEvent) {
      ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
      ctx.stroke();
    }
    
    // Clean up event listeners when drawing stops
    function stopDrawing() {
      canvas.removeEventListener('mousemove', draw);
      canvas.removeEventListener('mouseup', stopDrawing);
      canvas.removeEventListener('mouseleave', stopDrawing);
    }
    
    // Add event listeners for drawing
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);
  }

  // Another onMount to clear canvas after it's rendered
  onMount(() => {
    clearCanvas();
  });
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