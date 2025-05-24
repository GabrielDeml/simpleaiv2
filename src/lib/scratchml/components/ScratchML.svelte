<script lang="ts">
  import { onMount } from 'svelte';
  import * as tf from '@tensorflow/tfjs';
  import LayerPalette from './LayerPalette.svelte';
  import ModelCanvas from './ModelCanvas.svelte';
  import LayerConfig from './LayerConfig.svelte';
  import DemoButton from './DemoButton.svelte';
  import { modelStore } from '../modelStore';
  import { ModelCompiler } from '../modelCompiler';
  import { MnistData } from '$lib/mnist/dataLoader';
  import type { CompiledModel, TrainingConfig } from '../types';
  
  const { layers, connections, isValidModel } = modelStore;
  
  let configuringLayer: string | null = null;
  let compiledModel: CompiledModel | null = null;
  let isTraining = false;
  let trainingProgress = 0;
  let currentEpoch = 0;
  let loss = 0;
  let accuracy = 0;
  let validationErrors: string[] = [];
  
  // Drawing canvas for testing
  let drawCanvas: HTMLCanvasElement;
  let drawCtx: CanvasRenderingContext2D;
  let isDrawing = false;
  let prediction: number | null = null;
  
  // Training config
  let trainingConfig: TrainingConfig = {
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001,
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  };
  
  // MNIST data
  let mnistData: MnistData | null = null;
  
  async function compileModel() {
    try {
      validationErrors = ModelCompiler.validateModel($layers, $connections);
      
      if (validationErrors.length === 0) {
        compiledModel = ModelCompiler.compile($layers, $connections);
        
        // Compile for training
        const optimizer = tf.train[trainingConfig.optimizer](trainingConfig.learningRate);
        compiledModel.model.compile({
          optimizer,
          loss: trainingConfig.loss,
          metrics: trainingConfig.metrics
        });
        
        console.log('Model compiled successfully!', {
          params: compiledModel.totalParams,
          inputShape: compiledModel.inputShape,
          outputShape: compiledModel.outputShape
        });
      }
    } catch (error) {
      validationErrors = [error instanceof Error ? error.message : 'Unknown error'];
      compiledModel = null;
    }
  }
  
  async function startTraining() {
    if (!compiledModel || isTraining) return;
    
    isTraining = true;
    
    // Load MNIST data if not already loaded
    if (!mnistData) {
      mnistData = new MnistData();
      await mnistData.load();
    }
    
    // Get training data
    const batchSize = trainingConfig.batchSize;
    const trainData = mnistData.getTrainData();
    
    // Train the model
    await compiledModel.model.fit(trainData.xs, trainData.ys, {
      batchSize,
      epochs: trainingConfig.epochs,
      shuffle: true,
      validationSplit: 0.1,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          currentEpoch = epoch + 1;
          loss = logs?.loss || 0;
          accuracy = logs?.acc || 0;
          trainingProgress = ((epoch + 1) / trainingConfig.epochs) * 100;
        }
      }
    });
    
    isTraining = false;
  }
  
  function stopTraining() {
    // TensorFlow.js doesn't have a direct way to stop training
    // This would require implementing a custom callback
    isTraining = false;
  }
  
  function clearCanvas() {
    if (drawCtx) {
      drawCtx.fillStyle = 'black';
      drawCtx.fillRect(0, 0, 280, 280);
      prediction = null;
    }
  }
  
  async function predictDrawing() {
    if (!compiledModel || !drawCanvas) return;
    
    // Get image data and preprocess
    const imageData = drawCtx.getImageData(0, 0, 280, 280);
    
    // Create a temporary canvas for resizing
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d')!;
    
    // Draw scaled down image
    tempCtx.drawImage(drawCanvas, 0, 0, 28, 28);
    const scaledData = tempCtx.getImageData(0, 0, 28, 28);
    
    // Convert to tensor
    const input = tf.browser.fromPixels(scaledData, 1)
      .expandDims(0)
      .div(255.0);
    
    // Predict
    const output = compiledModel.model.predict(input) as tf.Tensor;
    const predictions = await output.data();
    prediction = predictions.indexOf(Math.max(...predictions));
    
    // Cleanup
    input.dispose();
    output.dispose();
  }
  
  function handleCanvasConfigureLayer(event: CustomEvent) {
    configuringLayer = event.detail;
  }
  
  function initDrawingCanvas() {
    if (drawCanvas) {
      drawCtx = drawCanvas.getContext('2d')!;
      drawCtx.fillStyle = 'black';
      drawCtx.fillRect(0, 0, 280, 280);
      drawCtx.strokeStyle = 'white';
      drawCtx.lineWidth = 20;
      drawCtx.lineCap = 'round';
      drawCtx.lineJoin = 'round';
    }
  }
  
  function startDrawing(event: MouseEvent) {
    isDrawing = true;
    const rect = drawCanvas.getBoundingClientRect();
    drawCtx.beginPath();
    drawCtx.moveTo(event.clientX - rect.left, event.clientY - rect.top);
  }
  
  function draw(event: MouseEvent) {
    if (!isDrawing) return;
    const rect = drawCanvas.getBoundingClientRect();
    drawCtx.lineTo(event.clientX - rect.left, event.clientY - rect.top);
    drawCtx.stroke();
  }
  
  function stopDrawing() {
    isDrawing = false;
  }
  
  onMount(() => {
    initDrawingCanvas();
  });
</script>

<div class="h-screen flex flex-col bg-gray-100 dark:bg-gray-900">
  <div class="px-6 py-4 bg-white dark:bg-gray-800 shadow-sm flex items-center justify-between">
    <div>
      <h1 class="text-2xl font-bold">Scratch for ML</h1>
      <p class="text-gray-600 dark:text-gray-400">Build neural networks visually</p>
    </div>
    <DemoButton />
  </div>
  
  <div class="flex-1 flex overflow-hidden">
    <LayerPalette />
    
    <div class="flex-1 flex flex-col">
      <ModelCanvas on:configureLayer={handleCanvasConfigureLayer} />
      
      <div class="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 p-4 flex gap-4 overflow-x-auto min-h-[250px]">
        <div class="flex-1 min-w-[300px]">
          <h3 class="text-lg font-semibold mb-3">Model Controls</h3>
          
          {#if validationErrors.length > 0}
            <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded p-3 mb-3">
              {#each validationErrors as error}
                <p class="text-sm text-red-600 dark:text-red-400">{error}</p>
              {/each}
            </div>
          {/if}
          
          <div class="space-y-3">
            <button
              on:click={compileModel}
              disabled={!$isValidModel || isTraining}
              class="px-4 py-2 rounded font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed bg-blue-500 text-white hover:bg-blue-600"
            >
              Compile Model
            </button>
            
            {#if compiledModel}
              <div class="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <p>Total parameters: {compiledModel.totalParams.toLocaleString()}</p>
                <p>Input shape: [{compiledModel.inputShape.join(', ')}]</p>
                <p>Output shape: [{compiledModel.outputShape.join(', ')}]</p>
              </div>
            {/if}
          </div>
          
          {#if compiledModel}
            <div class="mt-4 space-y-3">
              <h4 class="text-sm font-semibold mb-2">Training Configuration</h4>
              
              <div class="grid grid-cols-3 gap-3 mb-3">
                <label class="text-sm">
                  Epochs:
                  <input
                    type="number"
                    bind:value={trainingConfig.epochs}
                    min="1"
                    max="100"
                    disabled={isTraining}
                    class="block w-full mt-1 px-2 py-1 text-sm border rounded dark:bg-gray-700 dark:border-gray-600"
                  />
                </label>
                
                <label class="text-sm">
                  Batch Size:
                  <input
                    type="number"
                    bind:value={trainingConfig.batchSize}
                    min="1"
                    max="128"
                    disabled={isTraining}
                    class="block w-full mt-1 px-2 py-1 text-sm border rounded dark:bg-gray-700 dark:border-gray-600"
                  />
                </label>
                
                <label class="text-sm">
                  Learning Rate:
                  <input
                    type="number"
                    bind:value={trainingConfig.learningRate}
                    min="0.0001"
                    max="0.1"
                    step="0.0001"
                    disabled={isTraining}
                    class="block w-full mt-1 px-2 py-1 text-sm border rounded dark:bg-gray-700 dark:border-gray-600"
                  />
                </label>
              </div>
              
              <div class="flex gap-2">
                {#if !isTraining}
                  <button on:click={startTraining} class="px-4 py-2 rounded font-medium transition-colors bg-green-500 text-white hover:bg-green-600">
                    Start Training
                  </button>
                {:else}
                  <button on:click={stopTraining} class="px-4 py-2 rounded font-medium transition-colors bg-red-500 text-white hover:bg-red-600">
                    Stop Training
                  </button>
                {/if}
              </div>
              
              {#if isTraining || trainingProgress > 0}
                <div class="mt-3 space-y-2">
                  <div class="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div class="h-full bg-blue-500 transition-all duration-300" style="width: {trainingProgress}%"></div>
                  </div>
                  <p>Epoch: {currentEpoch}/{trainingConfig.epochs}</p>
                  <p>Loss: {loss.toFixed(4)} | Accuracy: {(accuracy * 100).toFixed(2)}%</p>
                </div>
              {/if}
            </div>
          {/if}
        </div>
        
        {#if compiledModel && compiledModel.outputShape[0] === 10}
          <div class="w-[320px] flex-shrink-0">
            <h3 class="text-lg font-semibold mb-3">Test Your Model</h3>
            <p class="text-sm text-gray-600 mb-2">Draw a digit (0-9)</p>
            
            <canvas
              bind:this={drawCanvas}
              width="280"
              height="280"
              on:mousedown={startDrawing}
              on:mousemove={draw}
              on:mouseup={stopDrawing}
              on:mouseleave={stopDrawing}
              class="border border-gray-300 dark:border-gray-600 rounded cursor-crosshair bg-black"
            ></canvas>
            
            <div class="flex gap-2 mt-2">
              <button on:click={clearCanvas} class="px-4 py-2 rounded font-medium transition-colors bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600">Clear</button>
              <button on:click={predictDrawing} class="px-4 py-2 rounded font-medium transition-colors bg-blue-500 text-white hover:bg-blue-600">Predict</button>
            </div>
            
            {#if prediction !== null}
              <div class="mt-3 text-center">
                <p class="text-lg font-bold">Prediction: {prediction}</p>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    </div>
  </div>
  
  {#if configuringLayer}
    <div 
      class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      on:click={() => configuringLayer = null}
      on:keydown={(e) => e.key === 'Escape' && (configuringLayer = null)}
      role="button"
      tabindex="0"
    >
      <div 
        class="relative"
        on:click|stopPropagation
        on:keydown|stopPropagation
        role="dialog"
        aria-modal="true"
        tabindex="-1"
      >
        <LayerConfig
          layerId={configuringLayer}
          on:close={() => configuringLayer = null}
        />
      </div>
    </div>
  {/if}
</div>