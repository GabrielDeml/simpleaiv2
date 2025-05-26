<script lang="ts">
  /**
   * TrainingProgress Component
   * 
   * Purpose: Modal overlay that displays real-time training progress,
   * metrics, and charts. Shows during model training and allows users
   * to monitor and control the training process.
   * 
   * Key features:
   * - Real-time progress bar and epoch tracking
   * - Live metrics display (loss, accuracy, validation metrics)
   * - Interactive charts via MetricsChart component
   * - Stop training capability
   * - Modal interface with backdrop click to close
   */
  
  import { createEventDispatcher } from 'svelte';
  import { isTraining, currentEpoch, trainingHistory, trainingConfig } from '$lib/nn-designer/stores';
  import MetricsChart from './MetricsChart.svelte';
  import { trainingManager } from '$lib/nn-designer/trainingManager';
  
  // Event dispatcher for parent component communication
  const dispatch = createEventDispatcher();
  
  /**
   * Handles modal close action
   * Dispatches 'close' event to parent component
   */
  function handleClose() {
    dispatch('close');
  }
  
  /**
   * Stops the training process
   * - Updates global training state
   * - Calls training manager to halt TensorFlow.js training
   */
  function handleStop() {
    isTraining.set(false);
    trainingManager.stopTraining();
  }
  
  // Reactive calculations for UI display
  // Progress percentage based on current epoch
  $: progress = ($currentEpoch / $trainingConfig.epochs) * 100;
  
  // Latest metric values from training history arrays
  $: latestLoss = $trainingHistory.loss[$trainingHistory.loss.length - 1] || 0;
  $: latestAccuracy = $trainingHistory.accuracy[$trainingHistory.accuracy.length - 1] || 0;
  $: latestValLoss = $trainingHistory.valLoss[$trainingHistory.valLoss.length - 1] || 0;
  $: latestValAccuracy = $trainingHistory.valAccuracy[$trainingHistory.valAccuracy.length - 1] || 0;
</script>

<!-- Modal overlay - clicking backdrop closes modal -->
<div class="modal-overlay" on:click={handleClose}>
  <!-- Modal content - stop propagation prevents backdrop click -->
  <div class="modal-content" on:click|stopPropagation>
    <!-- Header with title and close button -->
    <div class="modal-header">
      <h2>Training Progress</h2>
      <button class="close-btn" on:click={handleClose}>×</button>
    </div>
    
    <!-- Progress tracking section -->
    <div class="progress-section">
      <!-- Epoch counter and percentage -->
      <div class="progress-header">
        <span>Epoch {$currentEpoch} / {$trainingConfig.epochs}</span>
        <span>{progress.toFixed(0)}%</span>
      </div>
      <!-- Visual progress bar -->
      <div class="progress-bar">
        <div class="progress-fill" style="width: {progress}%"></div>
      </div>
    </div>
    
    <!-- Current metrics grid - 4 columns -->
    <div class="metrics-grid">
      <!-- Training loss -->
      <div class="metric">
        <span class="metric-label">Loss</span>
        <span class="metric-value">{latestLoss.toFixed(4)}</span>
      </div>
      <!-- Training accuracy (converted to percentage) -->
      <div class="metric">
        <span class="metric-label">Accuracy</span>
        <span class="metric-value">{(latestAccuracy * 100).toFixed(1)}%</span>
      </div>
      <!-- Validation loss -->
      <div class="metric">
        <span class="metric-label">Val Loss</span>
        <span class="metric-value">{latestValLoss.toFixed(4)}</span>
      </div>
      <!-- Validation accuracy (converted to percentage) -->
      <div class="metric">
        <span class="metric-label">Val Accuracy</span>
        <span class="metric-value">{(latestValAccuracy * 100).toFixed(1)}%</span>
      </div>
    </div>
    
    <!-- Chart container - shows metrics over time -->
    <div class="chart-container">
      {#if $trainingHistory.loss.length > 0}
        <!-- MetricsChart component renders when data is available -->
        <MetricsChart />
      {:else}
        <!-- Placeholder shown before training starts -->
        <div class="chart-placeholder">
          <p>Training metrics will appear here</p>
        </div>
      {/if}
    </div>
    
    <!-- Action buttons - context-sensitive -->
    <div class="modal-actions">
      {#if $isTraining}
        <!-- Stop button shown during active training -->
        <button class="btn btn-danger" on:click={handleStop}>Stop Training</button>
      {:else}
        <!-- Close button shown when training is complete/stopped -->
        <button class="btn btn-primary" on:click={handleClose}>Close</button>
      {/if}
    </div>
  </div>
</div>

<style>
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }
  
  .modal-content {
    background: #0f0f0f;
    border-radius: 12px;
    padding: 24px;
    width: 600px;
    max-width: 90vw;
    max-height: 80vh;
    overflow-y: auto;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
  }
  
  .modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
  }
  
  .modal-header h2 {
    font-size: 20px;
    font-weight: 500;
    margin: 0;
  }
  
  .close-btn {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    border: none;
    background: #1a1a1a;
    color: #737373;
    font-size: 20px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
  }
  
  .close-btn:hover {
    background: #262626;
    color: #ffffff;
  }
  
  .progress-section {
    margin-bottom: 24px;
  }
  
  .progress-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 14px;
    color: #a3a3a3;
  }
  
  .progress-bar {
    height: 8px;
    background: #262626;
    border-radius: 4px;
    overflow: hidden;
  }
  
  .progress-fill {
    height: 100%;
    background: linear-gradient(to right, #22c55e, #16a34a);
    transition: width 0.3s ease;
  }
  
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }
  
  .metric {
    background: #171717;
    padding: 16px;
    border-radius: 8px;
    text-align: center;
  }
  
  .metric-label {
    display: block;
    font-size: 12px;
    color: #737373;
    margin-bottom: 4px;
  }
  
  .metric-value {
    display: block;
    font-size: 18px;
    font-weight: 500;
    color: #ffffff;
  }
  
  .chart-container {
    background: #171717;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 24px;
    height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .chart-placeholder {
    text-align: center;
    color: #525252;
  }
  
  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
  }
  
  .btn {
    padding: 10px 24px;
    border-radius: 6px;
    border: none;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .btn-primary {
    background: linear-gradient(to right, #22c55e, #16a34a);
    color: white;
  }
  
  .btn-danger {
    background: #ef4444;
    color: white;
  }
  
  .btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  }
</style>