<script lang="ts">
  /**
   * DatasetSelector Component
   * 
   * Purpose: Allows users to select from available datasets for training.
   * Automatically updates the input layer shape when a dataset is selected
   * to match the expected input dimensions.
   * 
   * Key features:
   * - Radio button selection for datasets
   * - Visual dataset shape indicators
   * - Auto-updates input layer on selection
   * - Placeholder for custom dataset upload
   * - Disabled state for unavailable datasets
   */
  
  import { selectedDataset, layers, updateLayer } from '$lib/nn-designer/stores';
  import type { DatasetType } from '$lib/nn-designer/types';
  
  // Dataset configuration interface
  interface DatasetInfo {
    type: DatasetType;
    name: string;
    shape: string;          // Human-readable shape (e.g., "28×28")
    available: boolean;     // Whether dataset is implemented
    inputShape: number[];   // Actual shape array for input layer
  }
  
  // Available datasets with their configurations
  const datasets: DatasetInfo[] = [
    { type: 'mnist', name: 'MNIST', shape: '28×28', available: true, inputShape: [28, 28] },
    { type: 'cifar10', name: 'CIFAR-10', shape: '32×32×3', available: true, inputShape: [32, 32, 3] },
    { type: 'fashion-mnist', name: 'Fashion-MNIST', shape: '28×28', available: true, inputShape: [28, 28] },
    { type: 'imdb', name: 'IMDB Reviews', shape: '200 tokens', available: true, inputShape: [200] },
    { type: 'ag-news', name: 'AG News', shape: '150 tokens', available: true, inputShape: [150] }
  ];
  
  // Collapsible state
  let isExpanded = true;
  
  /**
   * Handles dataset selection
   * @param type - The dataset type to select
   * 
   * - Updates global selected dataset
   * - Automatically adjusts input layer shape to match dataset
   * - Only processes available datasets
   */
  function selectDataset(type: DatasetType) {
    const dataset = datasets.find(d => d.type === type);
    if (dataset?.available) {
      selectedDataset.set(type);
      
      // Auto-adjust input layer shape to match selected dataset
      const currentLayers = $layers;
      if (currentLayers.length > 0 && currentLayers[0].type === 'input') {
        updateLayer(currentLayers[0].id, { shape: dataset.inputShape });
      }
    }
  }
  
  /**
   * Placeholder for custom dataset upload functionality
   * Currently just logs to console
   */
  function handleUpload() {
    console.log('Upload custom dataset');
  }
</script>

<!-- Dataset selector panel -->
<div class="dataset-selector">
  <button class="section-header" on:click={() => isExpanded = !isExpanded}>
    <span class="header-text">DATASET</span>
    <span class="expand-icon" class:expanded={isExpanded}>▼</span>
  </button>
  
  {#if isExpanded}
    <!-- List of available datasets -->
    <div class="dataset-list">
    {#each datasets as dataset}
      <!-- 
        Radio button option for each dataset
        - Custom styled radio button
        - Shows active state with green highlight
        - Disabled state for unavailable datasets
      -->
      <label 
        class="dataset-option"
        class:active={$selectedDataset === dataset.type}
        class:disabled={!dataset.available}
      >
        <!-- Hidden native radio input -->
        <input
          type="radio"
          name="dataset"
          value={dataset.type}
          checked={$selectedDataset === dataset.type}
          disabled={!dataset.available}
          on:change={() => selectDataset(dataset.type)}
        />
        <!-- Custom radio button visual -->
        <span class="radio-icon"></span>
        <!-- Dataset name -->
        <span class="dataset-name">{dataset.name}</span>
        <!-- Dataset shape indicator -->
        <span class="dataset-shape">{dataset.shape}</span>
      </label>
    {/each}
    
      <!-- Upload button (placeholder for future functionality) -->
      <button class="upload-btn" on:click={handleUpload}>
        📁 Upload Custom Dataset
      </button>
    </div>
  {/if}
</div>

<style>
  .dataset-selector {
    border-bottom: 1px solid #262626;
  }

  .section-header {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    background: transparent;
    border: none;
    color: #737373;
    cursor: pointer;
    transition: color 0.2s;
  }

  .section-header:hover {
    color: #a3a3a3;
  }

  .header-text {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.5px;
  }

  .expand-icon {
    font-size: 10px;
    transition: transform 0.2s;
  }

  .expand-icon.expanded {
    transform: rotate(180deg);
  }
  
  .dataset-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 0 24px 16px 24px;
  }
  
  .dataset-option {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    background: #171717;
    border: 1px solid #262626;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .dataset-option.active {
    background: #052e16;
    border-color: #16a34a;
  }
  
  .dataset-option.disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .dataset-option:hover:not(.disabled) {
    border-color: #333333;
  }
  
  .dataset-option input {
    display: none;
  }
  
  .radio-icon {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    border: 1px solid #525252;
    position: relative;
  }
  
  .dataset-option.active .radio-icon {
    background: #22c55e;
    border-color: #22c55e;
  }
  
  .dataset-option.active .radio-icon::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: white;
  }
  
  .dataset-name {
    flex: 1;
    font-size: 12px;
    color: #ffffff;
  }
  
  .dataset-option.disabled .dataset-name {
    color: #a3a3a3;
  }
  
  .dataset-shape {
    font-size: 10px;
    color: #737373;
  }
  
  .dataset-option.disabled .dataset-shape {
    color: #525252;
  }
  
  .upload-btn {
    padding: 8px;
    background: none;
    border: 1px dashed #525252;
    border-radius: 6px;
    color: #737373;
    font-size: 11px;
    cursor: pointer;
    opacity: 0.5;
    transition: all 0.15s;
  }
  
  .upload-btn:hover {
    opacity: 1;
    border-color: #737373;
  }
</style>