<script lang="ts">
  import { selectedDataset } from '$lib/nn-designer/stores';
  import type { DatasetType } from '$lib/nn-designer/types';
  
  interface DatasetInfo {
    type: DatasetType;
    name: string;
    shape: string;
    available: boolean;
  }
  
  const datasets: DatasetInfo[] = [
    { type: 'mnist', name: 'MNIST', shape: '28×28', available: true },
    { type: 'cifar10', name: 'CIFAR-10', shape: '32×32×3', available: false },
    { type: 'fashion-mnist', name: 'Fashion-MNIST', shape: '28×28', available: false }
  ];
  
  function selectDataset(type: DatasetType) {
    if (datasets.find(d => d.type === type)?.available) {
      selectedDataset.set(type);
    }
  }
  
  function handleUpload() {
    console.log('Upload custom dataset');
  }
</script>

<div class="dataset-selector">
  <h3>DATASET</h3>
  
  <div class="dataset-list">
    {#each datasets as dataset}
      <label 
        class="dataset-option"
        class:active={$selectedDataset === dataset.type}
        class:disabled={!dataset.available}
      >
        <input
          type="radio"
          name="dataset"
          value={dataset.type}
          checked={$selectedDataset === dataset.type}
          disabled={!dataset.available}
          on:change={() => selectDataset(dataset.type)}
        />
        <span class="radio-icon"></span>
        <span class="dataset-name">{dataset.name}</span>
        <span class="dataset-shape">{dataset.shape}</span>
      </label>
    {/each}
    
    <button class="upload-btn" on:click={handleUpload}>
      📁 Upload Custom Dataset
    </button>
  </div>
</div>

<style>
  .dataset-selector {
    padding: 20px 24px;
    border-top: 1px solid #262626;
  }
  
  h3 {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.5px;
    color: #737373;
    margin: 0 0 16px 0;
  }
  
  .dataset-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .dataset-option {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 20px;
    background: #171717;
    border: 1px solid #262626;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
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
    width: 10px;
    height: 10px;
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
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: white;
  }
  
  .dataset-name {
    flex: 1;
    font-size: 13px;
    color: #ffffff;
  }
  
  .dataset-option.disabled .dataset-name {
    color: #a3a3a3;
  }
  
  .dataset-shape {
    font-size: 11px;
    color: #737373;
  }
  
  .dataset-option.disabled .dataset-shape {
    color: #525252;
  }
  
  .upload-btn {
    padding: 10px;
    background: none;
    border: 1px dashed #525252;
    border-radius: 8px;
    color: #737373;
    font-size: 12px;
    cursor: pointer;
    opacity: 0.5;
    transition: all 0.2s;
  }
  
  .upload-btn:hover {
    opacity: 1;
    border-color: #737373;
  }
</style>