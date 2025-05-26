<script lang="ts">
  import { layers, selectedLayerId, addLayer, removeLayer } from '$lib/nn-designer/stores';
  import { layerDefinitions } from '$lib/nn-designer/layerDefinitions';
  import type { LayerConfig, LayerType } from '$lib/nn-designer/types';
  
  let dragOverLayerId: string | null = null;
  
  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    if (e.dataTransfer) {
      e.dataTransfer.dropEffect = 'copy';
    }
  }
  
  function handleDrop(e: DragEvent, afterLayerId?: string) {
    e.preventDefault();
    const layerType = e.dataTransfer?.getData('layerType') as LayerType;
    
    if (layerType && layerDefinitions[layerType]) {
      const definition = layerDefinitions[layerType];
      const newLayer: LayerConfig = {
        id: `${layerType}-${Date.now()}`,
        type: layerType,
        name: definition.displayName,
        params: { ...definition.defaultParams }
      };
      
      addLayer(newLayer, afterLayerId);
    }
    
    dragOverLayerId = null;
  }
  
  function handleLayerDragOver(e: DragEvent, layerId: string) {
    e.preventDefault();
    dragOverLayerId = layerId;
  }
  
  function handleLayerDragLeave() {
    dragOverLayerId = null;
  }
  
  function selectLayer(layerId: string) {
    selectedLayerId.set(layerId);
  }
  
  function deleteLayer(layerId: string) {
    removeLayer(layerId);
  }
  
  function getLayerDisplayInfo(layer: LayerConfig) {
    const definition = layerDefinitions[layer.type];
    let subtitle = '';
    
    switch (layer.type) {
      case 'input':
        subtitle = `Shape: (${layer.params.shape.join(', ')})`;
        break;
      case 'dense':
        subtitle = `${layer.params.activation} activation`;
        break;
      case 'conv2d':
        subtitle = `Filters: ${layer.params.filters}, Kernel: ${layer.params.kernelSize}`;
        break;
      case 'maxpooling2d':
        subtitle = `Pool size: ${layer.params.poolSize}`;
        break;
      case 'dropout':
        subtitle = `Rate: ${layer.params.rate}`;
        break;
      case 'flatten':
        subtitle = 'Flatten layer';
        break;
    }
    
    return {
      name: layer.type === 'dense' ? `${layer.name} (${layer.params.units})` : layer.name,
      subtitle,
      color: definition.color,
      icon: definition.icon
    };
  }
  
  // Determine if this is an output layer
  function isOutputLayer(layer: LayerConfig, index: number): boolean {
    return index === $layers.length - 1 && layer.type === 'dense';
  }
</script>

<div class="network-canvas" on:dragover={handleDragOver} on:drop={(e) => handleDrop(e)}>
  <div class="network-flow">
    <div class="flow-line"></div>
    
    {#each $layers as layer, index (layer.id)}
      {@const displayInfo = getLayerDisplayInfo(layer)}
      {@const isOutput = isOutputLayer(layer, index)}
      {@const isSelected = $selectedLayerId === layer.id}
      
      <div class="layer-container">
        <div
          class="layer-node"
          class:selected={isSelected}
          class:output={isOutput}
          style="--layer-color: {isOutput ? '#f59e0b' : displayInfo.color}"
          on:click={() => selectLayer(layer.id)}
          role="button"
          tabindex="0"
        >
          {#if isSelected}
            <div class="selection-ring"></div>
          {/if}
          
          <div class="layer-accent"></div>
          <div class="layer-content">
            <div class="layer-title">{isOutput ? 'Output' : displayInfo.name}</div>
            <div class="layer-subtitle">{displayInfo.subtitle}</div>
          </div>
          
          <div class="layer-actions">
            {#if layer.type !== 'input'}
              <button class="delete-btn" on:click|stopPropagation={() => deleteLayer(layer.id)}>
                ×
              </button>
            {/if}
            <button class="edit-btn" class:active={isSelected}>
              Edit
            </button>
          </div>
        </div>
        
        {#if index < $layers.length - 1}
          <div class="connection">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
          </div>
          
          <div
            class="add-layer-btn"
            class:drag-over={dragOverLayerId === layer.id}
            on:dragover={(e) => handleLayerDragOver(e, layer.id)}
            on:dragleave={handleLayerDragLeave}
            on:drop={(e) => handleDrop(e, layer.id)}
            role="button"
            tabindex="0"
          >
            +
          </div>
        {/if}
      </div>
    {/each}
    
    {#if $layers.length === 0}
      <div class="empty-state">
        <p>Drag layers here to build your network</p>
      </div>
    {/if}
  </div>
</div>

<style>
  .network-canvas {
    flex: 1;
    padding: 80px 0;
    overflow-y: auto;
    display: flex;
    justify-content: center;
    min-height: 600px;
  }
  
  .network-flow {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
  }
  
  .flow-line {
    position: absolute;
    left: 50%;
    top: 60px;
    bottom: 60px;
    width: 2px;
    background: #262626;
    transform: translateX(-50%);
    z-index: 0;
  }
  
  .layer-container {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  
  .layer-node {
    position: relative;
    width: 200px;
    height: 60px;
    background: #171717;
    border: 1.5px solid var(--layer-color);
    border-radius: 8px;
    display: flex;
    align-items: center;
    padding: 0 16px;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  }
  
  .layer-node:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
  }
  
  .layer-node.selected {
    border-width: 2px;
  }
  
  .selection-ring {
    position: absolute;
    inset: -8px;
    border: 2px dashed var(--layer-color);
    border-radius: 12px;
    opacity: 0.5;
    animation: rotate 10s linear infinite;
  }
  
  @keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
  
  .layer-accent {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: var(--layer-color);
    border-radius: 8px 0 0 8px;
  }
  
  .layer-content {
    flex: 1;
    margin-left: 12px;
  }
  
  .layer-title {
    font-size: 15px;
    font-weight: 500;
    color: #ffffff;
  }
  
  .layer-subtitle {
    font-size: 12px;
    color: #a3a3a3;
    margin-top: 2px;
  }
  
  .layer-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .delete-btn {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #ef4444;
    border: none;
    color: white;
    font-size: 12px;
    cursor: pointer;
    opacity: 0.6;
    transition: opacity 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .delete-btn:hover {
    opacity: 1;
  }
  
  .edit-btn {
    padding: 4px 12px;
    border-radius: 4px;
    background: #262626;
    border: none;
    color: #a3a3a3;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .edit-btn.active {
    background: var(--layer-color);
    color: white;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
  }
  
  .connection {
    height: 30px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 4px;
  }
  
  .dot {
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: #525252;
  }
  
  .add-layer-btn {
    width: 80px;
    height: 28px;
    border-radius: 14px;
    background: #0a0a0a;
    border: 1px dashed #333333;
    color: #737373;
    font-size: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s;
    margin-bottom: 12px;
  }
  
  .add-layer-btn:hover {
    border-color: #525252;
    color: #a3a3a3;
  }
  
  .add-layer-btn.drag-over {
    border-color: #22c55e;
    background: rgba(34, 197, 94, 0.1);
    color: #22c55e;
  }
  
  .empty-state {
    padding: 40px;
    text-align: center;
    color: #525252;
  }
</style>