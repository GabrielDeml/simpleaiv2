<script lang="ts">
  /**
   * NetworkCanvas Component
   * 
   * Purpose: Main visual editor for designing neural networks. Displays layers
   * as connected nodes in a vertical flow, handles drag-and-drop layer insertion,
   * and provides interactive editing capabilities.
   * 
   * Key features:
   * - Visual representation of neural network architecture
   * - Drag-and-drop zones between layers for insertion
   * - Layer selection and deletion
   * - Dynamic styling based on layer type
   * - Automatic output layer detection and highlighting
   */
  
  import { layers, selectedLayerId, addLayer, removeLayer } from '$lib/nn-designer/stores';
  import { layerDefinitions } from '$lib/nn-designer/layerDefinitions';
  import type { LayerConfig, LayerType } from '$lib/nn-designer/types';
  
  // Tracks which layer's drop zone is being hovered during drag operations
  let dragOverLayerId: string | null = null;
  
  /**
   * Handles drag over events on the main canvas
   * Prevents default to allow dropping and sets the visual drop effect
   */
  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    if (e.dataTransfer) {
      e.dataTransfer.dropEffect = 'copy';
    }
  }
  
  /**
   * Handles dropping a new layer onto the canvas
   * @param e - The drop event containing layer type data
   * @param afterLayerId - Optional ID of layer to insert after
   * 
   * Creates a new layer instance from the dropped layer type and adds it
   * to the network. If afterLayerId is provided, inserts after that layer,
   * otherwise appends to the end.
   */
  function handleDrop(e: DragEvent, afterLayerId?: string) {
    e.preventDefault();
    
    // If dropping on a specific zone, stop event propagation
    if (afterLayerId) {
      e.stopPropagation();
    }
    
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
  
  /**
   * Handles drag over events on layer drop zones
   * Sets visual feedback for the specific drop zone being hovered
   */
  function handleLayerDragOver(e: DragEvent, layerId: string) {
    e.preventDefault();
    dragOverLayerId = layerId;
  }
  
  /**
   * Clears drag over state when leaving a drop zone
   */
  function handleLayerDragLeave() {
    dragOverLayerId = null;
  }
  
  /**
   * Selects a layer for editing in the properties panel
   * Updates the global selectedLayerId store
   */
  function selectLayer(layerId: string) {
    selectedLayerId.set(layerId);
  }
  
  /**
   * Removes a layer from the network
   * Input layers cannot be deleted (enforced in UI)
   */
  function deleteLayer(layerId: string) {
    removeLayer(layerId);
  }
  
  /**
   * Generates display information for a layer node
   * @param layer - The layer configuration
   * @returns Object with formatted name, subtitle, color, and icon
   * 
   * Creates human-readable subtitles based on layer type and parameters.
   * Dense layers show unit count in the title for quick reference.
   */
  function getLayerDisplayInfo(layer: LayerConfig) {
    const definition = layerDefinitions[layer.type];
    let subtitle = '';
    
    // Generate layer-specific subtitle with key parameters
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
      case 'output':
        subtitle = `${layer.params.activation} activation`;
        break;
    }
    
    return {
      name: (layer.type === 'dense' || layer.type === 'output') ? `${layer.name} (${layer.params.units})` : layer.name,
      subtitle,
      color: definition.color,
      icon: definition.icon
    };
  }
  
  /**
   * Determines if a layer should be displayed as the output layer
   * @param layer - The layer to check
   * @param index - The layer's position in the network
   * @returns true if this is the last layer and it's a dense layer
   * 
   * Output layers get special styling (orange color) to indicate
   * they produce the final network predictions.
   */
  function isOutputLayer(layer: LayerConfig, index: number): boolean {
    return layer.type === 'output' || (index === $layers.length - 1 && layer.type === 'dense');
  }
</script>

<!-- Main canvas area that accepts layer drops -->
<div class="network-canvas" on:dragover={handleDragOver} on:drop={(e) => handleDrop(e)}>
  <!-- Container for the vertical flow of layers -->
  <div class="network-flow">
    <!-- Background line connecting all layers visually -->
    <div class="flow-line"></div>
    
    <!-- Iterate through all layers in the network -->
    {#each $layers as layer, index (layer.id)}
      <!-- Pre-compute display values for this layer -->
      {@const displayInfo = getLayerDisplayInfo(layer)}
      {@const isOutput = isOutputLayer(layer, index)}
      {@const isSelected = $selectedLayerId === layer.id}
      
      <!-- Container for layer node and its connection elements -->
      <div class="layer-container">
        <!-- 
          The layer node itself - clickable, selectable, with dynamic styling
          - Color changes based on whether it's an output layer
          - Shows selection ring when selected
          - Includes edit/delete actions
        -->
        <div
          class="layer-node"
          class:selected={isSelected}
          class:output={isOutput}
          style="--layer-color: {isOutput ? '#f59e0b' : displayInfo.color}"
          on:click={() => selectLayer(layer.id)}
          role="button"
          tabindex="0"
        >
          <!-- Animated selection indicator -->
          {#if isSelected}
            <div class="selection-ring"></div>
          {/if}
          
          <!-- Colored accent bar on left side -->
          <div class="layer-accent"></div>
          
          <!-- Layer information display -->
          <div class="layer-content">
            <div class="layer-title">{displayInfo.name}</div>
            <div class="layer-subtitle">{displayInfo.subtitle}</div>
          </div>
          
          <!-- Action buttons (edit/delete) -->
          <div class="layer-actions">
            <!-- Delete button - hidden for input layers -->
            {#if layer.type !== 'input'}
              <button class="delete-btn" on:click|stopPropagation={() => deleteLayer(layer.id)}>
                ×
              </button>
            {/if}
            <!-- Edit button - highlighted when layer is selected -->
            <button class="edit-btn" class:active={isSelected}>
              Edit
            </button>
          </div>
        </div>
        
        <!-- Connection elements between layers -->
        {#if index < $layers.length - 1}
          <!-- Visual dots connecting layers -->
          <div class="connection">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
          </div>
          
          <!-- 
            Drop zone for inserting new layers
            - Appears between each pair of layers
            - Shows visual feedback when dragging over
            - Clicking could also insert a layer here
          -->
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
    
    <!-- Empty state when no layers exist -->
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
    opacity: 0.6;
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