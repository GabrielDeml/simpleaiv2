<script lang="ts">
  /**
   * LayerPalette Component
   * 
   * Purpose: Provides a draggable palette of available neural network layers
   * that users can drag onto the NetworkCanvas or click to add to the model.
   * 
   * Key features:
   * - Displays all available layer types from layerDefinitions
   * - Supports drag-and-drop to NetworkCanvas
   * - Click-to-add functionality as an alternative to dragging
   * - Keyboard accessibility (Enter/Space to add layers)
   * - Visual feedback on hover and during drag operations
   */
  
  import { layerDefinitions } from '$lib/nn-designer/layerDefinitions';
  import { addLayer } from '$lib/nn-designer/stores';
  import type { LayerType } from '$lib/nn-designer/types';
  import Tooltip from '$lib/components/Tooltip.svelte';
  import LayerInfoPopup from '$lib/components/LayerInfoPopup.svelte';
  
  
  // Collapsible state
  let isExpanded = true;
  
  // Popup state
  let showInfoPopup = false;
  let selectedInfoLayer: LayerType | null = null;
  
  /**
   * Handles the start of a drag operation
   * @param e - The drag event containing dataTransfer object
   * @param layerType - The type of layer being dragged (input, dense, conv2d, etc.)
   * 
   * Sets up the drag data transfer with the layer type so NetworkCanvas
   * can receive it on drop. The 'copy' effect indicates we're creating
   * a new instance, not moving an existing layer.
   */
  function handleDragStart(e: DragEvent, layerType: LayerType) {
    if (e.dataTransfer) {
      e.dataTransfer.effectAllowed = 'copy';
      e.dataTransfer.setData('layerType', layerType);
    }
  }
  
  /**
   * Handles keyboard interaction for accessibility
   * @param e - Keyboard event
   * @param layerType - The layer type to add when activated
   * 
   * Allows users to add layers using keyboard navigation by pressing
   * Enter or Space when a layer card is focused.
   */
  function handleKeyDown(e: KeyboardEvent, layerType: LayerType) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick(layerType);
    }
  }
  
  /**
   * Cleans up after a drag operation ends
   */
  function handleDragEnd() {
    // Currently no cleanup needed
  }
  
  /**
   * Generates a unique ID for a new layer instance
   * @param type - The layer type
   * @returns A unique ID combining type and timestamp
   * 
   * Using timestamp ensures uniqueness for layers created in quick succession
   */
  function generateLayerId(type: LayerType): string {
    return `${type}-${Date.now()}`;
  }
  
  /**
   * Handles click events to add a layer directly (without dragging)
   * @param layerType - The type of layer to add
   * 
   * Creates a new layer instance with:
   * - Unique ID
   * - Layer type
   * - Display name from layer definition
   * - Default parameters copied from layer definition
   * 
   * The spread operator on defaultParams ensures each layer gets its own
   * parameter object instance, preventing shared state issues.
   */
  function handleClick(layerType: LayerType) {
    const definition = layerDefinitions[layerType];
    const newLayer = {
      id: generateLayerId(layerType),
      type: layerType,
      name: definition.displayName,
      params: { ...definition.defaultParams }
    };
    
    addLayer(newLayer);
  }
  
  /**
   * Shows the detailed info popup for a layer type
   * @param layerType - The type of layer to show info for
   */
  function showLayerInfo(layerType: LayerType) {
    selectedInfoLayer = layerType;
    showInfoPopup = true;
  }
</script>

<!-- Main container for the layer palette sidebar -->
<div class="layer-palette">
  <button class="section-header" on:click={() => isExpanded = !isExpanded}>
    <span class="header-text">LAYERS</span>
    <span class="expand-icon" class:expanded={isExpanded}>▼</span>
  </button>
  
  {#if isExpanded}
    <!-- Container for all layer cards -->
    <div class="layer-cards">
      <!-- Iterate through all available layer types from layerDefinitions -->
      {#each Object.entries(layerDefinitions) as [type, definition]}
        <!-- 
          Individual layer card that can be dragged or clicked
          - CSS custom property for dynamic coloring based on layer type
          - draggable attribute enables HTML5 drag functionality
          - Multiple event handlers for different interaction methods
          - ARIA attributes for accessibility (role="button" + tabindex)
        -->
        <div
          class="layer-card"
          style="--layer-color: {definition.color}"
          draggable="true"
          on:dragstart={(e) => handleDragStart(e, type as LayerType)}
          on:dragend={handleDragEnd}
          on:click={() => handleClick(type as LayerType)}
          on:keydown={(e) => handleKeyDown(e, type as LayerType)}
          role="button"
          tabindex="0"
        >
          <!-- Colored accent bar on the left side of the card -->
          <div class="layer-accent"></div>
          
          <!-- Icon container with layer-specific icon/emoji -->
          <div class="layer-icon">
            <span>{definition.icon}</span>
          </div>
          
          <!-- Layer type display name -->
          <div class="layer-name">{definition.displayName}</div>
          
          <!-- Info icon with tooltip -->
          <button 
            class="layer-info" 
            on:click|stopPropagation={() => showLayerInfo(type as LayerType)}
            aria-label="Show {definition.displayName} details"
          >
            <Tooltip content="Click for detailed information" position="left" delay={200}>
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="8" cy="8" r="7.5" stroke="currentColor" stroke-opacity="0.5"/>
                <path d="M8 7V11" stroke="currentColor" stroke-linecap="round"/>
                <circle cx="8" cy="5" r="0.5" fill="currentColor"/>
              </svg>
            </Tooltip>
          </button>
          
          <!-- Decorative dots on the right (visual hint for draggability) -->
          <div class="layer-dots">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<!-- Layer info popup -->
<LayerInfoPopup 
  layerType={selectedInfoLayer} 
  isOpen={showInfoPopup} 
  onclose={() => showInfoPopup = false} 
/>

<style>
  .layer-palette {
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
  
  .layer-cards {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 0 24px 16px 24px;
  }
  
  .layer-card {
    position: relative;
    display: flex;
    align-items: center;
    gap: 12px;
    height: 36px;
    padding: 0 12px;
    background: #171717;
    border: 1px solid var(--layer-color);
    border-radius: 6px;
    cursor: move;
    transition: all 0.15s;
    user-select: none;
  }
  
  .layer-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  }
  
  .layer-card:active {
    transform: translateY(0);
  }
  
  .layer-accent {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
    background: var(--layer-color);
    border-radius: 6px 0 0 6px;
  }
  
  .layer-icon {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: color-mix(in srgb, var(--layer-color) 20%, transparent);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-left: 8px;
  }
  
  .layer-icon span {
    font-size: 10px;
    font-weight: 600;
    color: var(--layer-color);
  }
  
  .layer-name {
    flex: 1;
    font-size: 12px;
    color: #ffffff;
  }
  
  .layer-info {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    color: #737373;
    cursor: pointer;
    transition: all 0.2s;
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 0;
  }
  
  .layer-info:hover {
    color: #a3a3a3;
    background: rgba(255, 255, 255, 0.05);
  }
  
  .layer-info:active {
    transform: scale(0.95);
  }
  
  .layer-dots {
    display: flex;
    gap: 3px;
    margin-left: 4px;
  }
  
  .layer-dots span {
    width: 3px;
    height: 3px;
    border-radius: 50%;
    background: var(--layer-color);
    opacity: 0.5;
  }
</style>