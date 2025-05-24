<script lang="ts">
  import { onMount } from 'svelte';
  import { modelStore } from '../modelStore';
  import { getLayerType } from '../layers';
  import type { LayerNode } from '../types';
  
  let canvas: HTMLDivElement;
  let isDragging = false;
  let draggedLayerId: string | null = null;
  let connectingFrom: string | null = null;
  let mousePos = { x: 0, y: 0 };
  
  const { layers, connections, selectedLayer, hoveredLayer } = modelStore;
  
  function handleDrop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    const layerType = event.dataTransfer?.getData('layerType');
    
    if (layerType && canvas) {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left - 60; // Center the layer
      const y = event.clientY - rect.top - 30;
      
      modelStore.addLayer(layerType, { x, y });
    }
  }
  
  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    event.dataTransfer!.dropEffect = 'copy';
  }
  
  function handleLayerClick(layerId: string, event: MouseEvent | KeyboardEvent) {
    event.stopPropagation();
    
    if (connectingFrom) {
      // Complete connection
      if (connectingFrom !== layerId) {
        modelStore.addConnection(connectingFrom, layerId);
      }
      connectingFrom = null;
    } else {
      selectedLayer.set(layerId);
    }
  }
  
  function handleLayerDoubleClick(layerId: string, event: MouseEvent) {
    event.stopPropagation();
    // Emit event for opening config panel
    dispatchEvent(new CustomEvent('configureLayer', { detail: layerId }));
  }
  
  function handleLayerRightClick(layerId: string, event: MouseEvent) {
    event.preventDefault();
    event.stopPropagation();
    modelStore.removeLayer(layerId);
  }
  
  function handleOutputClick(layerId: string, event: MouseEvent | KeyboardEvent) {
    event.stopPropagation();
    connectingFrom = layerId;
  }
  
  function handleCanvasClick() {
    selectedLayer.set(null);
    connectingFrom = null;
  }
  
  function handleMouseMove(event: MouseEvent) {
    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      mousePos = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
      };
    }
    
    if (isDragging && draggedLayerId) {
      const layer = $layers.get(draggedLayerId);
      if (layer) {
        modelStore.updateLayer(draggedLayerId, {
          position: { x: mousePos.x - 60, y: mousePos.y - 30 }
        });
      }
    }
  }
  
  function startDragging(layerId: string, event: MouseEvent) {
    if (event.button === 0) { // Left click only
      isDragging = true;
      draggedLayerId = layerId;
      event.preventDefault();
    }
  }
  
  function stopDragging() {
    isDragging = false;
    draggedLayerId = null;
  }
  
  function getConnectionPath(from: LayerNode, to: LayerNode): string {
    const x1 = from.position.x + 120;
    const y1 = from.position.y + 30;
    const x2 = to.position.x;
    const y2 = to.position.y + 30;
    
    const dx = x2 - x1;
    const cp1x = x1 + dx * 0.5;
    const cp2x = x2 - dx * 0.5;
    
    return `M ${x1} ${y1} C ${cp1x} ${y1}, ${cp2x} ${y2}, ${x2} ${y2}`;
  }
  
  function getLayerColor(layerType: string): string {
    const type = getLayerType(layerType);
    const colors: Record<string, string> = {
      input: '#3B82F6',
      core: '#10B981',
      convolutional: '#8B5CF6',
      pooling: '#F59E0B',
      regularization: '#EF4444',
      normalization: '#6366F1',
      activation: '#6B7280'
    };
    return colors[type?.category || 'core'] || '#6B7280';
  }
  
  onMount(() => {
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', stopDragging);
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', stopDragging);
    };
  });
</script>

<div 
  class="relative flex-1 bg-gray-50 dark:bg-gray-900 overflow-hidden min-h-[600px]"
  bind:this={canvas}
>
  <!-- Drop zone for canvas -->
  <div
    class="absolute inset-0"
    on:drop={handleDrop}
    on:dragover={handleDragOver}
    on:click={handleCanvasClick}
    on:keydown={(e) => e.key === 'Escape' && handleCanvasClick()}
    role="button"
    tabindex="-1"
    aria-label="Neural network model canvas - click to deselect"
  ></div>
  <!-- SVG for connections -->
  <svg
    class="absolute inset-0 pointer-events-none"
    width="100%"
    height="100%"
  >
    <!-- Existing connections -->
    {#each $connections as connection}
      {@const fromLayer = $layers.get(connection.from)}
      {@const toLayer = $layers.get(connection.to)}
      {#if fromLayer && toLayer}
        <path
          d={getConnectionPath(fromLayer, toLayer)}
          class="transition-all"
          stroke="#6B7280"
          stroke-width="2"
          fill="none"
        />
      {/if}
    {/each}
    
    <!-- Connection being drawn -->
    {#if connectingFrom}
      {@const fromLayer = $layers.get(connectingFrom)}
      {#if fromLayer}
        <path
          d={`M ${fromLayer.position.x + 120} ${fromLayer.position.y + 30} L ${mousePos.x} ${mousePos.y}`}
          class="connection-drawing"
          stroke="#3B82F6"
          stroke-width="2"
          stroke-dasharray="5,5"
          fill="none"
        />
      {/if}
    {/if}
  </svg>
  
  <!-- Layer nodes -->
  {#each [...$layers.entries()] as [layerId, layer]}
    {@const layerType = getLayerType(layer.type)}
    <div
      class="absolute flex items-center bg-white dark:bg-gray-800 rounded-lg shadow-md cursor-move select-none transition-all hover:shadow-lg w-[120px] h-[60px] p-0
        {$selectedLayer === layerId ? 'ring-2 ring-blue-500 ring-offset-2' : ''}"
      style="left: {layer.position.x}px; top: {layer.position.y}px; background-color: {getLayerColor(layer.type)}"
      on:click={(e) => handleLayerClick(layerId, e)}
      on:keydown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleLayerClick(layerId, e);
        } else if (e.key === 'Delete' || e.key === 'Backspace') {
          e.preventDefault();
          modelStore.removeLayer(layerId);
        }
      }}
      on:dblclick={(e) => handleLayerDoubleClick(layerId, e)}
      on:contextmenu={(e) => handleLayerRightClick(layerId, e)}
      on:mousedown={(e) => startDragging(layerId, e)}
      on:mouseenter={() => hoveredLayer.set(layerId)}
      on:mouseleave={() => hoveredLayer.set(null)}
      role="button"
      tabindex="0"
      aria-label="{layerType?.name || layer.type} layer"
    >
      <!-- Input port -->
      {#if layer.type !== 'input'}
        <div class="absolute -left-1.5 top-1/2 -translate-y-1/2 w-3 h-3 bg-white border-2 border-gray-400 rounded-full cursor-pointer hover:border-blue-500 transition-colors"></div>
      {/if}
      
      <!-- Layer content -->
      <div class="flex-1 text-center text-white px-2">
        <div class="font-semibold text-sm">{layerType?.name || layer.type}</div>
        {#if layer.type === 'dense'}
          <div class="text-xs opacity-90">units: {layer.params.units || 128}</div>
        {:else if layer.type === 'conv2d'}
          <div class="text-xs opacity-90">filters: {layer.params.filters || 32}</div>
        {:else if layer.type === 'dropout'}
          <div class="text-xs opacity-90">rate: {layer.params.rate || 0.2}</div>
        {/if}
      </div>
      
      <!-- Output port -->
      <div 
        class="absolute -right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 bg-white border-2 border-gray-400 rounded-full cursor-pointer hover:border-blue-500 transition-colors pointer-events-auto"
        on:click={(e) => handleOutputClick(layerId, e)}
        on:keydown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            handleOutputClick(layerId, e);
          }
        }}
        role="button"
        tabindex="0"
        aria-label="Connect from {layerType?.name || layer.type} output"
      ></div>
    </div>
  {/each}
  
  {#if $layers.size === 0}
    <div class="absolute inset-0 flex items-center justify-center">
      <p class="text-gray-500">Drag layers from the palette to start building your model</p>
    </div>
  {/if}
</div>

<style>
  .connection-drawing {
    animation: dash 0.5s linear infinite;
  }
  
  @keyframes dash {
    to {
      stroke-dashoffset: -10;
    }
  }
</style>