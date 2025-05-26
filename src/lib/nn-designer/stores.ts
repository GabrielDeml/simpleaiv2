import { writable, derived } from 'svelte/store';
import type { LayerConfig, TrainingConfig, DatasetType, ModelSummary } from './types';

/**
 * Main store for the neural network architecture.
 * Contains an array of layer configurations in order from input to output.
 * Default initialized with a single input layer for MNIST.
 */
export const layers = writable<LayerConfig[]>([
  {
    id: 'input-1',
    type: 'input',
    name: 'Input Layer',
    params: { shape: [28, 28] }
  }
]);

/**
 * Currently selected layer ID for property editing.
 * When a layer is clicked in the canvas, its ID is stored here.
 */
export const selectedLayerId = writable<string | null>(null);

/**
 * Training hyperparameters and settings.
 * These values control how the model learns from data.
 */
export const trainingConfig = writable<TrainingConfig>({
  epochs: 10,
  batchSize: 32,
  learningRate: 0.001,
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  validationSplit: 0.2
});

/**
 * Currently selected dataset for training.
 * Determines input shape and number of output classes.
 */
export const selectedDataset = writable<DatasetType>('mnist');

/**
 * Flag indicating if model training is currently in progress.
 * Used to disable UI elements and show progress modal.
 */
export const isTraining = writable(false);

/**
 * Current epoch number during training (0-based).
 * Updated by training callbacks to show progress.
 */
export const currentEpoch = writable(0);

/**
 * Training metrics history for all epochs.
 * Used to plot learning curves and track model performance.
 */
export const trainingHistory = writable<{
  loss: number[];
  valLoss: number[];
  accuracy: number[];
  valAccuracy: number[];
}>({
  loss: [],
  valLoss: [],
  accuracy: [],
  valAccuracy: []
});

/**
 * Derived store that computes model statistics from the layer configuration.
 * Updates automatically when layers change.
 * Note: Full parameter calculation requires building the actual TF.js model.
 */
export const modelSummary = derived(layers, ($layers) => {
  // This will be calculated when we build the actual model
  // For now, return placeholder values
  // TODO: Integrate with ModelBuilder for accurate parameter counts
  const summary: ModelSummary = {
    totalParams: 0,
    trainableParams: 0,
    layerCount: $layers.length,
    outputShape: []
  };
  
  // Simple output shape calculation for last layer
  // Real implementation would trace shapes through the network
  if ($layers.length > 0) {
    const lastLayer = $layers[$layers.length - 1];
    if (lastLayer.type === 'dense') {
      summary.outputShape = [lastLayer.params.units];
    }
  }
  
  return summary;
});

/**
 * Adds a new layer to the network.
 * @param layer - The layer configuration to add
 * @param afterId - Optional ID of layer to insert after. If not provided, adds to end.
 */
export function addLayer(layer: LayerConfig, afterId?: string) {
  layers.update(currentLayers => {
    if (!afterId) {
      return [...currentLayers, layer];
    }
    
    const index = currentLayers.findIndex(l => l.id === afterId);
    if (index === -1) {
      return [...currentLayers, layer];
    }
    
    const newLayers = [...currentLayers];
    newLayers.splice(index + 1, 0, layer);
    return newLayers;
  });
}

/**
 * Removes a layer from the network.
 * Also clears selection if the removed layer was selected.
 * @param layerId - ID of the layer to remove
 */
export function removeLayer(layerId: string) {
  layers.update(currentLayers => {
    return currentLayers.filter(l => l.id !== layerId);
  });
  
  // Clear selection if removed layer was selected
  selectedLayerId.update(selected => selected === layerId ? null : selected);
}

/**
 * Updates the parameters of an existing layer.
 * Merges new params with existing ones (partial update).
 * @param layerId - ID of the layer to update
 * @param params - New parameter values to merge
 */
export function updateLayer(layerId: string, params: Record<string, any>) {
  layers.update(currentLayers => {
    return currentLayers.map(layer => {
      if (layer.id === layerId) {
        return { ...layer, params: { ...layer.params, ...params } };
      }
      return layer;
    });
  });
}

/**
 * Moves a layer up or down in the network architecture.
 * Prevents moving layers before the input layer.
 * @param layerId - ID of the layer to move
 * @param direction - Direction to move ('up' towards input, 'down' towards output)
 */
export function moveLayer(layerId: string, direction: 'up' | 'down') {
  layers.update(currentLayers => {
    const index = currentLayers.findIndex(l => l.id === layerId);
    if (index === -1) return currentLayers;
    
    const newIndex = direction === 'up' ? index - 1 : index + 1;
    
    // Check bounds
    if (newIndex < 0 || newIndex >= currentLayers.length) {
      return currentLayers;
    }
    
    // Don't allow moving before input layer (input must always be first)
    if (newIndex === 0 && currentLayers[0].type === 'input') {
      return currentLayers;
    }
    
    const newLayers = [...currentLayers];
    const [movedLayer] = newLayers.splice(index, 1);
    newLayers.splice(newIndex, 0, movedLayer);
    
    return newLayers;
  });
}

/**
 * Resets all training-related state.
 * Called before starting a new training session.
 */
export function resetTraining() {
  currentEpoch.set(0);
  trainingHistory.set({
    loss: [],
    valLoss: [],
    accuracy: [],
    valAccuracy: []
  });
}