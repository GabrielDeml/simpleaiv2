# Visual Neural Network Designer Documentation

## Overview

A browser-based neural network designer built with SvelteKit and TensorFlow.js. Create, train, and evaluate neural networks through an intuitive drag-and-drop interface with no backend required.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Systems](#core-systems)
3. [UI Components](#ui-components)
4. [Data Flow](#data-flow)
5. [API Reference](#api-reference)
6. [Usage Guide](#usage-guide)

## Architecture Overview

The application follows a clean separation of concerns:

- **UI Layer**: Svelte components for visual interaction
- **State Management**: Svelte stores for reactive data flow
- **ML Core**: TensorFlow.js integration for model building and training
- **Data Management**: Client-side dataset loading and preprocessing

### Technology Stack

- **Frontend**: SvelteKit 2.0
- **ML Framework**: TensorFlow.js 4.22
- **Styling**: Tailwind CSS
- **Build Tool**: Vite
- **Type System**: TypeScript

## Core Systems

### 1. Type System (`src/lib/nn-designer/types.ts`)

Defines the data structures used throughout the application:

```typescript
interface Layer {
  id: string;
  type: LayerType;
  config: LayerConfig;
}

type LayerType = 'input' | 'dense' | 'conv2d' | 'maxpooling2d' | 'dropout' | 'flatten';

interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationSplit: number;
  optimizer: 'adam' | 'sgd' | 'rmsprop';
  loss: 'categoricalCrossentropy' | 'meanSquaredError';
}
```

### 2. State Management (`src/lib/nn-designer/stores.ts`)

Centralized state using Svelte stores:

- **layers**: Array of layer configurations
- **selectedLayerId**: Currently selected layer for editing
- **trainingConfig**: Training hyperparameters
- **selectedDataset**: Active dataset ('mnist', 'cifar10', 'fashion-mnist')
- **Training state**: isTraining, currentEpoch, trainingHistory

Key functions:
- `addLayer(layer)`: Add new layer to network
- `removeLayer(id)`: Remove layer by ID
- `updateLayer(id, config)`: Update layer configuration
- `moveLayer(fromIndex, toIndex)`: Reorder layers

### 3. Model Builder (`src/lib/nn-designer/modelBuilder.ts`)

Converts visual layer configurations to TensorFlow.js models:

```typescript
class ModelBuilder {
  build(layers: Layer[]): tf.Sequential
  compile(model: tf.Sequential, config: TrainingConfig): void
  train(model: tf.Sequential, data, config, callbacks): Promise<tf.History>
  exportModel(model: tf.Sequential, name: string): Promise<void>
}
```

Features:
- Automatic input shape propagation
- Layer-specific parameter mapping
- Model compilation with optimizer selection
- Training with progress callbacks
- Model export (browser download or JSON)

### 4. Training Manager (`src/lib/nn-designer/trainingManager.ts`)

Handles dataset loading and training orchestration:

```typescript
class TrainingManager {
  loadDataset(datasetName: string): Promise<Dataset>
  prepareData(dataset: Dataset): { trainData, trainLabels, testData, testLabels }
  train(layers: Layer[], dataset: Dataset, config: TrainingConfig): Promise<void>
  predict(model: tf.Sequential, inputData: tf.Tensor): tf.Tensor
}
```

Supported datasets:
- MNIST: 28x28 grayscale digits
- CIFAR-10: 32x32 color images (10 classes)
- Fashion-MNIST: 28x28 grayscale fashion items

## UI Components

### Main Component (`NeuralNetworkDesigner.svelte`)

The root component providing:
- Three-panel layout (sidebar, canvas, properties)
- Toolbar with actions (Run, Save, Export, Clear)
- Model type selector (Sequential only currently)
- Training modal integration

### Layer Palette (`LayerPalette.svelte`)

Draggable layer cards:
- Visual representation of available layers
- Drag-to-add functionality
- Layer type icons and descriptions

### Network Canvas (`NetworkCanvas.svelte`)

Visual network editor:
- SVG-based layer visualization
- Drag-and-drop layer addition
- Layer selection and deletion
- Connection lines between layers
- Real-time shape validation

### Layer Properties (`LayerProperties.svelte`)

Dynamic property editor:
- Context-sensitive controls per layer type
- Input validation
- Real-time updates to model

### Model Summary (`ModelSummary.svelte`)

Real-time model statistics:
- Total parameter count
- Trainable/non-trainable parameters
- Layer-by-layer shape information
- Memory usage estimation

### Training Components

**TrainingConfig.svelte**:
- Hyperparameter controls
- Optimizer selection
- Loss function selection
- Validation split configuration

**TrainingProgress.svelte**:
- Modal overlay during training
- Real-time loss/accuracy charts
- Epoch progress indicator
- Stop training capability

**MetricsChart.svelte**:
- Live training metrics visualization
- Loss and accuracy plots
- Train/validation comparison

## Data Flow

1. **User Interaction** → UI Components
2. **UI Events** → Store Updates
3. **Store Changes** → Model Builder
4. **Model Builder** → TensorFlow.js
5. **Training Progress** → Store Updates
6. **Store Updates** → UI Re-render

### State Flow Example

```
User drags Dense layer → NetworkCanvas handles drop
→ addLayer() called → layers store updated
→ ModelSummary recalculates → UI updates
```

### Training Flow

```
User clicks Run → TrainingManager.train()
→ Load dataset → Build model → Compile
→ Training loop with callbacks → Update progress store
→ TrainingProgress component shows metrics
```

## API Reference

### Layer Configurations

#### Input Layer
```typescript
{
  type: 'input',
  config: {
    inputShape: number[] // e.g., [28, 28, 1]
  }
}
```

#### Dense Layer
```typescript
{
  type: 'dense',
  config: {
    units: number,
    activation: 'relu' | 'sigmoid' | 'tanh' | 'softmax'
  }
}
```

#### Conv2D Layer
```typescript
{
  type: 'conv2d',
  config: {
    filters: number,
    kernelSize: number | [number, number],
    strides: [number, number],
    padding: 'valid' | 'same',
    activation: string
  }
}
```

#### MaxPooling2D Layer
```typescript
{
  type: 'maxpooling2d',
  config: {
    poolSize: [number, number],
    strides: [number, number],
    padding: 'valid' | 'same'
  }
}
```

#### Dropout Layer
```typescript
{
  type: 'dropout',
  config: {
    rate: number // 0-1
  }
}
```

#### Flatten Layer
```typescript
{
  type: 'flatten',
  config: {} // No parameters
}
```

### Store API

```typescript
// Add a new layer
layers.update(currentLayers => [...currentLayers, newLayer]);

// Update layer configuration
updateLayer(layerId, { units: 128 });

// Remove a layer
removeLayer(layerId);

// Start training
isTraining.set(true);
currentEpoch.set(0);
```

### Model Builder API

```typescript
const builder = new ModelBuilder();

// Build model from layers
const model = builder.build(layersArray);

// Compile with training config
builder.compile(model, trainingConfig);

// Train with callbacks
const history = await builder.train(model, data, config, {
  onEpochEnd: (epoch, logs) => {
    // Update progress
  }
});

// Export model
await builder.exportModel(model, 'my-model');
```

## Usage Guide

### Creating a Model

1. **Add Input Layer**: Drag input layer from palette, set shape based on dataset
2. **Add Hidden Layers**: Drag and configure dense, conv2d, or other layers
3. **Add Output Layer**: Automatically added based on dataset (10 units for MNIST)

### Training a Model

1. **Select Dataset**: Choose from MNIST, CIFAR-10, or Fashion-MNIST
2. **Configure Training**: Set epochs, batch size, learning rate
3. **Click Run**: Training starts with live metrics display
4. **Monitor Progress**: Watch loss decrease and accuracy increase

### Exporting a Model

1. **Train Model**: Ensure model is trained
2. **Click Export**: Choose format (TensorFlow.js or Keras)
3. **Download**: Model files saved to browser downloads

### Best Practices

1. **Start Simple**: Begin with basic dense networks
2. **Match Input Shape**: Ensure input layer matches dataset dimensions
3. **Add Dropout**: Include dropout layers to prevent overfitting
4. **Monitor Validation**: Watch validation metrics to detect overfitting
5. **Experiment**: Try different architectures and hyperparameters

## Development

### Running Locally

```bash
npm install
npm run dev
```

### Building for Production

```bash
npm run build
npm run preview
```

### Type Checking

```bash
npm run check
npm run check:watch
```

## Limitations

- Sequential models only (no branching/merging)
- Limited to predefined datasets
- No custom data upload yet
- Browser memory constraints for large models
- No server-side training support

## Future Enhancements

- Custom dataset upload
- More layer types (LSTM, GRU, BatchNorm)
- Model architecture templates
- Training checkpoints and resumption
- Real-time inference playground
- Model performance profiling
- Export to other formats (ONNX, Core ML)