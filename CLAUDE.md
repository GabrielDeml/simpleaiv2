# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Core Development
- `npm run dev` - Start development server (default port 5173)
- `npm run dev -- --open` - Start dev server and open browser
- `npm run build` - Build for production
- `npm run preview` - Preview production build

### Type Checking
- `npm run check` - Run svelte-kit sync and type check once
- `npm run check:watch` - Run type checking in watch mode

### Testing
- `npm run test` - Run tests with Vitest
- `npm run test:ui` - Run tests with Vitest UI interface
- `npm run test:coverage` - Run tests with coverage report

## Architecture Overview

This is a **Visual Neural Network Designer** built with SvelteKit and TensorFlow.js. It provides a drag-and-drop interface for building, training, and evaluating neural networks directly in the browser.

### Core Systems

1. **Neural Network Designer** (`src/lib/nn-designer/`):
   - **State Management** (`stores.ts`): Svelte stores for layers, training config, and UI state
   - **Type System** (`types.ts`): TypeScript interfaces for layers, configs, and datasets
   - **Layer Definitions** (`layerDefinitions.ts`): Registry of available layer types with defaults
   - **Model Builder** (`modelBuilder.ts`): Converts visual layers to TensorFlow.js models
   - **Training Manager** (`trainingManager.ts`): Handles dataset loading and training orchestration

2. **UI Components** (`src/lib/components/nn-designer/`):
   - **LayerPalette**: Draggable layer cards for adding to network
   - **NetworkCanvas**: Visual network editor with drag-drop support
   - **LayerProperties**: Dynamic property editor for selected layers
   - **ModelSummary**: Real-time parameter count and shape calculation
   - **TrainingConfig**: Training hyperparameter controls
   - **TrainingProgress**: Modal showing live training metrics
   - **DatasetSelector**: Dataset switcher (MNIST, CIFAR-10, Fashion-MNIST)

3. **Data Architecture**:
   - Layers stored as array with unique IDs in Svelte store
   - Drag-and-drop uses HTML5 drag events with layer type transfer
   - Model compilation happens on-demand during training
   - Training state persisted across component lifecycles

### Key Implementation Details

- **Layer System**: Each layer type has predefined parameters and validation. Layers are stored as plain objects and converted to TF.js layers during model building.
- **Visual Rendering**: SVG-based network visualization with CSS animations. Layers positioned vertically with connection lines.
- **State Flow**: Unidirectional data flow using Svelte stores. UI components subscribe to stores and dispatch updates.
- **Model Building**: Sequential models only. Input shape derived from input layer, propagated through network automatically.
- **Training**: Runs in browser using TensorFlow.js. Progress updated via callbacks with epoch metrics stored in history.

### Dataset Integration
- MNIST loaded from Google's hosted sprites (65K images)
- Labels are pre-encoded as one-hot vectors
- Other datasets (CIFAR-10, Fashion-MNIST) have UI but need implementation

### Important Notes
- No backend required - all ML runs client-side
- Models can be saved to localStorage but not exported yet
- No linting configured
- Test framework: Vitest with jsdom environment, tests in `*.test.ts` files
- Accessibility warnings exist but don't affect functionality