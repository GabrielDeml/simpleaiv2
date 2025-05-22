# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

- **Start development server**: `npm run dev`
- **Build for production**: `npm run build`
- **Lint code**: `npm run lint`
- **Preview production build**: `npm run preview`

## Architecture Overview

This is a React + Vite application that implements an MNIST digit recognition trainer using TensorFlow.js. The architecture follows modern React patterns with clean separation of concerns:

### Core Structure
- **MNISTTrainer.jsx** - Main orchestrating component that coordinates all functionality
- **Custom Hooks Pattern** - Business logic abstracted into reusable hooks:
  - `useMNISTData.js` - Manages dataset loading, preprocessing, and state
  - `useMNISTModel.js` - Handles model lifecycle (creation, training, testing, inference)

### Component Architecture
- **ControlPanel.jsx** - Action buttons and controls
- **ImageCanvas.jsx** - Canvas component for MNIST digit visualization (uses forwardRef for imperative API)
- **TrainingLogs.jsx** - Real-time training progress display
- **ErrorBoundary.jsx** - React error boundary for graceful error handling

### Configuration System
All configuration is centralized in `src/constants/mnistConfig.js`:
- `DATASET_CONFIG` - Image dimensions, dataset size, train/test split
- `TRAINING_CONFIG` - Batch size, epochs, validation split
- `MODEL_CONFIG` - Neural network architecture (hidden units, dropout, activations)
- `DATA_URLS` - External MNIST data sources
- `PERFORMANCE_CONFIG` - Logging levels, memory management settings

### Data Flow
1. MNIST data loads from external URLs on app mount
2. Model is created with configurable architecture
3. Training runs asynchronously with real-time progress updates
4. Testing evaluates model performance on test dataset
5. Prediction visualizes results on random test samples

### Memory Management
- Automatic TensorFlow.js tensor disposal to prevent memory leaks
- Cleanup on component unmount
- Configurable memory management via `PERFORMANCE_CONFIG.MEMORY_CLEANUP_ENABLED`

### Error Handling Strategy
- Try/catch blocks throughout async operations
- React Error Boundaries for component-level errors
- User-friendly error messages with recovery options
- Graceful degradation when components fail

### Logging System
Configurable logging via `src/utils/logger.js` with levels: DEBUG, INFO, WARN, ERROR
Set via `PERFORMANCE_CONFIG.LOG_LEVEL` in config file.