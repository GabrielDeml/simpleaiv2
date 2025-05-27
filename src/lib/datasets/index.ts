/**
 * Central export for all dataset implementations.
 * Registers all available datasets and provides unified access.
 */

// Import and register all dataset implementations
import './mnist';
import './cifar10Dataset';
import './fashionMnistDataset';

// Re-export the interface and utilities
export * from './datasetInterface';
export { getDataset, datasetRegistry } from './datasetInterface';

// Import legacy loaders for backward compatibility
export { loadMNIST } from '../mnist/dataLoader';
export { loadCIFAR10 } from './cifar10';
export { loadFashionMNIST } from './fashionMnist';