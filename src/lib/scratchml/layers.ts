import type { LayerType } from './types';

export const availableLayers: LayerType[] = [
  // Input Layers
  {
    id: 'input',
    name: 'Input',
    category: 'input',
    description: 'Entry point for your data',
    defaultParams: {
      shape: [28, 28, 1]
    },
    paramSchema: [
      {
        name: 'shape',
        label: 'Input Shape',
        type: 'array',
        default: [28, 28, 1],
        description: 'Shape of input data (excluding batch size)'
      }
    ],
    constraints: {
      maxInputs: 0,
      minInputs: 0
    }
  },

  // Core Layers
  {
    id: 'dense',
    name: 'Dense',
    category: 'core',
    description: 'Fully connected layer',
    defaultParams: {
      units: 128,
      activation: 'relu'
    },
    paramSchema: [
      {
        name: 'units',
        label: 'Units',
        type: 'number',
        default: 128,
        min: 1,
        max: 1024,
        step: 1,
        description: 'Number of neurons'
      },
      {
        name: 'activation',
        label: 'Activation',
        type: 'select',
        default: 'relu',
        options: [
          { value: 'relu', label: 'ReLU' },
          { value: 'sigmoid', label: 'Sigmoid' },
          { value: 'tanh', label: 'Tanh' },
          { value: 'softmax', label: 'Softmax' },
          { value: 'linear', label: 'Linear' }
        ]
      }
    ]
  },

  // Convolutional Layers
  {
    id: 'conv2d',
    name: 'Conv2D',
    category: 'convolutional',
    description: '2D convolution layer',
    defaultParams: {
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same'
    },
    paramSchema: [
      {
        name: 'filters',
        label: 'Filters',
        type: 'number',
        default: 32,
        min: 1,
        max: 512,
        step: 1,
        description: 'Number of output filters'
      },
      {
        name: 'kernelSize',
        label: 'Kernel Size',
        type: 'number',
        default: 3,
        min: 1,
        max: 7,
        step: 2,
        description: 'Size of convolution window'
      },
      {
        name: 'activation',
        label: 'Activation',
        type: 'select',
        default: 'relu',
        options: [
          { value: 'relu', label: 'ReLU' },
          { value: 'sigmoid', label: 'Sigmoid' },
          { value: 'tanh', label: 'Tanh' },
          { value: 'linear', label: 'Linear' }
        ]
      },
      {
        name: 'padding',
        label: 'Padding',
        type: 'select',
        default: 'same',
        options: [
          { value: 'same', label: 'Same' },
          { value: 'valid', label: 'Valid' }
        ]
      }
    ]
  },

  // Pooling Layers
  {
    id: 'maxPooling2d',
    name: 'Max Pooling 2D',
    category: 'pooling',
    description: 'Max pooling operation',
    defaultParams: {
      poolSize: 2,
      strides: 2
    },
    paramSchema: [
      {
        name: 'poolSize',
        label: 'Pool Size',
        type: 'number',
        default: 2,
        min: 2,
        max: 4,
        step: 1
      },
      {
        name: 'strides',
        label: 'Strides',
        type: 'number',
        default: 2,
        min: 1,
        max: 4,
        step: 1
      }
    ]
  },

  // Normalization
  {
    id: 'batchNormalization',
    name: 'Batch Norm',
    category: 'normalization',
    description: 'Normalizes layer inputs',
    defaultParams: {},
    paramSchema: []
  },

  // Regularization
  {
    id: 'dropout',
    name: 'Dropout',
    category: 'regularization',
    description: 'Randomly drops connections',
    defaultParams: {
      rate: 0.2
    },
    paramSchema: [
      {
        name: 'rate',
        label: 'Dropout Rate',
        type: 'number',
        default: 0.2,
        min: 0,
        max: 0.9,
        step: 0.1,
        description: 'Fraction of inputs to drop'
      }
    ]
  },

  // Utility
  {
    id: 'flatten',
    name: 'Flatten',
    category: 'core',
    description: 'Flattens input to 1D',
    defaultParams: {},
    paramSchema: []
  }
];

export function getLayerType(typeId: string): LayerType | undefined {
  return availableLayers.find(layer => layer.id === typeId);
}