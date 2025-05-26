import type { LayerType, LayerConfig } from './types';

export interface LayerDefinition {
  type: LayerType;
  displayName: string;
  icon: string;
  color: string;
  defaultParams: Record<string, any>;
}

export const layerDefinitions: Record<LayerType, LayerDefinition> = {
  input: {
    type: 'input',
    displayName: 'Input Layer',
    icon: 'I',
    color: '#8b5cf6',
    defaultParams: {
      shape: [28, 28]
    }
  },
  dense: {
    type: 'dense',
    displayName: 'Dense',
    icon: 'D',
    color: '#22c55e',
    defaultParams: {
      units: 128,
      activation: 'relu',
      useBias: true,
      kernelInitializer: 'glorotUniform'
    }
  },
  conv2d: {
    type: 'conv2d',
    displayName: 'Conv2D',
    icon: 'C',
    color: '#f59e0b',
    defaultParams: {
      filters: 32,
      kernelSize: 3,
      strides: 1,
      padding: 'same',
      activation: 'relu',
      useBias: true
    }
  },
  maxpooling2d: {
    type: 'maxpooling2d',
    displayName: 'MaxPooling2D',
    icon: 'MP',
    color: '#06b6d4',
    defaultParams: {
      poolSize: 2,
      strides: 2,
      padding: 'valid'
    }
  },
  dropout: {
    type: 'dropout',
    displayName: 'Dropout',
    icon: 'Dr',
    color: '#ef4444',
    defaultParams: {
      rate: 0.2
    }
  },
  flatten: {
    type: 'flatten',
    displayName: 'Flatten',
    icon: 'F',
    color: '#a855f7',
    defaultParams: {}
  }
};