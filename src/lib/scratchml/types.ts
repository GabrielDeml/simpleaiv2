import type * as tf from '@tensorflow/tfjs';

export interface LayerType {
  id: string;
  name: string;
  category: 'input' | 'core' | 'convolutional' | 'pooling' | 'normalization' | 'regularization' | 'activation';
  icon?: string;
  description: string;
  defaultParams: Record<string, any>;
  paramSchema: ParamSchema[];
  constraints?: {
    minInputs?: number;
    maxInputs?: number;
    allowedPrevious?: string[];
    allowedNext?: string[];
  };
}

export interface ParamSchema {
  name: string;
  label: string;
  type: 'number' | 'string' | 'select' | 'boolean' | 'array';
  default: any;
  options?: { value: any; label: string }[];
  min?: number;
  max?: number;
  step?: number;
  description?: string;
}

export interface LayerNode {
  id: string;
  type: string;
  params: Record<string, any>;
  position: { x: number; y: number };
  inputs: string[];
  outputs: string[];
}

export interface ModelGraph {
  layers: Map<string, LayerNode>;
  connections: Connection[];
}

export interface Connection {
  from: string;
  to: string;
}

export interface CompiledModel {
  model: tf.Sequential | tf.LayersModel;
  inputShape: number[];
  outputShape: number[];
  totalParams: number;
}

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  optimizer: 'adam' | 'sgd' | 'rmsprop';
  loss: string;
  metrics: string[];
}