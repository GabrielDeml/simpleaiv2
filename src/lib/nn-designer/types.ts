export type LayerType = 'input' | 'dense' | 'conv2d' | 'maxpooling2d' | 'dropout' | 'flatten';

export interface LayerConfig {
  id: string;
  type: LayerType;
  name: string;
  params: Record<string, any>;
}

export interface InputLayerParams {
  shape: number[];
}

export interface DenseLayerParams {
  units: number;
  activation: 'relu' | 'sigmoid' | 'tanh' | 'softmax' | 'linear';
  useBias: boolean;
  kernelInitializer: string;
}

export interface Conv2DLayerParams {
  filters: number;
  kernelSize: number | [number, number];
  strides: number | [number, number];
  padding: 'valid' | 'same';
  activation: 'relu' | 'sigmoid' | 'tanh' | 'linear';
  useBias: boolean;
}

export interface MaxPooling2DLayerParams {
  poolSize: number | [number, number];
  strides: number | [number, number];
  padding: 'valid' | 'same';
}

export interface DropoutLayerParams {
  rate: number;
}

export interface FlattenLayerParams {
  // No parameters needed
}

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  optimizer: 'adam' | 'sgd' | 'rmsprop';
  loss: 'categoricalCrossentropy' | 'meanSquaredError' | 'binaryCrossentropy';
  validationSplit: number;
}

export interface ModelSummary {
  totalParams: number;
  trainableParams: number;
  layerCount: number;
  outputShape: number[];
}

export type DatasetType = 'mnist' | 'cifar10' | 'fashion-mnist' | 'custom';

export interface Dataset {
  type: DatasetType;
  name: string;
  shape: number[];
  classes: number;
  trainSize: number;
  testSize: number;
}