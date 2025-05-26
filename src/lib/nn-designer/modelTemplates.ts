import type { ModelTemplate } from './types';

/**
 * Predefined model templates for common neural network architectures.
 * These templates provide starting points for different types of problems.
 */
export const modelTemplates: ModelTemplate[] = [
  {
    id: 'simple-dense',
    name: 'Simple Dense Network',
    description: 'Basic fully-connected network for simple classification tasks',
    category: 'classification',
    recommendedDataset: 'mnist',
    layers: [
      {
        id: 'input-1',
        type: 'input',
        name: 'Input Layer',
        params: { shape: [28, 28, 1] }
      },
      {
        id: 'flatten-1',
        type: 'flatten',
        name: 'Flatten',
        params: {}
      },
      {
        id: 'dense-1',
        type: 'dense',
        name: 'Hidden Layer 1',
        params: {
          units: 128,
          activation: 'relu',
          useBias: true,
          kernelInitializer: 'glorotUniform'
        }
      },
      {
        id: 'dense-2',
        type: 'dense',
        name: 'Hidden Layer 2',
        params: {
          units: 64,
          activation: 'relu',
          useBias: true,
          kernelInitializer: 'glorotUniform'
        }
      },
      {
        id: 'output-1',
        type: 'output',
        name: 'Output Layer',
        params: {
          units: 10,
          activation: 'softmax',
          useBias: true,
          kernelInitializer: 'glorotUniform'
        }
      }
    ]
  },
  {
    id: 'deep-dense',
    name: 'Deep Dense Network',
    description: 'Multi-layer fully-connected network with dropout regularization',
    category: 'classification',
    recommendedDataset: 'fashion-mnist',
    layers: [
      {
        id: 'input-1',
        type: 'input',
        name: 'Input Layer',
        params: { shape: [28, 28, 1] }
      },
      {
        id: 'flatten-1',
        type: 'flatten',
        name: 'Flatten',
        params: {}
      },
      {
        id: 'dense-1',
        type: 'dense',
        name: 'Hidden Layer 1',
        params: {
          units: 256,
          activation: 'relu',
          useBias: true,
          kernelInitializer: 'glorotUniform'
        }
      },
      {
        id: 'dropout-1',
        type: 'dropout',
        name: 'Dropout 1',
        params: { rate: 0.3 }
      },
      {
        id: 'dense-2',
        type: 'dense',
        name: 'Hidden Layer 2',
        params: {
          units: 128,
          activation: 'relu',
          useBias: true,
          kernelInitializer: 'glorotUniform'
        }
      },
      {
        id: 'dropout-2',
        type: 'dropout',
        name: 'Dropout 2',
        params: { rate: 0.2 }
      },
      {
        id: 'dense-3',
        type: 'dense',
        name: 'Hidden Layer 3',
        params: {
          units: 64,
          activation: 'relu',
          useBias: true,
          kernelInitializer: 'glorotUniform'
        }
      },
      {
        id: 'output-1',
        type: 'output',
        name: 'Output Layer',
        params: {
          units: 10,
          activation: 'softmax',
          useBias: true,
          kernelInitializer: 'glorotUniform'
        }
      }
    ]
  },
  {
    id: 'simple-cnn',
    name: 'Simple CNN',
    description: 'Basic convolutional network for image classification',
    category: 'computer-vision',
    recommendedDataset: 'mnist',
    layers: [
      {
        id: 'input-1',
        type: 'input',
        name: 'Input Layer',
        params: { shape: [28, 28, 1] }
      },
      {
        id: 'conv2d-1',
        type: 'conv2d',
        name: 'Conv Layer 1',
        params: {
          filters: 32,
          kernelSize: 3,
          strides: 1,
          padding: 'same',
          activation: 'relu',
          useBias: true
        }
      },
      {
        id: 'maxpooling2d-1',
        type: 'maxpooling2d',
        name: 'MaxPool 1',
        params: {
          poolSize: 2,
          strides: 2,
          padding: 'valid'
        }
      },
      {
        id: 'conv2d-2',
        type: 'conv2d',
        name: 'Conv Layer 2',
        params: {
          filters: 64,
          kernelSize: 3,
          strides: 1,
          padding: 'same',
          activation: 'relu',
          useBias: true
        }
      },
      {
        id: 'maxpooling2d-2',
        type: 'maxpooling2d',
        name: 'MaxPool 2',
        params: {
          poolSize: 2,
          strides: 2,
          padding: 'valid'
        }
      },
      {
        id: 'flatten-1',
        type: 'flatten',
        name: 'Flatten',
        params: {}
      },
      {
        id: 'dense-1',
        type: 'dense',
        name: 'Dense Layer',
        params: {
          units: 128,
          activation: 'relu',
          useBias: true,
          kernelInitializer: 'glorotUniform'
        }
      },
      {
        id: 'output-1',
        type: 'output',
        name: 'Output Layer',
        params: {
          units: 10,
          activation: 'softmax',
          useBias: true,
          kernelInitializer: 'glorotUniform'
        }
      }
    ]
  },
  {
    id: 'advanced-cnn',
    name: 'Advanced CNN',
    description: 'Deep convolutional network with multiple conv blocks and regularization',
    category: 'computer-vision',
    recommendedDataset: 'cifar10',
    layers: [
      {
        id: 'input-1',
        type: 'input',
        name: 'Input Layer',
        params: { shape: [32, 32, 3] }
      },
      {
        id: 'conv2d-1',
        type: 'conv2d',
        name: 'Conv Block 1A',
        params: {
          filters: 32,
          kernelSize: 3,
          strides: 1,
          padding: 'same',
          activation: 'relu',
          useBias: true
        }
      },
      {
        id: 'conv2d-2',
        type: 'conv2d',
        name: 'Conv Block 1B',
        params: {
          filters: 32,
          kernelSize: 3,
          strides: 1,
          padding: 'same',
          activation: 'relu',
          useBias: true
        }
      },
      {
        id: 'maxpooling2d-1',
        type: 'maxpooling2d',
        name: 'MaxPool 1',
        params: {
          poolSize: 2,
          strides: 2,
          padding: 'valid'
        }
      },
      {
        id: 'dropout-1',
        type: 'dropout',
        name: 'Dropout 1',
        params: { rate: 0.25 }
      },
      {
        id: 'conv2d-3',
        type: 'conv2d',
        name: 'Conv Block 2A',
        params: {
          filters: 64,
          kernelSize: 3,
          strides: 1,
          padding: 'same',
          activation: 'relu',
          useBias: true
        }
      },
      {
        id: 'conv2d-4',
        type: 'conv2d',
        name: 'Conv Block 2B',
        params: {
          filters: 64,
          kernelSize: 3,
          strides: 1,
          padding: 'same',
          activation: 'relu',
          useBias: true
        }
      },
      {
        id: 'maxpooling2d-2',
        type: 'maxpooling2d',
        name: 'MaxPool 2',
        params: {
          poolSize: 2,
          strides: 2,
          padding: 'valid'
        }
      },
      {
        id: 'dropout-2',
        type: 'dropout',
        name: 'Dropout 2',
        params: { rate: 0.25 }
      },
      {
        id: 'flatten-1',
        type: 'flatten',
        name: 'Flatten',
        params: {}
      },
      {
        id: 'dense-1',
        type: 'dense',
        name: 'Dense Layer 1',
        params: {
          units: 512,
          activation: 'relu',
          useBias: true,
          kernelInitializer: 'glorotUniform'
        }
      },
      {
        id: 'dropout-3',
        type: 'dropout',
        name: 'Dropout 3',
        params: { rate: 0.5 }
      },
      {
        id: 'dense-2',
        type: 'dense',
        name: 'Dense Layer 2',
        params: {
          units: 256,
          activation: 'relu',
          useBias: true,
          kernelInitializer: 'glorotUniform'
        }
      },
      {
        id: 'output-1',
        type: 'output',
        name: 'Output Layer',
        params: {
          units: 10,
          activation: 'softmax',
          useBias: true,
          kernelInitializer: 'glorotUniform'
        }
      }
    ]
  }
];

/**
 * Get a model template by its ID.
 * @param templateId - The unique identifier of the template
 * @returns The model template or undefined if not found
 */
export function getTemplate(templateId: string): ModelTemplate | undefined {
  return modelTemplates.find(template => template.id === templateId);
}

/**
 * Get all templates in a specific category.
 * @param category - The category to filter by
 * @returns Array of templates in the specified category
 */
export function getTemplatesByCategory(category: ModelTemplate['category']): ModelTemplate[] {
  return modelTemplates.filter(template => template.category === category);
}