import * as tf from '@tensorflow/tfjs';
import type { LayerNode, Connection, CompiledModel } from './types';
import { getLayerType } from './layers';

export class ModelCompiler {
  static compile(
    layers: Map<string, LayerNode>,
    connections: Connection[]
  ): CompiledModel {
    // Find input layer
    const inputLayer = Array.from(layers.values()).find(l => l.type === 'input');
    if (!inputLayer) {
      throw new Error('Model must have an input layer');
    }

    // Build adjacency map
    const adjacency = new Map<string, string[]>();
    connections.forEach(conn => {
      if (!adjacency.has(conn.from)) {
        adjacency.set(conn.from, []);
      }
      adjacency.get(conn.from)!.push(conn.to);
    });

    // Topological sort to determine layer order
    const sortedLayers = this.topologicalSort(layers, connections);
    
    // Create TensorFlow model
    const model = tf.sequential();
    const layerMap = new Map<string, tf.layers.Layer>();

    for (const layerId of sortedLayers) {
      const layerNode = layers.get(layerId)!;
      const layerType = getLayerType(layerNode.type);
      
      if (!layerType) {
        throw new Error(`Unknown layer type: ${layerNode.type}`);
      }

      // Create TensorFlow layer
      const isFirstLayer = model.layers.length === 0;
      const inputShape = isFirstLayer && inputLayer.params.shape ? inputLayer.params.shape : undefined;
      const tfLayer = this.createTfLayer(layerNode, layerType, layerMap, isFirstLayer, inputShape);
      
      if (tfLayer) {
        model.add(tfLayer);
        layerMap.set(layerId, tfLayer);
      }
    }

    // Calculate model statistics
    const inputShape = inputLayer.params.shape || [28, 28, 1];
    const outputShape = model.outputs[0].shape.slice(1).map(d => d || 0);
    const totalParams = model.countParams();

    return {
      model,
      inputShape,
      outputShape,
      totalParams
    };
  }

  private static createTfLayer(
    node: LayerNode,
    layerType: any,
    layerMap: Map<string, tf.layers.Layer>,
    isFirst: boolean,
    inputShape?: number[]
  ): tf.layers.Layer | null {
    const params = { ...layerType.defaultParams, ...node.params };

    switch (node.type) {
      case 'input':
        // Input layer is handled by inputShape in first layer
        return null;

      case 'dense':
        return tf.layers.dense({
          units: params.units,
          activation: params.activation as any,
          inputShape: isFirst ? inputShape : undefined
        });

      case 'conv2d':
        return tf.layers.conv2d({
          filters: params.filters,
          kernelSize: params.kernelSize,
          activation: params.activation as any,
          padding: params.padding as any,
          inputShape: isFirst ? inputShape : undefined
        });

      case 'maxPooling2d':
        return tf.layers.maxPooling2d({
          poolSize: params.poolSize,
          strides: params.strides,
          inputShape: isFirst ? inputShape : undefined
        });

      case 'dropout':
        return tf.layers.dropout({
          rate: params.rate,
          inputShape: isFirst ? inputShape : undefined
        });

      case 'flatten':
        return tf.layers.flatten({
          inputShape: isFirst ? inputShape : undefined
        });

      case 'batchNormalization':
        return tf.layers.batchNormalization({
          inputShape: isFirst ? inputShape : undefined
        });

      default:
        throw new Error(`Unsupported layer type: ${node.type}`);
    }
  }

  private static topologicalSort(
    layers: Map<string, LayerNode>,
    connections: Connection[]
  ): string[] {
    const visited = new Set<string>();
    const result: string[] = [];
    
    // Build adjacency map
    const adjacency = new Map<string, string[]>();
    const inDegree = new Map<string, number>();
    
    // Initialize
    layers.forEach((_, id) => {
      adjacency.set(id, []);
      inDegree.set(id, 0);
    });
    
    // Build graph
    connections.forEach(conn => {
      adjacency.get(conn.from)!.push(conn.to);
      inDegree.set(conn.to, (inDegree.get(conn.to) || 0) + 1);
    });
    
    // Find all nodes with no incoming edges
    const queue: string[] = [];
    inDegree.forEach((degree, id) => {
      if (degree === 0) {
        queue.push(id);
      }
    });
    
    // Process queue
    while (queue.length > 0) {
      const current = queue.shift()!;
      result.push(current);
      
      // Remove edges from current
      adjacency.get(current)!.forEach(neighbor => {
        const newDegree = inDegree.get(neighbor)! - 1;
        inDegree.set(neighbor, newDegree);
        
        if (newDegree === 0) {
          queue.push(neighbor);
        }
      });
    }
    
    if (result.length !== layers.size) {
      throw new Error('Model contains cycles');
    }
    
    return result;
  }

  static validateModel(layers: Map<string, LayerNode>, connections: Connection[]): string[] {
    const errors: string[] = [];
    
    // Check for input layer
    const inputLayers = Array.from(layers.values()).filter(l => l.type === 'input');
    if (inputLayers.length === 0) {
      errors.push('Model must have at least one input layer');
    } else if (inputLayers.length > 1) {
      errors.push('Model can only have one input layer');
    }
    
    // Check for disconnected layers
    const connectedLayers = new Set<string>();
    connections.forEach(conn => {
      connectedLayers.add(conn.from);
      connectedLayers.add(conn.to);
    });
    
    layers.forEach((layer, id) => {
      if (layer.type !== 'input' && !connectedLayers.has(id)) {
        errors.push(`Layer "${layer.type}" is not connected`);
      }
    });
    
    // Check for cycles
    try {
      this.topologicalSort(layers, connections);
    } catch (e) {
      errors.push('Model contains circular connections');
    }
    
    return errors;
  }
}