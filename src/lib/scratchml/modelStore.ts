import { writable, derived } from 'svelte/store';
import type { LayerNode, ModelGraph, Connection } from './types';

function createModelStore() {
  const layers = writable<Map<string, LayerNode>>(new Map());
  const connections = writable<Connection[]>([]);
  const selectedLayer = writable<string | null>(null);
  const hoveredLayer = writable<string | null>(null);

  let layerIdCounter = 0;

  function generateLayerId(): string {
    return `layer_${++layerIdCounter}`;
  }

  function addLayer(type: string, position: { x: number; y: number }) {
    const id = generateLayerId();
    const newLayer: LayerNode = {
      id,
      type,
      position,
      params: {},
      inputs: [],
      outputs: []
    };

    layers.update(map => {
      const newMap = new Map(map);
      newMap.set(id, newLayer);
      return newMap;
    });

    return id;
  }

  function updateLayer(id: string, updates: Partial<LayerNode>) {
    layers.update(map => {
      const layer = map.get(id);
      if (layer) {
        const newMap = new Map(map);
        newMap.set(id, { ...layer, ...updates });
        return newMap;
      }
      return map;
    });
  }

  function removeLayer(id: string) {
    layers.update(map => {
      const newMap = new Map(map);
      newMap.delete(id);
      return newMap;
    });

    // Remove all connections involving this layer
    connections.update(conns => 
      conns.filter(conn => conn.from !== id && conn.to !== id)
    );
  }

  function addConnection(from: string, to: string) {
    connections.update(conns => {
      // Remove any existing connection to this target
      const filtered = conns.filter(conn => conn.to !== to);
      return [...filtered, { from, to }];
    });

    // Update layer inputs/outputs
    layers.update(map => {
      const fromLayer = map.get(from);
      const toLayer = map.get(to);
      
      if (fromLayer && toLayer) {
        const newMap = new Map(map);
        
        // Update from layer outputs
        if (!fromLayer.outputs.includes(to)) {
          newMap.set(from, {
            ...fromLayer,
            outputs: [...fromLayer.outputs, to]
          });
        }
        
        // Update to layer inputs
        newMap.set(to, {
          ...toLayer,
          inputs: [from] // Only one input per layer
        });
        
        return newMap;
      }
      return map;
    });
  }

  function removeConnection(from: string, to: string) {
    connections.update(conns => 
      conns.filter(conn => !(conn.from === from && conn.to === to))
    );

    // Update layer inputs/outputs
    layers.update(map => {
      const fromLayer = map.get(from);
      const toLayer = map.get(to);
      
      if (fromLayer && toLayer) {
        const newMap = new Map(map);
        
        // Update from layer outputs
        newMap.set(from, {
          ...fromLayer,
          outputs: fromLayer.outputs.filter(id => id !== to)
        });
        
        // Update to layer inputs
        newMap.set(to, {
          ...toLayer,
          inputs: toLayer.inputs.filter(id => id !== from)
        });
        
        return newMap;
      }
      return map;
    });
  }

  function clearModel() {
    layers.set(new Map());
    connections.set([]);
    selectedLayer.set(null);
    hoveredLayer.set(null);
    layerIdCounter = 0;
  }

  // Derived store for model validation
  const isValidModel = derived(
    [layers, connections],
    ([$layers, $connections]) => {
      // Check if we have layers
      if ($layers.size === 0) return false;
      
      // Check if we have an input layer
      const hasInput = Array.from($layers.values()).some(layer => layer.type === 'input');
      if (!hasInput) return false;
      
      // Check if all non-input layers have inputs
      const allConnected = Array.from($layers.values()).every(layer => {
        if (layer.type === 'input') return true;
        return layer.inputs.length > 0;
      });
      
      return allConnected;
    }
  );

  return {
    layers,
    connections,
    selectedLayer,
    hoveredLayer,
    isValidModel,
    addLayer,
    updateLayer,
    removeLayer,
    addConnection,
    removeConnection,
    clearModel
  };
}

export const modelStore = createModelStore();