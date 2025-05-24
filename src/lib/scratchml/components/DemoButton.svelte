<script>
  import { modelStore } from '../modelStore';
  
  function createDemoModel() {
    // Clear existing model
    modelStore.clearModel();
    
    // Add layers
    const inputId = modelStore.addLayer('input', { x: 100, y: 100 });
    const conv1Id = modelStore.addLayer('conv2d', { x: 300, y: 100 });
    const pool1Id = modelStore.addLayer('maxPooling2d', { x: 500, y: 100 });
    const conv2Id = modelStore.addLayer('conv2d', { x: 700, y: 100 });
    const pool2Id = modelStore.addLayer('maxPooling2d', { x: 900, y: 100 });
    const flattenId = modelStore.addLayer('flatten', { x: 500, y: 250 });
    const dense1Id = modelStore.addLayer('dense', { x: 300, y: 400 });
    const dropoutId = modelStore.addLayer('dropout', { x: 500, y: 400 });
    const dense2Id = modelStore.addLayer('dense', { x: 700, y: 400 });
    
    // Configure layers
    modelStore.updateLayer(conv1Id, { params: { filters: 32, kernelSize: 3 } });
    modelStore.updateLayer(conv2Id, { params: { filters: 64, kernelSize: 3 } });
    modelStore.updateLayer(dense1Id, { params: { units: 128 } });
    modelStore.updateLayer(dense2Id, { params: { units: 10, activation: 'softmax' } });
    modelStore.updateLayer(dropoutId, { params: { rate: 0.5 } });
    
    // Add connections
    modelStore.addConnection(inputId, conv1Id);
    modelStore.addConnection(conv1Id, pool1Id);
    modelStore.addConnection(pool1Id, conv2Id);
    modelStore.addConnection(conv2Id, pool2Id);
    modelStore.addConnection(pool2Id, flattenId);
    modelStore.addConnection(flattenId, dense1Id);
    modelStore.addConnection(dense1Id, dropoutId);
    modelStore.addConnection(dropoutId, dense2Id);
  }
</script>

<button 
  on:click={createDemoModel}
  class="px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded font-medium transition-colors"
>
  Load Demo CNN Model
</button>