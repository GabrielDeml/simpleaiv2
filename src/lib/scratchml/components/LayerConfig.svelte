<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { modelStore } from '../modelStore';
  import { getLayerType } from '../layers';
  import type { ParamSchema } from '../types';
  
  export let layerId: string | null = null;
  
  const dispatch = createEventDispatcher();
  const { layers } = modelStore;
  
  $: layer = layerId ? $layers.get(layerId) : null;
  $: layerType = layer ? getLayerType(layer.type) : null;
  $: params = layer ? { ...layerType?.defaultParams, ...layer.params } : {};
  
  function handleParamChange(paramName: string, value: any) {
    if (layer && layerId) {
      modelStore.updateLayer(layerId, {
        params: { ...layer.params, [paramName]: value }
      });
    }
  }
  
  function handleClose() {
    dispatch('close');
  }
  
  function renderParam(schema: ParamSchema) {
    const value = params[schema.name] ?? schema.default;
    
    switch (schema.type) {
      case 'number':
        return { component: 'number', value, min: schema.min, max: schema.max, step: schema.step };
      case 'select':
        return { component: 'select', value, options: schema.options };
      case 'boolean':
        return { component: 'checkbox', value };
      case 'array':
        return { component: 'array', value };
      default:
        return { component: 'text', value };
    }
  }
</script>

{#if layer && layerType}
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-96 max-h-[80vh] overflow-hidden flex flex-col">
    <div class="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
      <h3 class="text-lg font-semibold">{layerType.name} Configuration</h3>
      <button
        on:click={handleClose}
        class="text-2xl text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 w-8 h-8 flex items-center justify-center rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
        aria-label="Close"
      >
        ×
      </button>
    </div>
    
    <div class="p-4 overflow-y-auto flex-1">
      <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">{layerType.description}</p>
      
      <div class="space-y-4">
        {#each layerType.paramSchema as schema}
          {@const param = renderParam(schema)}
          <div class="space-y-2">
            <label for={schema.name} class="block text-sm font-medium text-gray-700 dark:text-gray-300">
              {schema.label}
              {#if schema.description}
                <span class="block text-xs text-gray-500 dark:text-gray-500 font-normal mt-1">{schema.description}</span>
              {/if}
            </label>
            
            {#if param.component === 'number'}
              <input
                type="number"
                id={schema.name}
                value={param.value}
                min={param.min}
                max={param.max}
                step={param.step}
                on:change={(e) => handleParamChange(schema.name, Number(e.currentTarget.value))}
                class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
            {:else if param.component === 'select'}
              <select
                id={schema.name}
                value={param.value}
                on:change={(e) => handleParamChange(schema.name, e.currentTarget.value)}
                class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              >
                {#each param.options || [] as option}
                  <option value={option.value}>{option.label}</option>
                {/each}
              </select>
            {:else if param.component === 'checkbox'}
              <input
                type="checkbox"
                id={schema.name}
                checked={param.value}
                on:change={(e) => handleParamChange(schema.name, e.currentTarget.checked)}
                class="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
              />
            {:else if param.component === 'array'}
              <input
                type="text"
                id={schema.name}
                value={JSON.stringify(param.value)}
                on:change={(e) => {
                  try {
                    const val = JSON.parse(e.currentTarget.value);
                    handleParamChange(schema.name, val);
                  } catch {}
                }}
                class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                placeholder="[28, 28, 1]"
              />
            {:else}
              <input
                type="text"
                id={schema.name}
                value={param.value}
                on:change={(e) => handleParamChange(schema.name, e.currentTarget.value)}
                class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
            {/if}
          </div>
        {/each}
      </div>
      
      {#if layerType.paramSchema.length === 0}
        <p class="text-sm text-gray-500 dark:text-gray-400 italic">This layer has no configurable parameters.</p>
      {/if}
    </div>
  </div>
{/if}