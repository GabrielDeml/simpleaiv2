<script lang="ts">
  import { getAvailableTemplates, loadTemplate } from '$lib/nn-designer/stores';
  import type { ModelTemplate } from '$lib/nn-designer/types';
  import ConfirmDialog from '../ConfirmDialog.svelte';
  import { showSuccess } from '$lib/stores/toastStore';

  // Get all available templates
  const templates = getAvailableTemplates();
  
  // Group templates by category
  const templatesByCategory = templates.reduce((acc, template) => {
    if (!acc[template.category]) {
      acc[template.category] = [];
    }
    acc[template.category].push(template);
    return acc;
  }, {} as Record<string, ModelTemplate[]>);

  // Handle template selection
  let selectedTemplate: ModelTemplate | null = null;
  let showConfirmLoad = false;
  
  function handleTemplateSelect(template: ModelTemplate) {
    selectedTemplate = template;
    showConfirmLoad = true;
  }
  
  function confirmLoad() {
    if (selectedTemplate) {
      loadTemplate(selectedTemplate);
      showSuccess(`Loaded "${selectedTemplate.name}" template successfully!`);
      showConfirmLoad = false;
      selectedTemplate = null;
    }
  }
  
  function cancelLoad() {
    showConfirmLoad = false;
    selectedTemplate = null;
  }
  
  // Collapsible state
  let isExpanded = false;
  
  // Category display names
  const categoryNames: Record<string, string> = {
    'classification': 'Classification',
    'computer-vision': 'Computer Vision',
    'custom': 'Custom'
  };
  
  // Category icons
  const categoryIcons: Record<string, string> = {
    'classification': '📊',
    'computer-vision': '🖼️',
    'custom': '⚙️'
  };
</script>

<div class="templates-section">
  <button class="section-header" on:click={() => isExpanded = !isExpanded}>
    <span class="header-text">MODEL TEMPLATES</span>
    <span class="expand-icon" class:expanded={isExpanded}>▼</span>
  </button>
  
  {#if isExpanded}
    <div class="templates-content">
      {#each Object.entries(templatesByCategory) as [category, categoryTemplates]}
        <div class="category-group">
          <div class="category-title">
            <span class="category-icon">{categoryIcons[category] || '📋'}</span>
            <span class="category-name">{categoryNames[category] || category}</span>
          </div>
          
          <div class="template-list">
            {#each categoryTemplates as template}
              <button 
                class="template-card" 
                on:click={() => handleTemplateSelect(template)}
                title={template.description}
              >
                <div class="template-info">
                  <span class="template-name">{template.name}</span>
                  <span class="layer-count">{template.layers.length}</span>
                </div>
              </button>
            {/each}
          </div>
        </div>
      {/each}
    </div>
  {/if}
  
  {#if showConfirmLoad && selectedTemplate}
    <ConfirmDialog
      title="Load Template"
      message={`Load "${selectedTemplate.name}" template? This will replace your current model.`}
      confirmText="Load"
      cancelText="Cancel"
      type="warning"
      onconfirm={confirmLoad}
      oncancel={cancelLoad}
    />
  {/if}
</div>

<style>
  .templates-section {
    border-bottom: 1px solid #262626;
  }

  .section-header {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    background: transparent;
    border: none;
    color: #737373;
    cursor: pointer;
    transition: color 0.2s;
  }

  .section-header:hover {
    color: #a3a3a3;
  }

  .header-text {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.5px;
  }

  .expand-icon {
    font-size: 10px;
    transition: transform 0.2s;
  }

  .expand-icon.expanded {
    transform: rotate(180deg);
  }

  .templates-content {
    padding: 0 24px 16px 24px;
  }

  .category-group {
    margin-bottom: 12px;
  }

  .category-title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 500;
    color: #a3a3a3;
    margin-bottom: 8px;
  }

  .category-icon {
    font-size: 12px;
  }

  .category-name {
    font-size: 11px;
  }

  .template-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .template-card {
    width: 100%;
    padding: 8px 12px;
    background: #171717;
    border: 1px solid #262626;
    border-radius: 4px;
    color: #ffffff;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .template-card:hover {
    background: #1f1f1f;
    border-color: #404040;
  }

  .template-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .template-name {
    font-size: 12px;
    font-weight: 500;
    color: #ffffff;
  }

  .layer-count {
    font-size: 10px;
    color: #737373;
    background: #262626;
    padding: 2px 6px;
    border-radius: 8px;
  }
</style>