<script lang="ts">
  export let content: string = '';
  export let position: 'top' | 'bottom' | 'left' | 'right' = 'top';
  export let delay: number = 500;
  
  let showTooltip = false;
  let timeoutId: number;
  let tooltipElement: HTMLDivElement;
  let triggerElement: HTMLDivElement;
  
  function handleMouseEnter() {
    timeoutId = window.setTimeout(() => {
      showTooltip = true;
    }, delay);
  }
  
  function handleMouseLeave() {
    clearTimeout(timeoutId);
    showTooltip = false;
  }
</script>

<div 
  class="tooltip-trigger" 
  bind:this={triggerElement}
  on:mouseenter={handleMouseEnter}
  on:mouseleave={handleMouseLeave}
>
  <slot />
  {#if showTooltip && content}
    <div 
      class="tooltip tooltip-{position}" 
      bind:this={tooltipElement}
      role="tooltip"
    >
      {content}
    </div>
  {/if}
</div>

<style>
  .tooltip-trigger {
    position: relative;
    display: inline-flex;
    align-items: center;
  }
  
  .tooltip {
    position: absolute;
    z-index: 9999;
    padding: 8px 12px;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 6px;
    color: #e5e5e5;
    font-size: 12px;
    line-height: 1.4;
    white-space: normal;
    max-width: 250px;
    pointer-events: none;
    animation: tooltipFadeIn 0.2s ease-out;
  }
  
  .tooltip-top {
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
  }
  
  .tooltip-bottom {
    top: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
  }
  
  .tooltip-left {
    right: calc(100% + 8px);
    top: 50%;
    transform: translateY(-50%);
  }
  
  .tooltip-right {
    left: calc(100% + 8px);
    top: 50%;
    transform: translateY(-50%);
  }
  
  .tooltip::after {
    content: '';
    position: absolute;
    width: 0;
    height: 0;
    border: 4px solid transparent;
  }
  
  .tooltip-top::after {
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border-top-color: #1a1a1a;
  }
  
  .tooltip-bottom::after {
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    border-bottom-color: #1a1a1a;
  }
  
  .tooltip-left::after {
    left: 100%;
    top: 50%;
    transform: translateY(-50%);
    border-left-color: #1a1a1a;
  }
  
  .tooltip-right::after {
    right: 100%;
    top: 50%;
    transform: translateY(-50%);
    border-right-color: #1a1a1a;
  }
  
  @keyframes tooltipFadeIn {
    from {
      opacity: 0;
      transform: translateX(-50%) translateY(4px);
    }
    to {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
  }
  
  .tooltip-left {
    animation: tooltipFadeInLeft 0.2s ease-out;
  }
  
  .tooltip-right {
    animation: tooltipFadeInRight 0.2s ease-out;
  }
  
  @keyframes tooltipFadeInLeft {
    from {
      opacity: 0;
      transform: translateY(-50%) translateX(4px);
    }
    to {
      opacity: 1;
      transform: translateY(-50%) translateX(0);
    }
  }
  
  @keyframes tooltipFadeInRight {
    from {
      opacity: 0;
      transform: translateY(-50%) translateX(-4px);
    }
    to {
      opacity: 1;
      transform: translateY(-50%) translateX(0);
    }
  }
</style>