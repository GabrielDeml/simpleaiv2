<!-- 
  This is the main page of the SvelteKit application
  In SvelteKit, files in the routes directory automatically become pages
  +page.svelte is the default page component for a route
-->

<script>
  // Import components from the lib directory
  // $lib is a SvelteKit alias that points to src/lib
  import MnistTrainer from '$lib/components/MnistTrainer.svelte';
</script>

<!-- The HTML content of the page -->
<h1>Simple AI - MNIST CNN Demo</h1>

<!-- 
  Using a Svelte component is just like using an HTML element
  The component will be rendered here with all its functionality
-->
<MnistTrainer />

<!-- 
  Key Svelte/SvelteKit concepts shown here:
  
  1. File-based routing: This file at src/routes/+page.svelte becomes the "/" route
  2. $lib alias: Convenient way to import from src/lib without relative paths
  3. Component usage: Import and use components like HTML elements
  4. No explicit export needed: SvelteKit handles the page component export
  
  When you run the app, this page will:
  - Display the title
  - Render the MnistTrainer component below it
  - The MnistTrainer handles all the MNIST training logic independently
-->