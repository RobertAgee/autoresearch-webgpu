<script lang="ts">
	import type { ExperimentRecord } from '$lib/research/prompt';

	let { experiments }: { experiments: ExperimentRecord[] } = $props();

	let sorted = $derived([...experiments].sort((a, b) => a.valBpb - b.valBpb).slice(0, 10));
</script>

<div class="space-y-1">
	{#each sorted as exp, i}
		<div class="flex items-center justify-between font-mono text-xs px-2 py-1 rounded {i === 0 ? 'bg-green-950/50 text-green-300' : 'text-gray-400'}">
			<span>#{exp.id}</span>
			<span class="tabular-nums">{exp.valBpb.toFixed(4)}</span>
			<span class="text-gray-500">{exp.totalSteps}st</span>
		</div>
	{/each}

	{#if experiments.length === 0}
		<p class="text-gray-500 text-xs font-mono px-2">no experiments yet</p>
	{/if}
</div>
