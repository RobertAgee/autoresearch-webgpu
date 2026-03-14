<script lang="ts">
	import { DEFAULT_CONFIG, estimateParams, type ExperimentConfig, type ParamConstraints } from '$lib/model/config';

	let { config = $bindable(), disabled = false, constraints = {} }: { config: ExperimentConfig; disabled?: boolean; constraints?: ParamConstraints } =
		$props();

	let paramCount = $derived(estimateParams(config));

	type Field = { key: keyof ExperimentConfig; label: string; min: number; max: number; step: number };
	const fields: Field[] = [
		{ key: 'nLayer', label: 'layers', min: 1, max: 12, step: 1 },
		{ key: 'nEmbd', label: 'd_model', min: 32, max: 512, step: 32 },
		{ key: 'nHead', label: 'heads', min: 1, max: 16, step: 1 },
		{ key: 'mlpRatio', label: 'mlp_ratio', min: 1, max: 8, step: 1 },
		{ key: 'seqLen', label: 'seq_len', min: 32, max: 512, step: 32 },
		{ key: 'batchSize', label: 'batch_size', min: 1, max: 64, step: 1 },
		{ key: 'lr', label: 'lr', min: 0.00001, max: 0.01, step: 0.0001 },
		{ key: 'weightDecay', label: 'weight_decay', min: 0, max: 1, step: 0.01 },
		{ key: 'trainSeconds', label: 'train_sec', min: 10, max: 300, step: 10 },
	];

	function effectiveMax(f: Field): number {
		const c = constraints[f.key];
		return c?.max != null ? Math.min(f.max, c.max) : f.max;
	}

	function effectiveMin(f: Field): number {
		const c = constraints[f.key];
		return c?.min != null ? Math.max(f.min, c.min) : f.min;
	}

	let maxParamsConstraint = $derived(constraints.maxParams?.max);
	let overMaxParams = $derived(maxParamsConstraint != null && paramCount > maxParamsConstraint);
</script>

<div class="space-y-1 text-xs font-mono">
	<div class="text-gray-400 {overMaxParams ? 'text-yellow-400' : ''}">
		~{(paramCount / 1e6).toFixed(2)}M params{#if maxParamsConstraint != null} <span class="text-gray-600">/ {(maxParamsConstraint / 1e6).toFixed(1)}M</span>{/if}
	</div>

	<div class="space-y-0.5">
		{#each fields as f}
			<label class="flex items-center justify-between py-0.5">
				<span class="text-gray-400">{f.label}</span>
				<input
					type="number"
					bind:value={config[f.key]}
					min={effectiveMin(f)}
					max={effectiveMax(f)}
					step={f.step}
					{disabled}
					class="w-16 bg-gray-800 border border-gray-700 rounded px-1.5 py-px text-right tabular-nums text-xs"
				/>
			</label>
		{/each}

		<label class="flex items-center justify-between py-0.5">
			<span class="text-gray-400">activation</span>
			<select bind:value={config.activation} {disabled} class="w-16 bg-gray-800 border border-gray-700 rounded px-1.5 py-px text-right appearance-none cursor-pointer text-xs">
				<option value="relu_sq">relu²</option>
				<option value="gelu">gelu</option>
				<option value="silu">silu</option>
			</select>
		</label>
	</div>

	<button
		onclick={() => (config = { ...DEFAULT_CONFIG })}
		{disabled}
		class="text-[10px] text-gray-500 hover:text-gray-300"
	>
		reset defaults
	</button>
</div>
