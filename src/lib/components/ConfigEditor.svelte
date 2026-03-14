<script lang="ts">
	import { DEFAULT_CONFIG, estimateParams, type ExperimentConfig, type ParamConstraints } from '$lib/model/config';
	import Tooltip from './Tooltip.svelte';

	let { config = $bindable(), disabled = false, constraints = {} }: { config: ExperimentConfig; disabled?: boolean; constraints?: ParamConstraints } =
		$props();

	let paramCount = $derived(estimateParams(config));

	type Field = { key: keyof ExperimentConfig; label: string; min: number; max: number; step: number; tip: string };
	const fields: Field[] = [
		{ key: 'nLayer', label: 'layers', min: 1, max: 12, step: 1, tip: 'Number of transformer blocks stacked sequentially. More layers let the model learn more complex representations, but increase training time.' },
		{ key: 'nEmbd', label: 'd_model', min: 32, max: 512, step: 32, tip: 'Embedding dimensionality — the width of the residual stream. This is the core size of the model; larger values are more expressive but add parameters quickly.' },
		{ key: 'nHead', label: 'heads', min: 1, max: 16, step: 1, tip: 'Number of parallel attention heads. Each head attends to different parts of the input, letting the model capture diverse patterns simultaneously.' },
		{ key: 'mlpRatio', label: 'mlp_ratio', min: 1, max: 8, step: 1, tip: 'Expansion factor for the feed-forward layer (hidden dim = d_model * mlp_ratio). Adds capacity per layer without increasing depth.' },
		{ key: 'seqLen', label: 'seq_len', min: 32, max: 512, step: 32, tip: 'Context window — how many tokens the model sees at once during training. Longer sequences capture longer-range dependencies but use more memory.' },
		{ key: 'batchSize', label: 'batch_size', min: 1, max: 64, step: 1, tip: 'Sequences per gradient update. Larger batches produce smoother gradient estimates but require more memory per step.' },
		{ key: 'lr', label: 'lr', min: 0.00001, max: 0.01, step: 0.0001, tip: 'Learning rate — step size for weight updates. Too high causes instability; too low means slow convergence.' },
		{ key: 'weightDecay', label: 'weight_decay', min: 0, max: 1, step: 0.01, tip: 'L2 regularization that penalizes large weights, helping prevent overfitting. Higher values apply stronger regularization.' },
		{ key: 'trainSeconds', label: 'train_sec', min: 10, max: 300, step: 10, tip: 'Wall-clock training budget. Training stops after this many seconds regardless of convergence.' },
	];

	const activationTip = 'Non-linear function applied inside each transformer layer. relu² (squared ReLU), gelu (Gaussian Error Linear Unit), and silu (Sigmoid Linear Unit) each affect gradient flow and expressiveness differently.';

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
	<div class="text-gray-400 {overMaxParams ? 'text-red-400' : ''}">
		~{(paramCount / 1e6).toFixed(2)}M params{#if maxParamsConstraint != null} <span class="text-gray-600">/ {(maxParamsConstraint / 1e6).toFixed(1)}M</span>{/if}
	</div>

	<div class="space-y-0.5">
		{#each fields as f}
			<label class="flex items-center justify-between py-0.5">
				<Tooltip text={f.tip}>
					<span class="text-gray-400 decoration-dotted underline underline-offset-2 decoration-gray-600 hover:text-gray-200 hover:decoration-gray-400 cursor-help transition-colors">{f.label}</span>
				</Tooltip>
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
			<Tooltip text={activationTip}>
				<span class="text-gray-400 decoration-dotted underline underline-offset-2 decoration-gray-600 hover:text-gray-200 hover:decoration-gray-400 cursor-help transition-colors">activation</span>
			</Tooltip>
			<select bind:value={config.activation} {disabled} class="w-16 bg-gray-800 border border-gray-700 rounded px-1.5 py-px text-right appearance-none cursor-pointer text-xs">
				<option value="relu_sq">relu²</option>
				<option value="gelu">gelu</option>
				<option value="silu">silu</option>
			</select>
		</label>
	</div>
</div>
