export type ResearchPhase = 'representation' | 'model' | 'optimizer' | 'mixed';

export type ResearchActivation = 'relu2' | 'gelu' | 'silu';

export type ResearchConfig = {
	seed: number;
	nLayer: number;
	nEmbd: number;
	nHead: number;
	mlpRatio: number;
	activation: ResearchActivation;
	softcap: number;
	batchSize: number;
	seqLen: number;
	lr: number;
	weightDecay: number;
	beta1: number;
	beta2: number;
	warmupRatio: number;
	cooldownRatio: number;
	embedInitScale: number;
	unembedInitScale: number;
	projInitScale: number;
	residualInitScale: number;
};

export type ResearchProposal = {
	reasoning: string;
	changes?: Partial<ResearchConfig>;
	config?: Partial<ResearchConfig>;
};

const CONFIG_PREFIX = '// RESEARCH_CONFIG ';

const PHASE_FIELDS: Record<ResearchPhase, (keyof ResearchConfig)[]> = {
	representation: [
		'batchSize',
		'seqLen',
		'activation',
		'softcap',
		'embedInitScale',
		'unembedInitScale',
		'projInitScale',
		'residualInitScale'
	],
	model: ['nLayer', 'nEmbd', 'nHead', 'mlpRatio'],
	optimizer: ['lr', 'weightDecay', 'beta1', 'beta2', 'warmupRatio', 'cooldownRatio'],
	mixed: [
		'batchSize',
		'seqLen',
		'activation',
		'softcap',
		'embedInitScale',
		'unembedInitScale',
		'projInitScale',
		'residualInitScale',
		'nLayer',
		'nEmbd',
		'nHead',
		'mlpRatio',
		'lr',
		'weightDecay',
		'beta1',
		'beta2',
		'warmupRatio',
		'cooldownRatio'
	]
};

export const BASELINE_RESEARCH_CONFIG: ResearchConfig = {
	seed: 42,
	nLayer: 3,
	nEmbd: 96,
	nHead: 4,
	mlpRatio: 4,
	activation: 'relu2',
	softcap: 15,
	batchSize: 8,
	seqLen: 128,
	lr: 1e-3,
	weightDecay: 0.1,
	beta1: 0.9,
	beta2: 0.95,
	warmupRatio: 0.1,
	cooldownRatio: 0.3,
	embedInitScale: 1,
	unembedInitScale: 0.001,
	projInitScale: 1,
	residualInitScale: 0
};

function clamp(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}

function roundToMultiple(value: number, multiple: number): number {
	return Math.round(value / multiple) * multiple;
}

function normalizeActivation(value: unknown, fallback: ResearchActivation): ResearchActivation {
	return value === 'gelu' || value === 'silu' || value === 'relu2' ? value : fallback;
}

export function phaseFields(phase: ResearchPhase): (keyof ResearchConfig)[] {
	return PHASE_FIELDS[phase];
}

export function researchPhaseForIteration(iteration: number): ResearchPhase {
	if (iteration < 4) return 'representation';
	if (iteration < 8) return 'model';
	if (iteration < 12) return 'optimizer';
	return 'mixed';
}

export function normalizeResearchConfig(input?: Partial<ResearchConfig> | null): ResearchConfig {
	const merged = { ...BASELINE_RESEARCH_CONFIG, ...(input ?? {}) };
	const nHead = Math.max(1, Math.round(Number(merged.nHead) || BASELINE_RESEARCH_CONFIG.nHead));
	const minWidth = nHead * 16;
	const nEmbd = Math.max(
		minWidth,
		roundToMultiple(Math.round(Number(merged.nEmbd) || BASELINE_RESEARCH_CONFIG.nEmbd), nHead)
	);

	return {
		seed: Math.max(1, Math.round(Number(merged.seed) || BASELINE_RESEARCH_CONFIG.seed)),
		nLayer: clamp(Math.round(Number(merged.nLayer) || BASELINE_RESEARCH_CONFIG.nLayer), 1, 6),
		nEmbd: clamp(nEmbd, minWidth, 320),
		nHead,
		mlpRatio: clamp(Math.round(Number(merged.mlpRatio) || BASELINE_RESEARCH_CONFIG.mlpRatio), 2, 6),
		activation: normalizeActivation(merged.activation, BASELINE_RESEARCH_CONFIG.activation),
		softcap: clamp(Number(merged.softcap) || BASELINE_RESEARCH_CONFIG.softcap, 8, 30),
		batchSize: clamp(Math.round(Number(merged.batchSize) || BASELINE_RESEARCH_CONFIG.batchSize), 2, 16),
		seqLen: clamp(roundToMultiple(Math.round(Number(merged.seqLen) || BASELINE_RESEARCH_CONFIG.seqLen), 32), 64, 256),
		lr: clamp(Number(merged.lr) || BASELINE_RESEARCH_CONFIG.lr, 1e-4, 3e-3),
		weightDecay: clamp(Number(merged.weightDecay) || BASELINE_RESEARCH_CONFIG.weightDecay, 0, 0.25),
		beta1: clamp(Number(merged.beta1) || BASELINE_RESEARCH_CONFIG.beta1, 0.8, 0.99),
		beta2: clamp(Number(merged.beta2) || BASELINE_RESEARCH_CONFIG.beta2, 0.9, 0.999),
		warmupRatio: clamp(Number(merged.warmupRatio) || BASELINE_RESEARCH_CONFIG.warmupRatio, 0, 0.4),
		cooldownRatio: clamp(Number(merged.cooldownRatio) || BASELINE_RESEARCH_CONFIG.cooldownRatio, 0, 0.6),
		embedInitScale: clamp(Number(merged.embedInitScale) || BASELINE_RESEARCH_CONFIG.embedInitScale, 0.05, 2),
		unembedInitScale: clamp(Number(merged.unembedInitScale) || BASELINE_RESEARCH_CONFIG.unembedInitScale, 0.0001, 0.05),
		projInitScale: clamp(Number(merged.projInitScale) || BASELINE_RESEARCH_CONFIG.projInitScale, 0.1, 2),
		residualInitScale: clamp(Number(merged.residualInitScale) || BASELINE_RESEARCH_CONFIG.residualInitScale, 0, 0.5)
	};
}

function sortedEntries(config: ResearchConfig): [keyof ResearchConfig, ResearchConfig[keyof ResearchConfig]][] {
	return Object.entries(config)
		.sort(([a], [b]) => a.localeCompare(b))
		.map(([key, value]) => [key as keyof ResearchConfig, value as ResearchConfig[keyof ResearchConfig]]);
}

export function summarizeResearchConfig(config: ResearchConfig): string {
	return [
		`shape L${config.nLayer} d${config.nEmbd} h${config.nHead} mlp${config.mlpRatio}`,
		`train bs${config.batchSize} ctx${config.seqLen} act=${config.activation} softcap=${config.softcap}`,
		`opt lr=${config.lr.toExponential(2)} wd=${config.weightDecay.toFixed(3)} betas=(${config.beta1.toFixed(2)},${config.beta2.toFixed(3)}) warm=${config.warmupRatio.toFixed(2)} cool=${config.cooldownRatio.toFixed(2)}`,
		`init emb=${config.embedInitScale.toFixed(3)} proj=${config.projInitScale.toFixed(3)} resid=${config.residualInitScale.toFixed(3)} unemb=${config.unembedInitScale.toFixed(4)} seed=${config.seed}`
	].join(' | ');
}

export function diffResearchConfig(base: ResearchConfig, next: ResearchConfig): string[] {
	const changes: string[] = [];
	for (const [key, value] of sortedEntries(next)) {
		if (base[key] !== value) {
			changes.push(`${String(key)}=${String(value)}`);
		}
	}
	return changes;
}

export function applyResearchProposal(
	base: ResearchConfig,
	proposal: ResearchProposal,
	phase: ResearchPhase
): { config: ResearchConfig; changedKeys: (keyof ResearchConfig)[] } {
	const rawChanges = (proposal.changes ?? proposal.config ?? {}) as Record<string, unknown>;
	const allowed = new Set<string>(phaseFields(phase));
	const acceptedEntries = Object.entries(rawChanges).filter(([key]) => allowed.has(key));

	if (acceptedEntries.length === 0) {
		throw new Error(`Proposal did not change any allowed ${phase} fields.`);
	}

	const maxChanges = phase === 'mixed' ? 4 : 3;
	if (acceptedEntries.length > maxChanges) {
		throw new Error(`Proposal changed ${acceptedEntries.length} fields; max ${maxChanges} allowed in ${phase} phase.`);
	}

	const changedKeys = acceptedEntries
		.filter(([key, value]) => base[key as keyof ResearchConfig] !== value)
		.map(([key]) => key as keyof ResearchConfig);

	if (changedKeys.length === 0) {
		throw new Error('Proposal made no effective changes relative to the current champion.');
	}

	const config = normalizeResearchConfig({
		...base,
		...(Object.fromEntries(acceptedEntries) as Partial<ResearchConfig>)
	});

	return { config, changedKeys };
}

function activationLine(activation: ResearchActivation): string {
	switch (activation) {
		case 'gelu':
			return '    h = nn.gelu(h);';
		case 'silu':
			return '    h = nn.silu(h);';
		case 'relu2':
		default:
			return '    h = np.square(nn.relu(h));';
	}
}

function residualInitLine(name: string, rows: string, cols: string, scale: number): string {
	if (scale <= 0) {
		return `  params[p + '.${name}'] = np.zeros([${rows}, ${cols}]); grabKey();`;
	}
	return `  params[p + '.${name}'] = random.uniform(grabKey(), [${rows}, ${cols}], { minval: -residualScale, maxval: residualScale });`;
}

export function buildTrainCodeFromConfig(input: ResearchConfig): string {
	const config = normalizeResearchConfig(input);
	const configJson = JSON.stringify(config);
	const lines = [
		`${CONFIG_PREFIX}${configJson}`,
		`const cfg = ${configJson};`,
		'const nLayer = cfg.nLayer, nEmbd = cfg.nEmbd, nHead = cfg.nHead, mlpRatio = cfg.mlpRatio;',
		'const headDim = nEmbd / nHead, mlpHidden = nEmbd * mlpRatio;',
		'const lr = cfg.lr, weightDecay = cfg.weightDecay, warmupRatio = cfg.warmupRatio, cooldownRatio = cfg.cooldownRatio;',
		'const batchSize = cfg.batchSize, seqLen = cfg.seqLen, softcap = cfg.softcap;',
		'const embedInitScale = cfg.embedInitScale, unembedInitScale = cfg.unembedInitScale;',
		'const projInitScale = cfg.projInitScale, residualInitScale = cfg.residualInitScale;',
		'',
		'const key = random.key(cfg.seed);',
		'const numKeys = 3 + nLayer * 8;',
		'const keys = random.split(key, numKeys);',
		'let ki = 0;',
		"const grabKey = () => { ki++; return ki < numKeys ? keys.ref.slice(ki - 1) : keys.slice(ki - 1); };",
		'',
		'const params = {};',
		"params['embed'] = random.normal(grabKey(), [VOCAB_SIZE, nEmbd]).mul(embedInitScale);",
		'const projScale = Math.sqrt(3) * Math.pow(nEmbd, -0.5) * projInitScale;',
		'const residualScale = Math.sqrt(3) * Math.pow(nEmbd, -0.5) * residualInitScale;',
		'for (let i = 0; i < nLayer; i++) {',
		"  const p = 'layer' + i;",
		"  params[p + '.attn.wq'] = random.uniform(grabKey(), [nEmbd, nHead * headDim], { minval: -projScale, maxval: projScale });",
		"  params[p + '.attn.wk'] = random.uniform(grabKey(), [nEmbd, nHead * headDim], { minval: -projScale, maxval: projScale });",
		"  params[p + '.attn.wv'] = random.uniform(grabKey(), [nEmbd, nHead * headDim], { minval: -projScale, maxval: projScale });",
		residualInitLine('attn.wout', 'nEmbd', 'nEmbd', config.residualInitScale),
		"  params[p + '.norm1'] = np.ones([nEmbd]); grabKey();",
		"  params[p + '.norm2'] = np.ones([nEmbd]); grabKey();",
		"  params[p + '.mlp.up'] = random.uniform(grabKey(), [nEmbd, mlpHidden], { minval: -projScale, maxval: projScale });",
		residualInitLine('mlp.down', 'mlpHidden', 'nEmbd', config.residualInitScale),
		'}',
		"params['final_norm'] = np.ones([nEmbd]);",
		"params['unembed'] = random.normal(grabKey(), [nEmbd, VOCAB_SIZE]).mul(unembedInitScale);",
		'await blockUntilReady(params);',
		'',
		'function rmsNorm(x, w) { return nn.standardize(x, -1, { epsilon: 1e-6 }).mul(w); }',
		'',
		'function ropeFreqs(sl, hd) {',
		'  const half = hd / 2;',
		'  const freqs = np.power(10000, np.negative(np.arange(0, half, 1, { dtype: np.float32 }).mul(2 / hd)));',
		'  const pos = np.arange(0, sl, 1, { dtype: np.float32 });',
		'  const angles = np.outer(pos, freqs);',
		'  return [np.cos(angles.ref), np.sin(angles)];',
		'}',
		'',
		'function applyRoPE(x, cos, sin) {',
		'  const half = x.shape[3] / 2;',
		'  const x1 = x.ref.slice([], [], [], [0, half]);',
		'  const x2 = x.slice([], [], [], [half]);',
		'  const c = cos.reshape([1, -1, 1, half]);',
		'  const s = sin.reshape([1, -1, 1, half]);',
		'  return np.concatenate([x1.ref.mul(c.ref).sub(x2.ref.mul(s.ref)), x1.mul(c).add(x2.mul(s))], -1);',
		'}',
		'',
		'function forward(p, inputIds) {',
		'  const [_b, sl] = inputIds.shape;',
		'  const ids = nn.oneHot(inputIds.reshape([-1]), VOCAB_SIZE);',
		"  let x = np.dot(ids, p['embed'].ref).reshape([-1, sl, nEmbd]);",
		'  const [rc, rs] = ropeFreqs(sl, headDim);',
		'  for (let i = 0; i < nLayer; i++) {',
		"    const pfx = 'layer' + i, last = i === nLayer - 1;",
		"    const n1 = rmsNorm(x.ref, p[pfx + '.norm1'].ref);",
		"    let q = np.dot(n1.ref, p[pfx + '.attn.wq'].ref).reshape([-1, sl, nHead, headDim]);",
		"    let k = np.dot(n1.ref, p[pfx + '.attn.wk'].ref).reshape([-1, sl, nHead, headDim]);",
		"    const v = np.dot(n1, p[pfx + '.attn.wv'].ref).reshape([-1, sl, nHead, headDim]);",
		'    q = applyRoPE(q, rc.ref, rs.ref);',
		'    k = applyRoPE(k, last ? rc : rc.ref, last ? rs : rs.ref);',
		'    const attn = nn.dotProductAttention(q, k, v, { isCausal: true });',
		"    x = x.add(np.dot(attn.reshape([-1, sl, nEmbd]), p[pfx + '.attn.wout'].ref));",
		"    const n2 = rmsNorm(x.ref, p[pfx + '.norm2'].ref);",
		"    let h = np.dot(n2, p[pfx + '.mlp.up'].ref);",
		activationLine(config.activation),
		"    h = np.dot(h, p[pfx + '.mlp.down'].ref);",
		'    x = x.add(h);',
		'  }',
		"  x = rmsNorm(x, p['final_norm'].ref);",
		'  return np.tanh(np.dot(x, p[\'unembed\'].ref).mul(1 / softcap)).mul(softcap);',
		'}',
		'',
		'function lossFn(p, input, target) {',
		'  const logits = forward(p, input);',
		'  const logProbs = nn.logSoftmax(logits, -1);',
		'  const targets = nn.oneHot(target.reshape([-1]), VOCAB_SIZE);',
		'  return logProbs.reshape([-1, VOCAB_SIZE]).mul(targets).sum().mul(-1 / (batchSize * seqLen));',
		'}',
		'',
		'let elapsed = 0;',
		'const optimizer = adamw(() => {',
		'  const progress = Math.min(elapsed / (trainSeconds * 1000), 1);',
		'  return lr * lrSchedule(progress, warmupRatio, cooldownRatio);',
		'}, { weightDecay, b1: cfg.beta1, b2: cfg.beta2 });',
		'',
		'let optState = optimizer.init(tree.ref(params));',
		'const lossGrad = jit(valueAndGrad(lossFn));',
		'let step = 0;',
		'const t0 = performance.now();',
		'',
		'while (elapsed < trainSeconds * 1000 && !signal.aborted) {',
		'  const batch = trainData.nextBatch(batchSize, seqLen);',
		'  const [lossVal, grads] = lossGrad(tree.ref(params), batch.input, batch.target);',
		'  const [updates, newState] = optimizer.update(grads, optState, tree.ref(params));',
		'  optState = newState;',
		'  for (const k in updates) { params[k] = applyUpdates({ [k]: params[k] }, { [k]: updates[k] })[k]; }',
		'  await blockUntilReady(params);',
		'  const loss = await lossVal.jsAsync();',
		'  elapsed = performance.now() - t0;',
		'  step++;',
		'  onStep({ step, loss, elapsed });',
		'  await yieldToUI();',
		'  if (isNaN(loss)) break;',
		'}',
		'',
		'const valBpb = await evaluate(params, forward, VOCAB_SIZE, valData, batchSize, seqLen);',
		'onReturn({ params, forward, vocabSize: VOCAB_SIZE, batchSize, seqLen, valBpb });'
	];

	return lines.join('\n');
}

export function extractResearchConfigFromCode(code: string): ResearchConfig | null {
	const match = code.match(/^\/\/ RESEARCH_CONFIG (.+)$/m);
	if (!match) return null;

	try {
		return normalizeResearchConfig(JSON.parse(match[1]) as Partial<ResearchConfig>);
	} catch {
		return null;
	}
}
