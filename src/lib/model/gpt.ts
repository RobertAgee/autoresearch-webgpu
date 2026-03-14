import { numpy as np, nn, random } from '@jax-js/jax';
import type { ExperimentConfig, Activation } from './config';

export type Params = { [key: string]: np.Array };

export function initParams(config: ExperimentConfig, key: np.Array): Params {
	const { nLayer, nEmbd, nHead, mlpRatio, vocabSize } = config;
	const headDim = nEmbd / nHead;
	const mlpHidden = nEmbd * mlpRatio;
	const params: Params = {};

	const numKeys = 3 + nLayer * 8;
	const keys = random.split(key, numKeys);
	let ki = 0;

	// Helper: grab a key from `keys`, using .ref to keep it alive for future slices.
	// The last call should use `keys` directly (no .ref) to consume it.
	const grabKey = () => {
		ki++;
		if (ki < numKeys) return keys.ref.slice(ki - 1);
		return keys.slice(ki - 1); // last use, consume keys
	};

	params['embed'] = random.normal(grabKey(), [vocabSize, nEmbd]).mul(1.0);

	for (let i = 0; i < nLayer; i++) {
		const s = Math.sqrt(3) * Math.pow(nEmbd, -0.5);
		const prefix = `layer${i}`;

		params[`${prefix}.attn.wq`] = random.uniform(grabKey(), [nEmbd, nHead * headDim], {
			minval: -s, maxval: s
		});
		params[`${prefix}.attn.wk`] = random.uniform(grabKey(), [nEmbd, nHead * headDim], {
			minval: -s, maxval: s
		});
		params[`${prefix}.attn.wv`] = random.uniform(grabKey(), [nEmbd, nHead * headDim], {
			minval: -s, maxval: s
		});
		params[`${prefix}.attn.wout`] = np.zeros([nEmbd, nEmbd]);
		grabKey(); // consume the key even though we don't use it

		params[`${prefix}.norm1`] = np.ones([nEmbd]);
		grabKey();

		params[`${prefix}.norm2`] = np.ones([nEmbd]);
		grabKey();

		params[`${prefix}.mlp.up`] = random.uniform(grabKey(), [nEmbd, mlpHidden], {
			minval: -s, maxval: s
		});
		params[`${prefix}.mlp.down`] = np.zeros([mlpHidden, nEmbd]);
		grabKey();
	}

	params['final_norm'] = np.ones([nEmbd]);
	params['unembed'] = random.normal(grabKey(), [nEmbd, vocabSize]).mul(0.001);

	return params;
}

function rmsNorm(x: np.Array, weight: np.Array): np.Array {
	return nn.standardize(x, -1, { epsilon: 1e-6 }).mul(weight);
}

function ropeFreqs(seqLen: number, headDim: number): [np.Array, np.Array] {
	const halfDim = headDim / 2;
	const freqExponents = np.arange(0, halfDim, 1, { dtype: np.float32 }).mul(2 / headDim);
	const invFreq = np.power(10000, np.negative(freqExponents));
	const positions = np.arange(0, seqLen, 1, { dtype: np.float32 });
	const angles = np.outer(positions, invFreq);
	return [np.cos(angles.ref), np.sin(angles)];
}

function applyRoPE(x: np.Array, cos: np.Array, sin: np.Array): np.Array {
	const half = x.shape[3] / 2;
	const x1 = x.ref.slice([], [], [], [0, half]);
	const x2 = x.slice([], [], [], [half]);
	const c = cos.reshape([1, -1, 1, half]);
	const s = sin.reshape([1, -1, 1, half]);
	return np.concatenate([
		x1.ref.mul(c.ref).sub(x2.ref.mul(s.ref)),
		x1.mul(c).add(x2.mul(s))
	], -1);
}

function activate(x: np.Array, activation: Activation): np.Array {
	switch (activation) {
		case 'relu_sq':
			return np.square(nn.relu(x));
		case 'gelu':
			return nn.gelu(x);
		case 'silu':
			return nn.silu(x);
	}
}

export function forward(
	params: Params,
	config: ExperimentConfig,
	inputIds: np.Array
): np.Array {
	const { nLayer, nHead, nEmbd, mlpRatio, activation, useRoPE, softcapValue } = config;
	const headDim = nEmbd / nHead;
	const [_batch, seqLen] = inputIds.shape;

	// oneHot + matmul instead of gather (gather transpose not implemented in jax-js)
	const oneHotIds = nn.oneHot(inputIds.reshape([-1]), config.vocabSize);
	let x = np.dot(oneHotIds, params['embed'].ref).reshape([-1, seqLen, nEmbd]);

	let ropeCos: np.Array | null = null;
	let ropeSin: np.Array | null = null;
	if (useRoPE) {
		[ropeCos, ropeSin] = ropeFreqs(seqLen, headDim);
	}

	for (let i = 0; i < nLayer; i++) {
		const prefix = `layer${i}`;
		const isLastLayer = i === nLayer - 1;

		// x is reused for residual connection: .ref for rmsNorm, consume in .add
		const normed = rmsNorm(x.ref, params[`${prefix}.norm1`].ref);

		// normed is used 3 times (q, k, v projections)
		let q = np.dot(normed.ref, params[`${prefix}.attn.wq`].ref).reshape([-1, seqLen, nHead, headDim]);
		let k = np.dot(normed.ref, params[`${prefix}.attn.wk`].ref).reshape([-1, seqLen, nHead, headDim]);
		const v = np.dot(normed, params[`${prefix}.attn.wv`].ref).reshape([-1, seqLen, nHead, headDim]);

		if (useRoPE && ropeCos && ropeSin) {
			// .ref on rope arrays if not last layer (they're reused across layers)
			const cosArg = isLastLayer ? ropeCos : ropeCos.ref;
			const sinArg = isLastLayer ? ropeSin : ropeSin.ref;
			q = applyRoPE(q, cosArg.ref, sinArg.ref);
			k = applyRoPE(k, cosArg, sinArg);
		}

		const attnOut = nn.dotProductAttention(q, k, v, { isCausal: true });
		const projected = np.dot(attnOut.reshape([-1, seqLen, nEmbd]), params[`${prefix}.attn.wout`].ref);
		x = x.add(projected);

		const normed2 = rmsNorm(x.ref, params[`${prefix}.norm2`].ref);
		let h = np.dot(normed2, params[`${prefix}.mlp.up`].ref);
		h = activate(h, activation);
		h = np.dot(h, params[`${prefix}.mlp.down`].ref);
		x = x.add(h);
	}

	x = rmsNorm(x, params['final_norm'].ref);
	let logits = np.dot(x, params['unembed'].ref);

	if (softcapValue > 0) {
		logits = np.tanh(logits.mul(1 / softcapValue)).mul(softcapValue);
	}

	return logits;
}

export function lossFn(
	params: Params,
	config: ExperimentConfig,
	inputIds: np.Array,
	targetIds: np.Array
): np.Array {
	const logits = forward(params, config, inputIds);
	const batchSize = targetIds.shape[0];
	const seqLen = targetIds.shape[1];

	const logProbs = nn.logSoftmax(logits, -1);
	const targets = nn.oneHot(targetIds.reshape([-1]), config.vocabSize);
	const flatLogProbs = logProbs.reshape([-1, config.vocabSize]);

	return flatLogProbs.mul(targets).sum().mul(-1 / (batchSize * seqLen));
}
