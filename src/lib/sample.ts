import { numpy as np, nn, blockUntilReady } from '@jax-js/jax';
import type { Params, ForwardFn } from './prepare';
import { decode } from './data/tokenizer';

export async function sampleText(
	params: Params,
	forwardFn: ForwardFn,
	vocabSize: number,
	seqLen: number,
	prompt: string = '',
	maxTokens: number = 200,
	temperature: number = 0.8,
	onToken?: (textSoFar: string) => void
): Promise<string> {
	const encoder = new TextEncoder();
	const promptBytes = prompt ? Array.from(encoder.encode(prompt)) : [0];
	const tokens: number[] = [...promptBytes];

	for (let i = 0; i < maxTokens; i++) {
		const contextStart = Math.max(0, tokens.length - seqLen);
		const context = tokens.slice(contextStart);

		const inputIds = np.array(context, { dtype: np.int32 }).reshape([1, context.length]);
		const logits = forwardFn(params, inputIds);

		const lastLogits = logits.slice(0, [-1]).reshape([vocabSize]);
		const scaled = lastLogits.mul(1 / temperature);
		const probs = nn.softmax(scaled);
		await blockUntilReady(probs);

		const probsArr = (await probs.jsAsync()) as number[];
		const nextToken = sampleFromProbs(probsArr);
		tokens.push(nextToken);

		if (onToken) {
			onToken(decode(new Uint8Array(tokens)));
		}
	}

	return decode(new Uint8Array(tokens));
}

function sampleFromProbs(probs: number[]): number {
	const r = Math.random();
	let cumulative = 0;
	for (let i = 0; i < probs.length; i++) {
		cumulative += probs[i];
		if (r < cumulative) return i;
	}
	return probs.length - 1;
}
