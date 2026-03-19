import { numpy as np } from '@jax-js/jax';
import type { Params } from './prepare';

type ParamMeta = {
	key: string;
	shape: number[];
	dtype: string;
	offset: number;
	byteLength: number;
};

function typedArrayForMeta(meta: ParamMeta, slice: Uint8Array): Float32Array | Int32Array | Uint32Array {
	switch (meta.dtype) {
		case 'float32':
			return new Float32Array(slice.buffer, slice.byteOffset, slice.byteLength / 4);
		case 'int32':
			return new Int32Array(slice.buffer, slice.byteOffset, slice.byteLength / 4);
		case 'uint32':
			return new Uint32Array(slice.buffer, slice.byteOffset, slice.byteLength / 4);
		default:
			return new Float32Array(slice.buffer, slice.byteOffset, slice.byteLength / 4);
	}
}

async function packParams(params: Params): Promise<{ metas: ParamMeta[]; buffer: Uint8Array }> {
	const metas: ParamMeta[] = [];
	const buffers: Uint8Array[] = [];
	let totalBytes = 0;

	for (const key of Object.keys(params)) {
		const arr = params[key];
		const data = await arr.ref.data();
		const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
		metas.push({
			key,
			shape: arr.ref.shape,
			dtype: arr.ref.dtype,
			offset: totalBytes,
			byteLength: data.byteLength
		});
		buffers.push(bytes);
		totalBytes += data.byteLength;
	}

	const buffer = new Uint8Array(totalBytes);
	for (let i = 0; i < buffers.length; i++) {
		buffer.set(buffers[i], metas[i].offset);
	}

	return { metas, buffer };
}

export async function saveWeights(experimentId: number, params: Params): Promise<string> {
	const { metas, buffer } = await packParams(params);
	const formData = new FormData();
	formData.set('meta', JSON.stringify(metas));
	formData.set('file', new Blob([buffer], { type: 'application/octet-stream' }), `exp-${experimentId}.bin`);

	const response = await fetch(`/api/weights/${experimentId}`, {
		method: 'POST',
		body: formData
	});
	if (!response.ok) {
		throw new Error(await response.text());
	}
	const data = await response.json() as { path: string };
	return data.path;
}

export async function loadWeights(experimentId: number): Promise<Params | null> {
	const response = await fetch(`/api/weights/${experimentId}`);
	if (response.status === 404) return null;
	if (!response.ok) {
		throw new Error(await response.text());
	}

	const metaHeader = response.headers.get('X-Weights-Meta');
	if (!metaHeader) {
		throw new Error('Missing weights metadata');
	}
	const metas = JSON.parse(metaHeader) as ParamMeta[];
	const buffer = new Uint8Array(await response.arrayBuffer());
	const params: Params = {};

	for (const meta of metas) {
		const slice = buffer.slice(meta.offset, meta.offset + meta.byteLength);
		params[meta.key] = np.array(
			typedArrayForMeta(meta, slice) as never,
			{ shape: meta.shape, dtype: meta.dtype as never }
		);
	}

	return params;
}

export async function deleteSavedWeights(experimentId: number): Promise<void> {
	await fetch(`/api/weights/${experimentId}`, {
		method: 'DELETE'
	});
}

export async function clearSavedWeights(): Promise<void> {
	await fetch('/api/weights', {
		method: 'DELETE'
	});
}
