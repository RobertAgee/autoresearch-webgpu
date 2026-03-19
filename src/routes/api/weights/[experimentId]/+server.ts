import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import {
	deleteWeightsArtifact,
	loadWeightsArtifact,
	saveWeightsArtifact,
	type WeightMeta
} from '$lib/server/artifacts-service';

function parseExperimentId(value: string): number {
	const parsed = Number(value);
	if (!Number.isInteger(parsed) || parsed < 1) {
		throw new Error('Invalid experiment id');
	}
	return parsed;
}

export const GET: RequestHandler = async ({ params }) => {
	try {
		const experimentId = parseExperimentId(params.experimentId);
		const artifact = await loadWeightsArtifact(experimentId);
		if (!artifact) {
			return json({ error: 'Weights not found' }, { status: 404 });
		}
		return new Response(artifact.buffer, {
			headers: {
				'Content-Type': 'application/octet-stream',
				'X-Weights-Meta': JSON.stringify(artifact.metas),
				'Cache-Control': 'no-store'
			}
		});
	} catch (error) {
		return json(
			{ error: error instanceof Error ? error.message : String(error) },
			{ status: 400 }
		);
	}
};

export const POST: RequestHandler = async ({ params, request }) => {
	try {
		const experimentId = parseExperimentId(params.experimentId);
		const formData = await request.formData();
		const file = formData.get('file');
		const metaRaw = formData.get('meta');
		if (!(file instanceof Blob) || typeof metaRaw !== 'string') {
			return json({ error: 'Missing weights file or meta' }, { status: 400 });
		}
		const metas = JSON.parse(metaRaw) as WeightMeta[];
		const path = await saveWeightsArtifact(
			experimentId,
			new Uint8Array(await file.arrayBuffer()),
			metas
		);
		return json({ path });
	} catch (error) {
		return json(
			{ error: error instanceof Error ? error.message : String(error) },
			{ status: 500 }
		);
	}
};

export const DELETE: RequestHandler = async ({ params }) => {
	try {
		const experimentId = parseExperimentId(params.experimentId);
		await deleteWeightsArtifact(experimentId);
		return json({ ok: true });
	} catch (error) {
		return json(
			{ error: error instanceof Error ? error.message : String(error) },
			{ status: 400 }
		);
	}
};
