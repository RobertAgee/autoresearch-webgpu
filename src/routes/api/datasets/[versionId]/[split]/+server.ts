import { readFile } from 'node:fs/promises';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getDatasetSplitPath } from '$lib/server/datasets/service';

export const GET: RequestHandler = async ({ params }) => {
	const versionId = Number(params.versionId);
	const split = params.split === 'train' ? 'train' : params.split === 'validation' || params.split === 'val' ? 'validation' : null;
	if (!Number.isInteger(versionId) || versionId < 1 || !split) {
		return json({ error: 'Invalid dataset version or split.' }, { status: 400 });
	}

	try {
		const bytes = await readFile(getDatasetSplitPath(versionId, split));
		return new Response(bytes, {
			headers: {
				'Content-Type': 'application/octet-stream',
				'Cache-Control': 'no-store'
			}
		});
	} catch (error) {
		return json(
			{ error: error instanceof Error ? error.message : String(error) },
			{ status: 404 }
		);
	}
};
