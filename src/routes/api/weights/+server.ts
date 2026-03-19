import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { clearWeightsArtifacts } from '$lib/server/artifacts-service';

export const DELETE: RequestHandler = async () => {
	try {
		await clearWeightsArtifacts();
		return json({ ok: true });
	} catch (error) {
		return json(
			{ error: error instanceof Error ? error.message : String(error) },
			{ status: 500 }
		);
	}
};
