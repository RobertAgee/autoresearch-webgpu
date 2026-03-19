import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { ensureExperimentStorage, listInferencesForExperiment } from '$lib/server/experiments-service';

export const GET: RequestHandler = async ({ params }) => {
	ensureExperimentStorage();
	const experimentId = Number(params.experimentId);
	if (!Number.isInteger(experimentId) || experimentId < 1) {
		return json({ error: 'Invalid experiment id' }, { status: 400 });
	}
	return json({
		inferences: listInferencesForExperiment(experimentId)
	});
};
