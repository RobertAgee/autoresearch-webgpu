import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { ensureExperimentStorage, getBestExperimentRow } from '$lib/server/experiments-service';

function parseOptionalNumber(value: string | null): number | null {
	if (value == null || value === '') return null;
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : null;
}

export const GET: RequestHandler = async ({ url }) => {
	ensureExperimentStorage();
	const datasetVersionId = parseOptionalNumber(url.searchParams.get('datasetVersionId'));
	return json({
		best: getBestExperimentRow(datasetVersionId)
	});
};
