import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import {
	clearExperimentData,
	ensureExperimentStorage,
	deleteExperimentRows,
	insertExperimentRow,
	insertInferenceRow,
	insertLossCurveRows,
	listExperimentRecords,
	updateExperimentWeightsPath
} from '$lib/server/experiments-service';

type ActionBody =
	| {
		action: 'insertExperiment';
		experiment: Parameters<typeof insertExperimentRow>[0];
	}
	| {
		action: 'insertLossCurve';
		experimentId: number;
		curve: { step: number; loss: number }[];
	}
	| {
		action: 'insertInference';
		inference: Parameters<typeof insertInferenceRow>[0];
	}
	| {
		action: 'updateWeightsPath';
		id: number;
		weightsPath: string;
	}
	| {
		action: 'deleteExperiments';
		ids: number[];
	}
	| {
		action: 'clearAllData';
	};

function parseOptionalNumber(value: string | null): number | null {
	if (value == null || value === '') return null;
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : null;
}

export const GET: RequestHandler = async ({ url }) => {
	ensureExperimentStorage();
	const datasetVersionId = parseOptionalNumber(url.searchParams.get('datasetVersionId'));
	const projectId = parseOptionalNumber(url.searchParams.get('projectId'));
	return json({
		experiments: listExperimentRecords(datasetVersionId, projectId)
	});
};

export const POST: RequestHandler = async ({ request }) => {
	try {
		ensureExperimentStorage();
		const body = await request.json() as ActionBody;
		switch (body.action) {
			case 'insertExperiment':
				return json({ id: insertExperimentRow(body.experiment) });
			case 'insertLossCurve':
				insertLossCurveRows(body.experimentId, body.curve);
				return json({ ok: true });
			case 'insertInference':
				return json({ id: insertInferenceRow(body.inference) });
			case 'updateWeightsPath':
				updateExperimentWeightsPath(body.id, body.weightsPath);
				return json({ ok: true });
			case 'deleteExperiments':
				return json({ deleted: await deleteExperimentRows(body.ids) });
			case 'clearAllData':
				await clearExperimentData();
				return json({ ok: true });
			default:
				return json({ error: 'Unsupported action' }, { status: 400 });
		}
	} catch (error) {
		return json(
			{ error: error instanceof Error ? error.message : String(error) },
			{ status: 500 }
		);
	}
};
