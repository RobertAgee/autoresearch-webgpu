import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import {
	importDatasetFromHuggingFace,
	inspectHuggingFaceDataset,
	listDatasetVersions,
	setActiveDatasetVersion,
	type HuggingFaceDatasetInspection
} from '$lib/server/datasets/service';
import { proposeDatasetRecipeWithBackend } from '$lib/server/datasets/proposal';
import type { ResearchEndpointProfile } from '$lib/research/providers';

type DatasetAction =
	| {
		action: 'importFromHuggingFace';
		datasetId: string;
		label?: string;
		recipeKey?: string | null;
		textFields?: string[] | null;
		samplePrompt?: string | null;
		maxTrainExamples?: number | null;
		maxValidationExamples?: number | null;
	}
	| {
		action: 'proposeRecipe';
		inspection: HuggingFaceDatasetInspection;
		profile?: ResearchEndpointProfile | null;
	}
	| {
		action: 'setActive';
		versionId: number;
	};

function parseNumberOrNull(value: unknown): number | null {
	if (value == null || value === '') return null;
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : null;
}

export const GET: RequestHandler = async ({ url }) => {
	if (url.searchParams.get('inspect') === 'huggingface') {
		return json(await inspectHuggingFaceDataset(url.searchParams.get('datasetId') ?? ''));
	}
	return json(listDatasetVersions());
};

export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = await request.json() as DatasetAction;
		if (body.action === 'setActive') {
			return json(setActiveDatasetVersion(Number(body.versionId)));
		}

		if (body.action === 'importFromHuggingFace') {
			return json(await importDatasetFromHuggingFace({
				datasetId: body.datasetId,
				label: body.label,
				recipeKey: body.recipeKey,
				textFields: body.textFields,
				samplePrompt: body.samplePrompt,
				maxTrainExamples: parseNumberOrNull(body.maxTrainExamples),
				maxValidationExamples: parseNumberOrNull(body.maxValidationExamples)
			}));
		}

		if (body.action === 'proposeRecipe') {
			return json({
				proposal: await proposeDatasetRecipeWithBackend(body.inspection, body.profile)
			});
		}

		return json({ error: 'Unsupported action' }, { status: 400 });
	} catch (error) {
		return json(
			{ error: error instanceof Error ? error.message : String(error) },
			{ status: 500 }
		);
	}
};
