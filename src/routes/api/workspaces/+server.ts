import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import {
	getExperimentWorkspaceRowById,
	insertExperimentWorkspaceRow,
	listExperimentWorkspaceRows
} from '$lib/server/experiment-workspaces-service';

type ActionBody =
	| {
		action: 'createWorkspace';
		workspace: {
			name: string;
			sourceType: 'dataset' | 'results';
			datasetVersionId?: number | null;
			baseExperimentId?: number | null;
			readme?: string;
			notes?: string;
		};
	}
	| {
		action: 'getWorkspace';
		id: number;
	};

export const GET: RequestHandler = async () => {
	return json({
		workspaces: listExperimentWorkspaceRows()
	});
};

export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = await request.json() as ActionBody;
		switch (body.action) {
			case 'createWorkspace':
				return json({ id: insertExperimentWorkspaceRow(body.workspace) });
			case 'getWorkspace':
				return json({ workspace: getExperimentWorkspaceRowById(body.id) });
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
