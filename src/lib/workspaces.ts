export type ExperimentWorkspaceRecord = {
	id: number;
	name: string;
	source_type: 'dataset' | 'results';
	dataset_version_id: number | null;
	base_experiment_id: number | null;
	readme: string;
	notes: string;
	created_at: string;
};

async function readJson<T>(response: Response): Promise<T> {
	if (!response.ok) {
		throw new Error(await response.text());
	}
	return response.json() as Promise<T>;
}

export async function listExperimentWorkspaces(): Promise<ExperimentWorkspaceRecord[]> {
	const data = await readJson<{ workspaces: ExperimentWorkspaceRecord[] }>(await fetch('/api/workspaces'));
	return data.workspaces;
}

export async function createExperimentWorkspace(workspace: {
	name: string;
	sourceType: 'dataset' | 'results';
	datasetVersionId?: number | null;
	baseExperimentId?: number | null;
	readme?: string;
	notes?: string;
}): Promise<number> {
	const data = await readJson<{ id: number }>(await fetch('/api/workspaces', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			action: 'createWorkspace',
			workspace
		})
	}));
	return data.id;
}
