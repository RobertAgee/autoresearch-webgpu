import { getCatalogDb } from './catalog-db';

export type ExperimentWorkspaceRow = {
	id: number;
	name: string;
	source_type: 'dataset' | 'results';
	dataset_version_id: number | null;
	base_experiment_id: number | null;
	readme: string;
	notes: string;
	created_at: string;
};

export function listExperimentWorkspaceRows(): ExperimentWorkspaceRow[] {
	return getCatalogDb().prepare(`
		SELECT *
		FROM experiment_workspaces
		ORDER BY created_at DESC, id DESC
	`).all() as ExperimentWorkspaceRow[];
}

export function getExperimentWorkspaceRowById(id: number): ExperimentWorkspaceRow | null {
	return getCatalogDb().prepare(`
		SELECT *
		FROM experiment_workspaces
		WHERE id = ?
	`).get(id) as ExperimentWorkspaceRow | undefined ?? null;
}

export function insertExperimentWorkspaceRow(workspace: {
	name: string;
	sourceType: 'dataset' | 'results';
	datasetVersionId?: number | null;
	baseExperimentId?: number | null;
	readme?: string;
	notes?: string;
	createdAt?: string;
}): number {
	const result = getCatalogDb().prepare(`
		INSERT INTO experiment_workspaces (
			name,
			source_type,
			dataset_version_id,
			base_experiment_id,
			readme,
			notes,
			created_at
		) VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
	`).run(
		workspace.name.trim(),
		workspace.sourceType,
		workspace.datasetVersionId ?? null,
		workspace.baseExperimentId ?? null,
		workspace.readme?.trim() ?? '',
		workspace.notes?.trim() ?? '',
		workspace.createdAt ?? null
	);
	return Number(result.lastInsertRowid);
}

export function updateExperimentWorkspaceBaseExperimentRow(id: number, baseExperimentId: number | null): void {
	getCatalogDb().prepare(`
		UPDATE experiment_workspaces
		SET base_experiment_id = ?
		WHERE id = ?
	`).run(baseExperimentId, id);
}
