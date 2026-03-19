import type { RequestHandler } from './$types';
import {
	ensureExperimentStorage,
	exportExperimentsZip,
	exportProjectExperimentsZip
} from '$lib/server/experiments-service';

export const GET: RequestHandler = async ({ url }) => {
	ensureExperimentStorage();
	const projectIdValue = url.searchParams.get('projectId');
	const projectId = projectIdValue == null || projectIdValue === '' ? null : Number(projectIdValue);
	const bytes = projectId != null && Number.isFinite(projectId)
		? await exportProjectExperimentsZip(projectId)
		: await exportExperimentsZip();
	return new Response(bytes, {
		headers: {
			'Content-Type': 'application/zip',
			'Content-Disposition': `attachment; filename="${projectId != null ? `autoresearch-project-${projectId}` : 'autoresearch-experiments'}.zip"`,
			'Cache-Control': 'no-store'
		}
	});
};
