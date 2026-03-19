import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { ensureExperimentStorage, importExperimentsZip } from '$lib/server/experiments-service';

export const POST: RequestHandler = async ({ request }) => {
	try {
		ensureExperimentStorage();
		const formData = await request.formData();
		const file = formData.get('file');
		if (!(file instanceof Blob)) {
			return json({ error: 'Missing file upload' }, { status: 400 });
		}
		return json(await importExperimentsZip(file));
	} catch (error) {
		return json(
			{ error: error instanceof Error ? error.message : String(error) },
			{ status: 500 }
		);
	}
};
