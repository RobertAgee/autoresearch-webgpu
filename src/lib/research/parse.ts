import type { ResearchProposal } from './config';

/**
 * Shared research response parsing logic.
 */
export function parseClaudeResponse(text: string): ResearchProposal | null {
	try {
		const parsed = JSON.parse(text);
		if (parsed && typeof parsed.reasoning === 'string') return parsed;
	} catch {}

	const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
	if (fenceMatch) {
		try {
			const parsed = JSON.parse(fenceMatch[1].trim());
			if (parsed && typeof parsed.reasoning === 'string') return parsed;
		} catch {}
	}

	try {
		const start = text.indexOf('{');
		if (start >= 0) {
			let depth = 0, end = start;
			for (let i = start; i < text.length; i++) {
				if (text[i] === '{') depth++;
				else if (text[i] === '}') { depth--; if (depth === 0) { end = i; break; } }
			}
			const parsed = JSON.parse(text.slice(start, end + 1));
			if (parsed && typeof parsed.reasoning === 'string') return parsed;
		}
	} catch {}

	return null;
}
