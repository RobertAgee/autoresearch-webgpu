/**
 * Shared Claude response parsing logic.
 */
export function parseClaudeResponse(text: string): { code: string; reasoning: string } | null {
	try {
		const parsed = JSON.parse(text);
		if (parsed.code && parsed.reasoning) return parsed;
	} catch {}

	const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
	if (fenceMatch) {
		try {
			const parsed = JSON.parse(fenceMatch[1].trim());
			if (parsed.code) return { code: parsed.code, reasoning: parsed.reasoning || '' };
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
			if (parsed.code) return { code: parsed.code, reasoning: parsed.reasoning || '' };
		}
	} catch {}

	return null;
}
