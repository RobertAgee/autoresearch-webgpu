const decoder = new TextDecoder();

export function decode(ids: ArrayLike<number>): string {
	return decoder.decode(new Uint8Array(ids));
}
