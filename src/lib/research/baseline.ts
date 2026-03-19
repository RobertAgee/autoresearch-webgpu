import { BASELINE_RESEARCH_CONFIG, buildTrainCodeFromConfig } from './config';

/**
 * The baseline training code is assembled from the constrained research config.
 */
export const BASELINE_CODE = buildTrainCodeFromConfig(BASELINE_RESEARCH_CONFIG);
