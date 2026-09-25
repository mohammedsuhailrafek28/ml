export type ThresholdResult = 'at_or_above' | 'below';

export type PredictionResponse = {
  disease: string;
  prediction: 0 | 1;
  model_score: number;
  decision_threshold: number;
  threshold_result: ThresholdResult;
  score_type: 'uncalibrated_model_score';
  model_identifier: string;
  release_status: string;
  intended_use: string;
  limitations: string[];
  disclaimer: string;
};
