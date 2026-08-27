export type ModelMetrics={accuracy:number;precision:number;recall:number;specificity:number;f1:number;roc_auc:number};
export type TopFactor={feature:string;importance:number};
export type PredictionResponse={disease:string;prediction:number;label:string;probability:number|null;threshold:number;selectedModel:string;modelMetrics?:ModelMetrics|null;topFactors?:TopFactor[];limitations:string[];disclaimer:string};
