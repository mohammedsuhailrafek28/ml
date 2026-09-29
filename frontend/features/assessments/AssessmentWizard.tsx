'use client';

import {zodResolver} from '@hookform/resolvers/zod';
import {useEffect, useMemo, useRef, useState} from 'react';
import {useForm} from 'react-hook-form';
import {z} from 'zod';
import PredictionResult, {ReleaseBadge} from '../../components/PredictionResult';
import {predictDisease} from '../../lib/api';
import type {PredictionResponse} from '../../lib/api-types';
import {assessmentCopy} from './assessment-copy';
import {diseaseContracts} from './generated/contracts';

type Disease = keyof typeof diseaseContracts;
type Field = (typeof diseaseContracts)[Disease]['features'][number];
type Values = Record<string, string | number | null>;

const prettyName: Record<Disease, string> = {
  liver: 'Liver', diabetes: 'Diabetes', heart: 'Heart', kidney: 'Kidney', parkinsons: "Parkinson's",
};

function schemaFor(fields: readonly Field[]) {
  const shape: Record<string, z.ZodTypeAny> = {};
  for (const field of fields) {
    if (field.control === 'select' && field.options) {
      const values = field.options.map((option) => option.value) as [string, ...string[]];
      const category = z.string({required_error: `${field.label} is required.`}).refine((input) => values.includes(input), `Choose ${field.label.toLowerCase()}.`);
      shape[field.name] = field.required
        ? z.preprocess((input) => input === '' || input === undefined ? undefined : input, category)
        : z.preprocess((input) => input === '' || input === undefined ? null : input, category.nullable().optional());
      continue;
    }
    let numeric = z.number({required_error: `${field.label} is required.`}).finite('Enter a finite number.');
    if (field.kind === 'integer') numeric = numeric.int(`${field.label} must be a whole number.`);
    if (field.min !== null) numeric = numeric.min(field.min, `Enter at least ${field.min}${field.unit ? ` ${field.unit}` : ''}.`);
    if (field.max !== null) numeric = numeric.max(field.max, `Enter no more than ${field.max}${field.unit ? ` ${field.unit}` : ''}.`);
    shape[field.name] = field.required
      ? z.preprocess((input) => input === '' || input === undefined ? undefined : input, numeric)
      : z.preprocess((input) => input === '' || input === undefined ? null : input, numeric.nullable().optional());
  }
  return z.object(shape).strict();
}

function errorText(error: unknown) {
  if (!(error instanceof Error)) return 'The assessment could not be completed. Check your connection and try again.';
  try {
    const body = JSON.parse(error.message);
    return String(body?.error?.message ?? body?.detail ?? body?.message ?? 'The assessment could not be completed. Check your entries and try again.');
  } catch {
    return 'The assessment could not be completed. Check your connection and try again.';
  }
}

export default function AssessmentWizard({disease}: {disease: Disease}) {
  const definition = diseaseContracts[disease];
  const copy = assessmentCopy[disease];
  const fields = definition.features as readonly Field[];
  const groups = useMemo(() => {
    const groupNames = [...new Set(fields.map((field) => field.group))];
    return groupNames.map((name) => ({name, fields: fields.filter((field) => field.group === name)}));
  }, [fields]);
  const schema = useMemo(() => schemaFor(fields), [fields]);
  const blankValues = Object.fromEntries(fields.map((field) => [field.name, field.control === 'select' || field.required ? '' : null]));
  const {register, trigger, getValues, setValue, reset, formState: {errors}} = useForm<Values>({
    resolver: zodResolver(schema) as never,
    mode: 'onTouched',
    defaultValues: blankValues,
  });
  const [stage, setStage] = useState<'intro'|'form'|'review'|'processing'|'results'>('intro');
  const [step, setStep] = useState(0);
  const [acknowledged, setAcknowledged] = useState(false);
  const [unavailable, setUnavailable] = useState<Record<string, boolean>>({});
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState('');
  const heading = useRef<HTMLHeadingElement>(null);
  const experimental = disease === 'parkinsons';

  useEffect(() => {
    if (stage === 'form' || stage === 'review' || stage === 'results') heading.current?.focus();
  }, [stage, step]);

  function payload(): Record<string, unknown> {
    const values = getValues();
    return Object.fromEntries(fields.map((field) => {
      const value = values[field.name];
      return [field.name, value === '' || value === undefined ? null : value];
    }));
  }

  async function next() {
    setError('');
    const valid = await trigger(groups[step].fields.map((field) => field.name));
    if (valid) setStep((current) => current + 1);
    else document.querySelector<HTMLElement>('[aria-invalid="true"]')?.focus();
  }

  async function review() {
    setError('');
    const valid = await trigger(fields.map((field) => field.name));
    if (valid) setStage('review');
    else document.querySelector<HTMLElement>('[aria-invalid="true"]')?.focus();
  }

  async function submit() {
    if (!acknowledged) return;
    setStage('processing');
    setError('');
    try {
      setResult(await predictDisease(disease, payload()));
      setStage('results');
    } catch (problem) {
      setError(errorText(problem));
      setStage('review');
    }
  }

  if (stage === 'intro') return <main className="page assessment-page">
    <ReleaseBadge status={experimental ? 'experimental' : 'educational/research release'}/>
    <h1>{copy.title}</h1>
    <div className="assessment-intro">
      <p>{copy.description}</p>
      {'note' in copy && <p>{copy.note}</p>}
      {'evidence' in copy && <p className="evidence-note"><strong>Experimental evidence.</strong> {copy.evidence.replace('Experimental evidence: ', '')}</p>}
      {experimental && <p><strong>Pre-computed biomarkers only.</strong> The values below come from voice-analysis software such as Praat; the app does not record or analyze voice recordings.</p>}
      {disease === 'kidney' && <p className="evidence-note">{assessmentCopy.kidney.note}</p>}
      <p className="disclaimer"><strong>Educational use only.</strong> The uncalibrated model score is not a disease probability, diagnosis, screening result, triage decision, or treatment recommendation.</p>
      <button className="button" type="button" onClick={() => setStage('form')}>Begin entering {experimental ? 'biomarkers' : 'measurements'}</button>
    </div>
  </main>;

  if (stage === 'processing') return <main className="page assessment-page" aria-busy="true" aria-live="polite">
    <ReleaseBadge status={experimental ? 'experimental' : 'educational/research release'}/>
    <h1 tabIndex={-1} ref={heading}>Preparing your result</h1>
    <div className="assessment-panel"><p>Validating your entries and running the persisted {prettyName[disease]} model…</p><div className="loading-track" aria-hidden="true"><span/></div></div>
  </main>;

  if (stage === 'results' && result) return <main className="page assessment-page">
    <h1 tabIndex={-1} ref={heading}>{prettyName[disease]} model threshold result</h1>
    <PredictionResult result={result} disease={disease} measurements={payload()}/>
    <button className="button button-secondary" type="button" onClick={() => {reset(blankValues); setResult(null); setAcknowledged(false); setUnavailable({}); setStep(0); setError(''); setStage('intro');}}>Start a new assessment</button>
  </main>;

  if (stage === 'review') return <main className="page assessment-page">
    <ReleaseBadge status={experimental ? 'experimental' : 'educational/research release'}/>
    <h1 tabIndex={-1} ref={heading}>Review {prettyName[disease]} {experimental ? 'voice biomarkers' : 'measurements'}</h1>
    <div className="assessment-panel">
      <p>Check each value before sending it to the model. Entries stay in this page and are not saved in browser storage.</p>
      <dl className="review-list">{fields.map((field) => {
        const value = getValues(field.name);
        const display = value === null || value === '' || value === undefined ? (field.nullable ? 'Not available — the model pipeline will impute this value' : 'Not entered') : `${value}${field.unit ? ` ${field.unit}` : ''}`;
        return <div key={field.name}><dt>{field.label}</dt><dd>{display}</dd></div>;
      })}</dl>
      <label className="acknowledgement"><input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)}/>
        {experimental
          ? 'I understand this is an experimental educational demonstration with weak validation evidence. It accepts pre-computed biomarkers, not raw audio; the 7-subject holdout had ROC-AUC 0.586 and specificity 0.0; and its output is not a diagnosis or screening result.'
          : 'I understand this is an educational machine-learning output, not medical advice, a diagnosis, or a screening result.'}
      </label>
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="wizard-actions">
        <button className="button button-secondary" type="button" onClick={() => {setError(''); setStage('form'); setStep(groups.length - 1);}}>Edit entries</button>
        <button className="button" type="button" disabled={!acknowledged} onClick={submit}>Run prediction</button>
      </div>
    </div>
  </main>;

  const current = groups[step];
  return <main className="page assessment-page">
    <ReleaseBadge status={experimental ? 'experimental' : 'educational/research release'}/>
    <h1 tabIndex={-1} ref={heading}>{current.name}</h1>
    <div className="wizard-progress">
      <p>Step {step + 1} of {groups.length}</p>
      <progress value={step + 1} max={groups.length} aria-label={`Step ${step + 1} of ${groups.length}`}/>
    </div>
    <form className="assessment-panel" noValidate onSubmit={(event) => {event.preventDefault(); void (step < groups.length - 1 ? next() : review());}}>
      {experimental && <p className="field-context">Pre-computed voice biomarkers from voice-analysis software. This module does not record or process raw audio.</p>}
      <div className="field-grid">
        {current.fields.map((field) => {
          const errorMessage = errors[field.name]?.message;
          const helpId = `${disease}-${field.display_order}-help`;
          const errorId = `${disease}-${field.display_order}-error`;
          const describedBy = [helpId, errorMessage ? errorId : null].filter(Boolean).join(' ');
          return <div className="form-field" key={field.name}>
            <label htmlFor={`${disease}-${field.display_order}`}>{field.label}{field.required && <span className="required-marker"> (required)</span>}{field.unit && <span className="field-unit"> · {field.unit}</span>}</label>
            <p id={helpId} className="field-help">{field.help}{field.min !== null && field.max !== null ? ` Allowed range: ${field.min} to ${field.max}${field.unit ? ` ${field.unit}` : ''}. These are input checks, not clinical cut-offs.` : ''}</p>
            {field.control === 'select' && field.options
              ? <select id={`${disease}-${field.display_order}`} aria-describedby={describedBy} aria-invalid={Boolean(errorMessage)} {...register(field.name)}>
                  <option value="">Choose an option</option>{field.options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                </select>
              : <div className="number-entry">
                  <input id={`${disease}-${field.display_order}`} type="number" inputMode={field.kind === 'integer' ? 'numeric' : 'decimal'} step={field.step} min={field.min ?? undefined} max={field.max ?? undefined} aria-describedby={describedBy} aria-invalid={Boolean(errorMessage)} {...register(field.name, {setValueAs: (value) => value === '' ? '' : Number(value)})}/>
                  {!field.required && <button type="button" className="text-control" onClick={() => {const markUnavailable = !unavailable[field.name]; setUnavailable((current) => ({...current, [field.name]: markUnavailable})); setValue(field.name, markUnavailable ? null : '', {shouldValidate: true});}}>{unavailable[field.name] ? 'Enter a value' : 'Mark not available'}</button>}
                </div>}
            {errorMessage && <p className="form-error" id={errorId}>{String(errorMessage)}</p>}
          </div>;
        })}
      </div>
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="wizard-actions">
        <button className="button button-secondary" type="button" disabled={step === 0} onClick={() => {setError(''); setStep((value) => Math.max(0, value - 1));}}>Back</button>
        <button className="button" type="submit">{step < groups.length - 1 ? 'Continue' : 'Review entries'}</button>
      </div>
    </form>
  </main>;
}
