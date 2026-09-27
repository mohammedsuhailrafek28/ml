import {notFound} from 'next/navigation';
import AssessmentWizard from '../../../features/assessments/AssessmentWizard';
import {diseaseContracts} from '../../../features/assessments/generated/contracts';

export function generateStaticParams() {
  return Object.keys(diseaseContracts).map((disease) => ({disease}));
}

export default async function DiseaseAssessment({params}: {params: Promise<{disease: string}>}) {
  const {disease} = await params;
  if (!Object.hasOwn(diseaseContracts, disease)) notFound();
  return <AssessmentWizard disease={disease as keyof typeof diseaseContracts}/>;
}
