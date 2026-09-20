export const documents = [
  {
    id: "cardio-followup",
    title: "Cardiology follow-up",
    type: "Progress note",
    patient: "Synthetic Patient A",
    date: "2026-02-14",
    safety: "Synthetic",
  },
  {
    id: "copd-discharge",
    title: "COPD discharge summary",
    type: "Discharge summary",
    patient: "Synthetic Patient B",
    date: "2026-03-02",
    safety: "Synthetic",
  },
  {
    id: "diabetes-review",
    title: "Diabetes medication review",
    type: "Endocrinology note",
    patient: "Synthetic Patient C",
    date: "2026-03-18",
    safety: "Synthetic",
  },
  {
    id: "chest-ct",
    title: "Chest CT report",
    type: "Radiology report",
    patient: "Synthetic Patient D",
    date: "2026-04-05",
    safety: "Synthetic",
  },
  {
    id: "anticoagulation",
    title: "Anticoagulation counseling",
    type: "Medication note",
    patient: "Synthetic Patient E",
    date: "2026-04-21",
    safety: "Synthetic",
  },
];

export const chunks = [
  {
    id: "cardio-01",
    docId: "cardio-followup",
    page: 1,
    heading: "Assessment",
    text: "Blood pressure remains above goal at 162/94 mmHg despite adherence to lisinopril 10 mg daily. The synthetic patient reports no chest pain, dizziness, or shortness of breath.",
  },
  {
    id: "cardio-02",
    docId: "cardio-followup",
    page: 2,
    heading: "Plan",
    text: "Increase lisinopril from 10 mg to 20 mg daily. Record home blood pressure twice each day and repeat a basic metabolic panel in two weeks to monitor potassium and kidney function.",
  },
  {
    id: "copd-01",
    docId: "copd-discharge",
    page: 1,
    heading: "Hospital course",
    text: "The synthetic patient was treated for an acute COPD exacerbation with dyspnea and wheezing. Oxygen saturation improved from 89% to 95% after bronchodilator therapy.",
  },
  {
    id: "copd-02",
    docId: "copd-discharge",
    page: 2,
    heading: "Discharge medications",
    text: "Use albuterol as needed, continue tiotropium daily, and complete prednisone 40 mg daily for five days. Return for worsening breathlessness, blue lips, confusion, or persistent low oxygen readings.",
  },
  {
    id: "diabetes-01",
    docId: "diabetes-review",
    page: 1,
    heading: "Results",
    text: "Hemoglobin A1c improved from 8.4% to 7.1% over three months. Fasting glucose readings are usually between 110 and 135 mg/dL without reported hypoglycemia.",
  },
  {
    id: "diabetes-02",
    docId: "diabetes-review",
    page: 1,
    heading: "Plan",
    text: "Continue metformin 1000 mg twice daily, nutrition counseling, and regular activity. Repeat A1c in three months and call if glucose is repeatedly below 70 mg/dL.",
  },
  {
    id: "ct-01",
    docId: "chest-ct",
    page: 1,
    heading: "Findings",
    text: "Noncontrast chest CT shows a solitary 6 mm solid pulmonary nodule in the right upper lobe. No pleural effusion, focal consolidation, or enlarged mediastinal lymph nodes are seen.",
  },
  {
    id: "ct-02",
    docId: "chest-ct",
    page: 2,
    heading: "Impression",
    text: "For this synthetic low-risk scenario, obtain a follow-up chest CT in 12 months to document stability of the 6 mm pulmonary nodule. Earlier imaging may be appropriate if risk factors change.",
  },
  {
    id: "anticoag-01",
    docId: "anticoagulation",
    page: 1,
    heading: "Medication review",
    text: "Continue apixaban 5 mg twice daily for stroke prevention in nonvalvular atrial fibrillation. The synthetic patient reports no missed doses and no current bleeding.",
  },
  {
    id: "anticoag-02",
    docId: "anticoagulation",
    page: 1,
    heading: "Safety counseling",
    text: "Seek urgent evaluation for uncontrolled bleeding, black stools, vomiting blood, a severe headache, or a significant fall while taking apixaban. Avoid starting NSAIDs without clinician review.",
  },
];

export const evaluationQueries = [
  {
    id: "hypertension",
    label: "Hypertension plan",
    question: "How was the high blood pressure treated and monitored?",
    relevant: ["cardio-02", "cardio-01"],
  },
  {
    id: "copd",
    label: "COPD discharge",
    question: "What treatment was prescribed for the COPD flare?",
    relevant: ["copd-02", "copd-01"],
  },
  {
    id: "diabetes",
    label: "Diabetes trend",
    question: "What was the A1c result and diabetes medication plan?",
    relevant: ["diabetes-01", "diabetes-02"],
  },
  {
    id: "nodule",
    label: "Nodule follow-up",
    question: "What follow-up was recommended for the pulmonary nodule?",
    relevant: ["ct-02", "ct-01"],
  },
  {
    id: "bleeding",
    label: "Anticoagulation safety",
    question: "What bleeding warning was given for apixaban?",
    relevant: ["anticoag-02", "anticoag-01"],
  },
];

