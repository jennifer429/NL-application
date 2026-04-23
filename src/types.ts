import { z } from "zod";

export const StudySchema = z.object({
  study_id: z.string(),
  study_description: z.string(),
  study_date: z.string(),
});

export const CaseSchema = z.object({
  case_id: z.string(),
  patient_id: z.string().optional(),
  patient_name: z.string().optional(),
  current_study: StudySchema,
  prior_studies: z.array(StudySchema),
});

export const RequestSchema = z.object({
  challenge_id: z.string().optional(),
  schema_version: z.number().optional(),
  generated_at: z.string().optional(),
  cases: z.array(CaseSchema),
});

export const PredictionSchema = z.object({
  case_id: z.string(),
  study_id: z.string(),
  predicted_is_relevant: z.boolean(),
});

export const ResponseSchema = z.object({
  predictions: z.array(PredictionSchema),
});

export type Study = z.infer<typeof StudySchema>;
export type Case = z.infer<typeof CaseSchema>;
export type Request = z.infer<typeof RequestSchema>;
export type Prediction = z.infer<typeof PredictionSchema>;
export type Response = z.infer<typeof ResponseSchema>;
