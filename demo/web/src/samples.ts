export type SampleNote = {
  id: string;
  title: string;
  text: string;
};

/** Synthetic notes only. Do not add real patient text. */
export const SAMPLES: SampleNote[] = [
  {
    id: "clinic",
    title: "Clinic note",
    text: "Casey Example, MRN 00000000, was seen on 02/02/2020 for asthma. Phone (555) 010-0199. Started lisinopril 10 mg daily.",
  },
  {
    id: "discharge",
    title: "Discharge snippet",
    text: "Patient Jordan Sample (DOB 01/01/1980) lives at 100 Example Avenue. Discharged with metformin for type 2 diabetes. Email jordan.sample@example.com.",
  },
];
