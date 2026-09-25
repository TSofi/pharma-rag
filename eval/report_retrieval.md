# Evaluation report

- Questions: **16**  ·  top_k = 6  ·  min_score = 0.62  ·  LLM = gemini
- Retrieval hit@6: **14/14 (100%)**

| # | Type | Question | Retrieval | Top-3 retrieved (drug · section · score) |
|---|---|---|---|---|
| 1 | indication | What is metformin used for? | ✅ | metformin · Indications and Usage · 0.811<br>metformin · Mechanism of Action · 0.801<br>metformin · Drug Interactions · 0.758 |
| 2 | contraindication | What are the contraindications of lisinopril? | ✅ | lisinopril · Contraindications · 0.842<br>lisinopril · Drug Interactions · 0.783<br>lisinopril · Adverse Reactions · 0.782 |
| 3 | interaction | Does warfarin interact with NSAIDs? | ✅ | warfarin · Drug Interactions · 0.797<br>warfarin · Drug Interactions · 0.774<br>warfarin · Drug Interactions · 0.77 |
| 4 | dosage | What is the usual adult dose of amoxicillin? | ✅ | amoxicillin · Dosage and Administration · 0.886<br>amoxicillin · Dosage and Administration · 0.866<br>amoxicillin · Dosage and Administration · 0.861 |
| 5 | special population | Can atorvastatin be used during pregnancy? | ✅ | atorvastatin · Use in Specific Populations · 0.864<br>atorvastatin · Pregnancy · 0.861<br>atorvastatin · Pregnancy · 0.852 |
| 6 | safety | What is the boxed warning for sertraline? | ✅ | sertraline · Boxed Warning · 0.844<br>sertraline · Pregnancy · 0.781<br>sertraline · Warnings and Precautions · 0.779 |
| 7 | adverse reactions | What are the most common side effects of amlodipine? | ✅ | amlodipine · Adverse Reactions · 0.822<br>amlodipine · Adverse Reactions · 0.816<br>amlodipine · Adverse Reactions · 0.797 |
| 8 | mechanism | How does omeprazole work? | ✅ | omeprazole · Mechanism of Action · 0.793<br>omeprazole · Indications and Usage · 0.761<br>omeprazole · Indications and Usage · 0.758 |
| 9 | interaction | Should ciprofloxacin be taken together with antacids? | ✅ | ciprofloxacin · Drug Interactions · 0.773<br>ciprofloxacin · Drug Interactions · 0.749<br>ciprofloxacin · Drug Interactions · 0.74 |
| 10 | dosage (OTC label) | What is the maximum daily dose of acetaminophen for adults? | ✅ | acetaminophen · Warnings · 0.858<br>acetaminophen · Dosage and Administration · 0.813<br>acetaminophen · Overdosage · 0.757 |
| 11 | brand name lookup | What is Eliquis used for? | ✅ | apixaban · Indications and Usage · 0.764<br>apixaban · Warnings and Precautions · 0.751<br>apixaban · Warnings and Precautions · 0.742 |
| 12 | overdose | What are the signs of levothyroxine overdose? | ✅ | levothyroxine · Overdosage · 0.848<br>levothyroxine · Adverse Reactions · 0.816<br>levothyroxine · Adverse Reactions · 0.798 |
| 13 | brand + lay wording | Can I drink alcohol while taking Zoloft? | ✅ | sertraline · Overdosage · 0.664<br>sertraline · Drug Interactions · 0.652<br>sertraline · Dosage and Administration · 0.647 |
| 14 | out of library (must refuse) | What is Ozempic used for? | — | omeprazole · Dosage and Administration · 0.585<br>apixaban · Overdosage · 0.581<br>acetaminophen · Purpose · 0.574 |
| 15 | edge: comparison the labels don't make | Which is better for high blood pressure, losartan or lisinopril? | ✅ | losartan · Drug Interactions · 0.788<br>lisinopril · Drug Interactions · 0.774<br>lisinopril · Drug Interactions · 0.768 |
| 16 | off-topic (must refuse) | What is the best pizza topping? | — | pantoprazole · Adverse Reactions · 0.492<br>pantoprazole · Use in Specific Populations · 0.484<br>omeprazole · Dosage and Administration · 0.468 |
