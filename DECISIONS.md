# Decisions

<!-- Fill in each section at the step the activity website gives, then commit and push.
     Write each answer after its label. Keep the headings and labels exactly as they are:
     `python -m analysis` reads them. Limit: 150 words for Before results.
     Fill in Recommendations and Design choice and AI use last, once your app is built. -->

## Before results
For each dataset: positive class; which error matters more for the job (missed positive or false alarm) and why.
- readmission: The positive class is readmission within 30 days. A missed positive matters more because the patient may not be flagged for added discharge planning or follow-up; a false alarm may use resources for someone who would not be readmitted.
- diabetes: The positive class is doctor diagnosis or HbA1c at least 6.5%. A missed positive matters more because the person may not be flagged for follow-up testing; a false alarm may lead to an unnecessary follow-up test.

## Recommendations
- readmission: At 0.5, the models flag almost nobody. Random roest has the highest Test AUROC (0.651). At 0.10, its sensitivity is 54%, specificity 66%, precision 13.6%, and its flags 1,428 of 4,000 patients. You could choose "none is adequate" because it still misses about 46% of readmissions orchoose random forst at 0.10 while explicitly accepting the high false alarm workload.
- diabetes: If catching positive matters most, logistic regression at 0.10 is 1 option. Test AUROC 0.827, sensitivity 91%, specificity 52%, precision 30.7%, and 589 of 1052 patients flagged. At 0.2, its dlags fewer (390) and his higher precision (39.7%), but sensitivity falls to 78%. Therefore you should pick the balance you think is appropriate

## Design choice and AI use
A few sentences each. The design choice must be your own idea, not an AI tool's.
- design choice: One thing I wanted to show was the the four groups in a confusion matrixx.
- AI suggestion I turned down: One suggestion I turned down was to use 0.50 for both jobs.