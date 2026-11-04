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
- readmission: I recommend the random forest at a 0.10 threshold for readmission because msised readmissions could mean patients dont recive extra discharge planning or follow-up. On the Test set, it caught 54% of readmissions and flags 1428 of 4,000 patients. I would accept the added follow-up workload, while recognizing that it still misses about 46% of readmissions and that only 13.6% of flagged patients are readmitted.
- diabetes: I recommend logistic regression at a 0.10 threshold for diabetes screening because finding people who may need follow-up testing matters most. On the Test set, it catches 91% of positives and flags 589 of 1,052 adults. I accept the extra follow-up workload, while recognizing that its precision is 30.7%, so many people flagged will not meet the positive-class definition.

## Design choice and AI use
A few sentences each. The design choice must be your own idea, not an AI tool's.
- design choice: I chose to show a confusion matrix because it separates patients the model flags correctly from missed positives and false alarms. That helps a clinical lead see the workload and missed cases at a chosen threshold.
- AI suggestion I turned down: An AI tool suggested using a 0.50 threshold for both jobs. I turned that down because the jobs have different priorities, and I wanted to compare thresholds that better show the tradeoff between missed positives and follow-up workload.