# Calibrated, Abstaining RAG ("knows what it doesn't know")

##### Pitch: A RAG system that quantifies its own confidence at both retrieval and generation time, and explicitly abstains or asks a clarifying question instead of hallucinating when evidence is weak or ambiguous — using a custom confidence score, not a prompted "say I don't know if unsure."
 
 
 ###### Why it's genuinely rarer: Almost every GitHub RAG project optimizes for "give an answer." Very few build actual uncertainty quantification into the retrieval+generation pipeline — it requires combining signals (retrieval score gap between top-1 and top-k, cross-encoder reranker confidence, self-consistency across sampled generations) into a calibrated decision rule, then validating that the calibration is honest (does "low confidence" actually correlate with "wrong answer"?). That validation step is what most people skip and what makes this defensible in an interview.



###### Hardest part: Building and validating the calibration — proving your confidence score is actually predictive of correctness (e.g. via a reliability diagram / ECE-style metric), not just a number that looks reasonable.


#### Why it fits you: You've already done uncertainty quantification (MC dropout) in GETHER — this is the same rigor applied to a new problem, which is a much stronger interview story than a from-scratch topic.



