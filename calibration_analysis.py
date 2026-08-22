"""
A reliability diagram and ECE are the standard, established 
way researchers answer exactly that question 
for any model that outputs a confidence/probability not just a RAG specific topic 

aab what is well-calibrated actaully ?
among all the times your system said "I'm 80% confident," 
it was actually correct about 80% of the time. Not 95%, not 50% 
stated confidence ==  empirical accuracy at that confidence level.

mechanisms :

    bin predictions by confidence score 
               |   taking all 76 predicts and sorting by combined_confidence and split into buckets 
               |
    For each bin, compute two numbers:    
                |   Average confidence and Average accuracy
                |     
    Plot confidence (x-axis) vs. accuracy (y-axis)
               |
               |
     Deviation from the diagonal is the miscalibration  

     mtlb if    ECE = Σ (n_bin / N) * |accuracy_bin - confidence_bin|
     case 1 . points above the diagonal mean the system is underconfident (it's actually more accurate than it claims)
     case 2 . points below the diagonal mean it's overconfident (claims more certainty than its actual accuracy supports) 
"""
