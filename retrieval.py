"""
given a new query and get the top-k-passages , computing a number which 
will tell me how confident the retrieval was 

core idea of project is 'top score alone " is bad confidence signal if the top passage's similarity score is high, we're confident." That's wrong, and knowing why it's wrong is an interview-worthy insight.

What's more informative is the relative structure of the score distribution:
If the top result scores much higher than the 2nd, 3rd, 4th results → there's a clear "winner," meaning the corpus likely contains a passage that specifically addresses this query. High confidence.
If the top few results all score roughly the same → the retriever can't clearly distinguish what's relevant, either because the corpus lacks a good match, or the query is ambiguous, or several passages are plausibly relevant. Low confidence.
"""

