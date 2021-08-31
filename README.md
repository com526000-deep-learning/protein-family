# Protein Family Classification
Classify the protein structures through the sequences of amino acid on the Research Collaborators for Structural Bioinformatics Protein Data Bank dataset. We implemented the following deep neural networks:

- 1d CNN without n-gram
- 1d CNN with 4-gram
- GRU with 3-gram
- 1d CNN + GRU with 3-gram
and other not-so-great DNNs...

Result:
Above 0.97 F1-score on 10 classes of protein structures using 1d CNN with 4-gram.
