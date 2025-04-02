# TFBind
TFBind: Predict whether the transcription factor protein sequence binds to the DNA gene sequence

# Dependencies
TFBind works under Python 3.11.4
The required dependencies for TFBind are as follows：
python ==3.11.4
torch==2.0.1
numpy
pandas==2.1.1
scikit-learn

# Input
TFBind Two files are required for input, one is the amino acid protein sequence of the transcription factor, and the other is the DNA nucleotide sequence. The protein sequence does not exceed 1451bp, and the DNA sequence does not exceed 150bp.

# Output
The output is 1 and 0. 1 represents the potential combination possibility in model calculation, and 0 represents the potential non-combination possibility.

# Example
pip install TFBind
import TFBind
tf = TFBind.TFBind()
result = tf.get_combin_list(protein_sequence, dna_sequence)
print(result)

# Note
{"protein_length_max": 1451, "dna_length_max": 150, "max_length": 1880}, these are the maximum training lengths of protein, DNA, and combined in the training data set. Use the input amino acid 80bp sequence as a sliding window to determine whether there is an amino acid sequence with a chip binding site. The specific code is in __init__.py.In the result list, the fields start, end, and dna are included, representing the start position, end position, and predicted binding sequence of the input sequence.

Publication
Liu J, Shi X, Zhang Z, Cen X, Lin L, Wang X, Chen Z, Zhang Y, Zheng X, Wu B, Miao Y. Deep Neural Network-Mining of Rice Drought-Responsive TF-TAG Modules by a Combinatorial Analysis of ATAC-Seq and RNA-Seq. Plant Cell Environ. 2025 Mar 31. doi: 10.1111/pce.15489. Epub ahead of print. PMID: 40165388.









