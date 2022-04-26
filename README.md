# chembl_lstm.py

A script that uses LSTM (long short term memory) network to predict the AlogP value (a measure of hydrophobicity) of a chemical, given its SMILES-format formula. Uses a dataset obtained from ChEMBL (https://www.ebi.ac.uk/chembl/).

The file data.filtered.csv contains 909690 data rows, each of which represents a chemical and its properties. Of the 42 columns, columns 6 and 8 (counting from 0) contain the AlogP and Smiles data, respectively. 

AlogP is a float like 22.57 or -8.86. Some Smiles strings are:
* CC(=O)Oc1cc(S(=O)(=O)Nc2n[nH]c(Nc3ccc(C)cc3)n2)c(S)cc1Cl
* O=C1N[C@H](c2c[nH]c3cc(Br)ccc23)CN=C1c1c[nH]c2cc(Br)ccc12

To process each string as a "sentence", identify "words". To help the network understand the sentences, treat certain sequences as words:
 * Cl, Br (multi-letter atoms in the "organic subset")
 * =O (double-bonded oxygen, can only attach to 1 atom) 
 * Any unit in square brackets (e.g. [nH], [C@H], [N+], [O-])
 * but not letter-number combinations such as C1 or c23

All other characters are treated as words. Assumption: Square brackets will not contain square brackets. i.e. [[Se]O4] is impossible.
