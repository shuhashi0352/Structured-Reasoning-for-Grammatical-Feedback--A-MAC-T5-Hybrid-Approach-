OOD => Out-of-Distribution
Data is evenly distributed across error tags -> n instances per error tag

Oneshot data was filtered out in the order of:

train(50) => test(100) => dev(50)

Meaning:
Total_rest (oneshot) = Original - 50 - 100 - 50 
= Original - 200

Test data (97 + 3 = 100)
R:VERB:INFL: 2 examples missing
R:NOUN:POSS: 1 example missing
=> making up for the rest of three instances by adding three types of "OTHERS"
(R:OTHER, U:OTHER, M:OTHER)

Dev data (48 + 2 = 50)
R:VERB:INFL: 1 example missing
R:NOUN:POSS: 1 example missing
=> making up for the rest of three instances by adding two types of "OTHERS"
(R:OTHER, U:OTHER)




TAG TYPES (99 in total)
M:ADJ: 2 examples
M:ADV: 2 examples
M:CONJ: 2 examples
M:CONTR: 2 examples
M:DET: 2 examples
M:NOUN: 2 examples
M:NOUN:POSS: 2 examples
M:OTHER: 2 examples
M:PART: 2 examples
M:PREP: 2 examples
M:PRON: 2 examples
M:VERB: 2 examples
M:VERB:FORM: 2 examples
M:VERB:TENSE: 2 examples
R:ADJ: 2 examples
R:ADJ:FORM: 2 examples
R:ADV: 2 examples
R:CONJ: 2 examples
R:CONTR: 2 examples
R:DET: 2 examples
R:MORPH: 2 examples
R:NOUN: 2 examples
R:NOUN:INFL: 2 examples
R:NOUN:NUM: 2 examples
R:NOUN:POSS: 2 examples
R:ORTH: 2 examples
R:OTHER: 2 examples
R:PART: 2 examples
R:PREP: 2 examples
R:PRON: 2 examples
R:SPELL: 2 examples
R:VERB: 2 examples
R:VERB:FORM: 2 examples
R:VERB:INFL: 1 examples (1 missing from the original)
R:VERB:SVA: 2 examples
R:VERB:TENSE: 2 examples
R:WO: 2 examples
U:ADJ: 2 examples
U:ADV: 2 examples
U:CONJ: 2 examples
U:CONTR: 2 examples
U:DET: 2 examples
U:NOUN: 2 examples
U:OTHER: 2 examples
U:PART: 2 examples
U:PREP: 2 examples
U:PRON: 2 examples
U:VERB: 2 examples
U:VERB:FORM: 2 examples
U:VERB:TENSE: 2 examples

