Hadoop Exercises

1. CoOccurance Matrix

	a. The ''Pairs'' Design Pattern
	The basic (and maybe most intuitive) implementation of this exercise is the Pair design pattern. The basic idea is to emit, for each couple of words in the same line, the couple itself (or pair) and the value 1. For example, in the line w1 w2 w3 w1, we emit (w1,w2):1, (w1, w3):1, (w2:w1):1, (w2,w3):1, (w2:w1):1, (w3,w1):1, (w3:w2):1, (w3,w1):1, (w1, w2):1, (w1, w3):1. Essentially, the reducers need to collect enough information from mapper to ''cover'' each individual ''cell'' of the co-occurrence matrix.

	b. The ''Stripes'' Design Pattern
	For example, in the line w1 w2 w3 w1, we emit:
		w1:{w2:1, w3:1}, w2:{w1:2,w3:1}, w3:{w1:2, w2:1}, w1:{w2:1, w3:1}
	Note that, instead, we could emit also:
		w1:{w2:2, w3:2}, w2:{w1:2,w3:1}, w3:{w1:2, w2:1}
