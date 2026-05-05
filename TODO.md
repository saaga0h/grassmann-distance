# TODO

## REPL Experiments

### Grassmann distance for complementarity
Density matrix implementation already exists for complementarity ranking.
Open question: can Grassmann distance also work for complementarity, or is
it fundamentally a similarity-only metric? The geometry suggests DG measures
structural similarity of local manifolds — complementarity (what fills gaps)
may need a different framing. Worth testing in REPL with real embeddings:
- Compare DG rankings vs density matrix rankings on known complementary pairs
- Check whether large Grassmann distance correlates with complementarity
  or just with dissimilarity (not the same thing)
- If DG can do both, the module interface simplifies significantly
