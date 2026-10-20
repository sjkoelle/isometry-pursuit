# TODO

- [x] ~~Add a formal non-linear isometry definition to `archival/background.tex`, copied from the workshop paper version.~~
- [ ] In `archival/background.tex`, revise "Functions minimized by isometries" to include distortion as a conformality-minimized loss from Kohli et al., then contrast it with losses uniquely minimized by isometries.
- [ ] Run a quick experiment to check whether the method-of-frames solution norm of a rescaled matrix, using the normalization in this paper, equals the method-of-frames solution norm of its inverse.
- [ ] Add warm starts for restricted-BP backward pruning so each candidate deletion can initialize from the previous coefficient matrix with the removed row dropped.
