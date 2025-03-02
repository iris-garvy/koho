# koho

Spectral theory for cellular sheaves, and sheaf neural network library. 

> Do Not Use Yet!!

### what

I read the paper [Towards A Spectral Theory of Cellular Sheaves](https://arxiv.org/pdf/1808.01513) recently and realized how much of a game changer this is. It also sort of gave me a new perspective on how powerful [sheaf neural networks](https://arxiv.org/abs/2012.06333) can be. While traditionally used in graph settings, there's nothing stopping you from building one over a generic cell complex (as the authors note), as it fundamentally is the same construction, same laplacian respresentation and everything. Tangentially, there was this paper recently introduced called [Higher-Order Laplacian Renormalization](https://arxiv.org/abs/2401.11298) that extends the ideas of renormalization group theory in physics to arbitrary higher-order networks. This is kind of amazing, it pretty much would directly let you analyze multiscale laplacian dynamics on the network right away. This has nothing to do with cellular sheaves currently, but why not see what happens if you use sheaf laplacians, could this be extensible to cell complexes generally as well?

This library is an attempt to build a generic cellular sheaf, sheaf neural network over a cell complex, a whole set of spectral analysis tools, along with an attempt at exploring sheaf laplacian renormalization.