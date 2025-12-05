# README.md – Benchmark Problem Sets for IPM-GNN Evaluation

This repository contains the extended benchmark suite used in the project "Benchmarking IPM-GNN: Performance, Stability and Transferability in Linear Programming".  
We evaluate IPM-GNN on five structurally diverse families of linear programming instances that are widely used to test generalization of learning-based solvers.

| Problem Type       | Short Name | Description                                                                                 | Typical Size (vars × cons) | Density / Characteristics                              | Source / Generator                                                                 |
|-------------------|------------|---------------------------------------------------------------------------------------------|----------------------------|--------------------------------------------------------|------------------------------------------------------------------------------------|
| Maximum Weight Independent Set | `Indset`   | LP relaxation of the maximum-weight independent set problem on random Erdős–Rényi graphs. | ≈ 1,000 × 1,000            | Very sparse (edge probability ≈ 0.05)                  | Custom generator based on NetworkX + random weights                                |
| Set Covering      | `Setcover` | Classic minimum-cost set-covering instances (set universe size ≈ 1,000, ≈ 1,000 sets).      | ≈ 700–1,000 × 1,000        | Moderate sparsity, highly structured row/column patterns | Generated using the standard Beasley-style generator (adapted from OR-Library)     |
| Uncapacitated Facility Location | `Fac`      | Standard uncapacitated facility location problems with clients ≈ facilities ≈ 900–1,000.   | ≈ 900 × 1,800–2,000        | Block-structured (assignment + opening costs)          | Instances adapted from the classic UFLib and OR-Library datasets                   |
| Combinatorial Auction Winner Determination | `Cauction` | Winner-determination problem in combinatorial auctions (≈ 500–600 bids, single-minded bidders). | ≈ 500–600 × 1,000–2,000   | Extremely sparse, highly irregular structure           | Generated using the CATS 2.0 (Combinatorial Auction Test Suite) generator          |
| Dense Random LPs  | `Hrandom`  | Randomly generated dense LPs with controlled density (default 0.9).                        | 500 × 500 (training)<br>up to 3,000 × 3,000 (scaling tests) | Very high density (≈ 90 % nonzeros), no combinatorial structure | Custom dense random generator (uniform coefficients in [0,1], RHS in [0,10])      |

### Why these five problems?
- They cover a broad spectrum of real-world combinatorial optimization structures (graph, set system, assignment, auction, and purely random).
- They exhibit dramatically different sparsity patterns and numerical properties, making them ideal for testing cross-domain transferability.
- Traditional solvers (simplex/IPM) behave very differently on each class, whereas a truly general learned solver should remain stable.



Happy solving!