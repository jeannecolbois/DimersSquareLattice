# SquareLatticeDimers

Julia code for computing the partition function of the classical dimer model on the 2D
square lattice with [PEPSKit.jl](https://github.com/QuantumKitHub/PEPSKit.jl), used as a
benchmark for 
- CTMRG convergence
- environment initialization
- correlation functions 
The exact per-site free energy is Catalan's constant divided by π.

The local tensor follows the PEPSKit leg convention `T: W ⊗ S ← N ⊗ E`.

Note that the two scripts below use *different* PEPSKit branches. 

## Usage

### Entropy and correlations: DimersSquare.jl

```julia
χs, λ, exact = compute_oneper()
```

This runs CTMRG (`leading_boundary`) for a series of bond dimensions and plots the
convergence of log(λ) against the exact value.

```julia
main()
```
This runs CTMRG and computes the correlations for a series of bond dimensions and saves all the results. 

`DimersSquareTestInit.jl` is a work-in-progress script to check the custom environment
initializer on the `lb/initialize_env` branch of PEPSKit.
