
# ---------------------------------------------------------
# Author: Jeanne Colbois
# Year: 2026
# Description: Testing initialization of CTMRG using dimer models on the square lattice
# ---------------------------------------------------------

import Pkg; Pkg.activate(".");
using Random, LinearAlgebra
using TensorKit
using DelimitedFiles
using QuadGK
#########

# MAIN SCRIPT

########



Pkg.add(url="https://github.com/QuantumKitHub/PEPSKit.jl.git", rev="lb/initialize_env")
using PEPSKit

include("./src/Dimers.jl")
include("./src/CTMRGEnvFromT.jl")

Z = dimers_partitionfunction()
envref = CTMENVfromT(Z)


env0_prod = initialize_ctmrg_environment(Z, ProductStateInitialization(ones));
env1_prod = initialize_ctmrg_environment(Z, ApplicationInitialization(ones));

# trying to understand what is going on: 
for dir in 1:4
    Cnew = TensorMap(PEPSKit.EnlargedCorner(InfiniteSquareNetwork(Z), env0_prod, (dir,1,1)))

    @show envref.corners[dir].data 
    @show Cnew.data
    @show env1_prod.corners[dir].data
end


log2χ = 6
grow_alg = (;alg = :simultaneous, 
trunc = truncrank(2^log2χ), 
tol = 1e-9, 
maxiter = log2χ)
# grow envref until χ
envref_grown,info = leading_boundary(envref, Z; grow_alg...); 
envref_final, info = leading_boundary(envref_grown, Z; trunc = truncrank(2^log2χ), tol = 1.0e-8, maxiter = 10000, verbosity = 2);
λ =  abs(network_value(Z, envref_final)); 
@show abs(log(λ) -  MathConstants.catalan/π)

# same with the product state initialization
env0_prod_grown,info = leading_boundary(env0_prod, Z; grow_alg...); 
env0_prod_final, info = leading_boundary(env0_prod_grown, Z; trunc = truncrank(2^log2χ), tol = 1.0e-8, maxiter = 10000, verbosity = 2);
λ_prod =  abs(network_value(Z, env0_prod_final)); 
@show abs(log(λ_prod) -  MathConstants.catalan/π)
