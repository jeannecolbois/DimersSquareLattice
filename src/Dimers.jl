# ---------------------------------------------------------
# Author: Jeanne Colbois
# Year: 2026
# Description: Partition function of dimers on the square lattice using PEPSKit
# ---------------------------------------------------------

##########
# HELPER FUNCTIONS
##########
# Note : PEPSkit convention : 
#
#``T : W ⊗ S ← N ⊗ E``. Here, ``N``, ``E``, ``S`` and ``W`` denote the north, east, south
#and west spaces, respectively.
#
#           N ←
#          ╱
#         ╱
#  ← W---- ----E ←
#       ╱
#      ╱
#  ←  S 


"""
    function dimers_square_lattice()

    creates the local tensor for the full covering of dimers on the square lattice
    returns a TensorMap
"""
function dimers_square_lattice(; monomers::Bool=false)
    # Local tensor 
    # 1 = no dimer; 2 = dimer present
    # O lives on the vertices of the square lattice
    #      1
    #      |
    #   1- O - 2 and rotations are non-zero, all other terms are zero
    #      |
    #      1


    O = zeros(2,2,2,2);  # note: here W, S, N, E tacitly
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        if (i + j + k + l) == 5 # there is exactly one dimer
            O[i,j,k,l] = 1.0 
        end
    end 
    TMS = ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2
    if !monomers 
        return TensorMap(O, TMS)
    else 
        # implement a local tensor that counts 1 if there is NO dimer around the site (monomer present)
        Omonomer = zeros(2,2,2,2)
        Omonomer[1,1,1,1] = 1.0
        return TensorMap(O, TMS), TensorMap(Omonomer, TMS)
    end 
end

""" 
    function dimers_partitionfunction()

    creates the infinite partition function object for dimers on the square lattice
    returns an InfinitePartitionFunction
"""
function dimers_partitionfunction(; monomers::Bool=false, unitcell::Tuple{Int,Int}=(1,1))
    if !monomers
        T = dimers_square_lattice()
        Z = PEPSKit.InfinitePartitionFunction(T, unitcell=unitcell)
        return Z
    else
        T, Tmonomer = dimers_square_lattice(monomers=monomers)
        Z = PEPSKit.InfinitePartitionFunction(T, unitcell=unitcell)
        return Z, Tmonomer
    end 
end 

function monomercorrelations(Z, Tm,jmin, jmax, env)
    i = CartesianIndex(1,1); 
    js = collect(CartesianIndex(1, j) for j in jmin:jmax)
    return correlator_horizontal(Z, Tm, Tm, i, js,  env)
end


function compute(;correlation::Bool=false, reuse::Bool=true)

    χs = [8, 16, 32, 64]#, 56, 64]
    
    Z,Tmono = dimers_partitionfunction(monomers = true)
    if correlation
        spectrumh = [zeros(ComplexF64, min(20,χ)) for χ in χs]
        spectrumv = [zeros(ComplexF64, min(20,χ))  for χ in χs]
        ξhs = zeros(length(χs))
        ξvs = zeros(length(χs))
        delsh = zeros(length(χs))
        delsv = zeros(length(χs))
        correls = zeros(ComplexF64, length(χs), 199)
        loc = zeros(length(χs))
    end
    λ = zeros(length(χs ))
    envold = nothing; 
    for (i, χ) in enumerate(χs)
        println("----------------Computing for χ = $χ----------------")
        Venv = ℂ^χ # bond dimension χ
        if !reuse || i == 1
            env₀ = CTMRGEnv(Z, Venv)
            @show network_value(Z, env₀)
        else 
            println("Reusing environment from previous χ")
            dummy_alg = (;
                alg = :sequential,
                trunc = truncspace(Venv),
                maxiter = 5,
            ) # grow environment 
            env₀, = leading_boundary(envold, Z; dummy_alg...)
            @show expectation_value(Z, (1,1) => Tmono, env₀)
            @show network_value(Z, env₀)
        end
        #@warn("Check how to set largest real; check which CTMRG scheme is the default; Check how to grow from a given starting environment")
        simultaneous_alg = (;
            alg = :simultaneous,
            tol = 1e-11,
            maxiter = 20000,
            verbosity = 2,
            trunc = FixedSpaceTruncation(),
        )
        env, = leading_boundary(env₀, Z; simultaneous_alg...);
        λ[i] = abs(network_value(Z, env)) 
        loc[i] = expectation_value(Z, (1,1) => Tmono, env)
        if correlation
            ξ_h, ξ_v, λ_h, λ_v = correlation_length(Z, env; num_vals = min(20, χ))
            λ_h = λ_h[1]; λ_v = λ_v[1];
            ξhs[i] = ξ_h[1];
            ξvs[i] = ξ_v[1];
            spectrumh[i] = λ_h;
            spectrumv[i] = λ_v;
            delsh[i] = -log.(abs(λ_h[4]/λ_h[2]))
            delsv[i] = -log.(abs(λ_v[4]/λ_v[2]))
            correls[i, :] = monomercorrelations(Z, Tmono, 2, 200, env)
        end
        envold = env;
    end
    if correlation
        return χs, λ, ξhs, ξvs, delsh, delsv, spectrumh, spectrumv, loc, correls 
    else 
        return χs, λ, loc
    end
end
