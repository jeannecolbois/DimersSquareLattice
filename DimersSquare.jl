
# ---------------------------------------------------------
# Author: Jeanne Colbois
# Year: 2026
# Description: Testing initialization of CTMRG using dimer models on the square lattice
# ---------------------------------------------------------

import Pkg; Pkg.activate(".");
using Plots
using Random, LinearAlgebra
using TensorKit
using DelimitedFiles
using LaTeXStrings
using QuadGK
Pkg.develop(url="https://github.com/QuantumKitHub/PEPSKit.jl.git")
using PEPSKit
#########

# MAIN SCRIPT

#########
include("./src/Dimers.jl")
include("./src/correlators.jl")
function compute_oneper()
    χs = [8,12,16,20,32]
    
    Z = dimers_partitionfunction()
    
    @show Z
    λ = zeros(length(χs ))
    for (i, χ) in enumerate(χs)
        Venv = ℂ^χ # bond dimension χ 
        println("Random environment initialization with χ = $χ")
        env₀ = CTMRGEnv(Z, Venv)
        env, = leading_boundary(env₀, Z; tol = 1.0e-9, maxiter = 10000);
        λ[i] = abs(network_value(Z, env)) 
        @show log(λ[i])
        @show exact= MathConstants.catalan/π
        @warn("TODO : implement correlation functions : dimer-dimer; monomer-monomer")
    end
    p = plot(χs, log.(λ) .- MathConstants.catalan/π, marker = :o, xlabel = "χ", ylabel = "log(λ)", title = "Convergence of partition function with bond dimension χ")
    display(p)
    return χs, λ, MathConstants.catalan/π
end


function main(; recompute::Bool= true)
    if recompute
        χs, λoneper, ξhs, ξvs, delsh, delsv, spectrumh, spectrumv, loc, correls = compute(;correlation = true)
        
        writedlm("./results/Dimers_chis.dat", χs)
        writedlm("./results/Dimers_lambdaoneper.dat", λoneper)
        writedlm("./results/Dimers_xih.dat", ξhs)
        writedlm("./results/Dimers_xiv.dat", ξvs)
        writedlm("./results/Dimers_delh.dat", delsh)
        writedlm("./results/Dimers_delv.dat", delsv)
        for (i, χ) in enumerate(χs)
            writedlm("./results/Dimers_spectrumh_chi$(χ).dat", spectrumh[i])
            writedlm("./results/Dimers_spectrumv_chi$(χ).dat", spectrumv[i])
        end
        writedlm("./results/Dimers_monomercorrelations.dat", real.(correls))
        writedlm("./results/Dimers_monomerexpectation.dat", loc)
    else 
        χs = Vector{Int64}(readdlm("./results/Dimers_chis.dat")[:])
        λoneper = readdlm("./results/Dimers_lambdaoneper.dat")[:]
        ξhs = readdlm("./results/Dimers_xih.dat")[:]
        ξvs = readdlm("./results/Dimers_xiv.dat")[:]
        delsv = readdlm("./results/Dimers_delv.dat")[:]
        delsh = readdlm("./results/Dimers_delh.dat")[:]
        loc = readdlm("./results/Dimers_monomerexpectation.dat")[:]
        correls = readdlm("./results/Dimers_monomercorrelations.dat")
    end

    p1 = plot(dpi = 200, size = (400,300), fontfamily = "Computer Modern", framestyle = :box)
    plot!(p1, 1 ./ χs, log.(λoneper) .- MathConstants.catalan/π, marker = :o, msw = 0, xlabel = L"1/\chi", ylabel = L"S-S_{\mathrm{exact}}", title = "Entropy convergence", legend = false, 
    xlims = [0, maximum(1 ./ χs)*1.1])
    savefig(p1, "./results/Dimers_entropy_convergence.svg")

    p2 = plot(dpi = 200, size = (400,300), fontfamily = "Computer Modern", framestyle = :box)
    plot!(p2, 1 ./ ξhs, log.(λoneper) .- MathConstants.catalan/π, marker = :o, msw = 0, xlabel = L"1/\xi", ylabel = L"S-S_{\mathrm{exact}}", title = "Entropy convergence", legend = false)
    plot!(p2, xlims = [0, maximum(1 ./ ξhs)*1.1])
    savefig(p2, "./results/Dimers_entropy_convergence_xi.svg")

    p3 = plot(dpi = 200, size = (400,300), fontfamily = "Computer Modern", framestyle = :box, foreground_color_legend = false, background_color_legend = false) 
    plot!(p3, 1 ./ χs, 1 ./ ξhs, marker = :o, msw = 0, xlabel = L"1/\chi", ylabel = L"1/\xi_h", label = "horizontal", title = "Correlation length vs bond dim", legend = false)
    plot!(p3, 1 ./ χs, 1 ./ ξvs, marker = (:diamond,3), msw = 0, xlabel = L"1/\chi", ylabel = L"1/\xi", label = "vertical", title = "Correlation length vs bond dim", legend = false)
    plot!(p3, xlims = [0, maximum(1 ./ χs)*1.1], ylims = [0, maximum(vcat(1 ./ ξhs, 1 ./ ξvs))*1.1])
    savefig(p3, "./results/Dimers_xi_vs_chi.svg")

    p4 = plot(dpi = 200, size = (400,300), fontfamily = "Computer Modern", framestyle = :box)
    plot!(p4, delsh, 1 ./ ξhs, marker = :o, msw = 0, xlabel = L"\delta_h", ylabel = L"1/\xi_h", label = "horizontal", title = "Correlation length vs gap", legend = false)
    plot!(p4, delsv, 1 ./ ξvs, marker = (:diamond,3), msw = 0, xlabel = L"\delta", ylabel = L"1/\xi", label = "vertical", title = "Correlation length vs gap", legend = false)
    plot!(p4, xlims = [0, maximum(vcat(delsh, delsv))*1.1], ylims = [0, maximum(vcat(1 ./ ξhs, 1 ./ ξvs))*1.1])
    savefig(p4, "./results/Dimers_xi_vs_delh.svg")
    
    p5 = plot(dpi = 200, size = (440,270), fontfamily = "Computer Modern", foreground_color_legend = false, background_color_legend = false, framestyle = :box)
    for (i, χ) in enumerate(χs)
        plot!(p5, collect(1:199), abs.(correls[i, :]), msw = 0, label = L"%$(χ)")
    end
    plot!(p5, 1:199, 0.25 ./ sqrt.(collect(1:199)), lw = 2, ls = :dash, color = :black, label = L"\mathrm{Exact: }\frac{1}{4 \sqrt{r}}")
    plot!(p5, xlims = [1, 200], ylims = [1e-4, 1], yscale = :log10, xscale = :log10, legend = :outerright)
    plot!(p5, xlabel = L"r", ylabel = L"\langle m_i m_{i+r} \rangle ", title = "Monomer-monomer correlations")
    savefig(p5, "./results/Dimers_monomercorrelations.svg")
    display(p5)

    p6 = plot(dpi = 200, size = (400,300), fontfamily = "Computer Modern", framestyle = :box, foreground_color_legend = false, background_color_legend = false)
    plot!(p6,1 ./ ξhs  , correls[:,2], marker = :o, msw = 0, xlabel = L"1/\xi", ylabel = L"\langle m_i m_{i+r} \rangle ", label = L"r = 2", title = "Monomer corr. at even dist.")
    plot!(p6,1 ./ ξhs  , correls[:,4], marker = :o, msw = 0, xlabel = L"1/\xi", ylabel = L"\langle m_i m_{i+r} \rangle ", label = L"r = 4")
    plot!(p6,1 ./ ξhs  , correls[:,10], marker = :o, msw = 0, xlabel = L"1/\xi", ylabel = L"\langle m_i m_{i+r} \rangle ", label = L"r = 10",
    xlims = [0, maximum(1 ./ ξhs)*1.1], ylims = [0, maximum(correls[:,2])*1.1])
    savefig(p6, "./results/Dimers_monomercorrelations_distance2.svg")
    return p1, p2, p3, p4, p5, p6
end