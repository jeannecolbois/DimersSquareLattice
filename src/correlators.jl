
# Additional functions for PEPSKit for partition functions 
# Jeanne Colbois, 2026
using TensorOperations

# function below should be in PEPSKit.jl / src / algorithms / contractions / correlator / partitionfunction.jl

function start_correlator(
    i::CartesianIndex{2}, 
    Z::InfinitePartitionFunction, 
    Oi::TensorMap,
    env::CTMRGEnv)

    E_north = env.edges[PEPSKit.NORTH]
    """
        out --- E_north -- in 
                  |
                 out 
    """
    
    E_south= env.edges[PEPSKit.SOUTH]
    """
                 out 
                 |
        in --- E_south -- out         
    """
    E_west = env.edges[PEPSKit.WEST]
    """
        in
        | 
        E_west --- out
        | 
        out 
    """
    C_northwest = env.corners[PEPSKit.NORTHWEST]
    """
        C_nowrthwest --- in
            |
            out
    """
    
    C_southwest = env.corners[PEPSKit.SOUTHWEST]
    """
        in 
        | 
        C_southwest --- out
    """
    Tz = Z[1+mod(i[1],length(Z))] # get the tensor at the right unit cell
    #Oi = Oi[1+mod(i[1],length(Oi))] # get the operator at the right unit cell

    """
    ``T : W ⊗ S ← N ⊗ E``. Here, ``N``, ``E``, ``S`` and ``W`` denote the north, east, south
    and west spaces, respectively.
    
               N ←
              ╱
             ╱
      ← W---- ----E ←
           ╱
          ╱
      ←  S 

        
        CNW ←   r-1
        ↓    
        EW → r
        ↓    
        CSW →    r+1
        c-1  c  
    """    
    
    @tensor Eleft[χS Dw; χN] := 
        C_northwest[χNW; χN] * 
        E_west[χSW Dw; χNW] *
        C_southwest[χS; χSW]
    
    """
        
        CNW ← EN ←   r-1
        ↓    |
        EW - T    r
        ↓    |
        CSW → Es →   r+1
        c-1  c 
    """

    @tensor Vn[χSE De; χNE] :=  # normal tensor 
        Eleft[χS Dw; χN] *
        Tz[Dw Ds; Dn De] * 
        E_north[χN Dn; χNE] *
        E_south[χSE Ds; χS]

    @tensor Vo[χSE De; χNE] :=  # with operator 
        Eleft[χS Dw; χN] *
        Oi[Dw Ds; Dn De] * 
        E_north[χN Dn; χNE] *
        E_south[χSE Ds; χS]
    
    """ 
    checked 
    """

    return Vn, Vo
end

function end_correlator_numerator(
    j::CartesianIndex{2}, 
    Vo::TensorMap,
    Oj::TensorMap,
    env::CTMRGEnv)

    """
        out --- E_north -- in 
                  |
                 out 
    """
    E_north = env.edges[PEPSKit.NORTH]
    
    """
                 out 
                 |
        in --- E_south -- out         
    """
    E_south= env.edges[PEPSKit.SOUTH]
    
    """
               out
                | 
        out --- E_east 
                | 
                in 
    """
    E_east = env.edges[PEPSKit.EAST]
    
    
    
    C_northeast = env.corners[PEPSKit.NORTHEAST]
    """
            out- C_northeast
                         |
                        in
    """

    C_southeast = env.corners[PEPSKit.SOUTHEAST]
    """             out 
                    |       
            in- C_southeast
    """

    #Oj = Oj[1+mod(j[1],length(Oj))] # get the operator at the right unit cell

    # apply the operator Oj at site j 

    @tensor Vo[χS, De; χN] := 
        Vo[χSW, Dw; χNW] *
        Oj[Dw, Ds; Dn, De] *
        E_north[χNW, Dn; χN] *
        E_south[χS, Ds; χSW]

    # close 
    @tensor numerator = 
        Vo[χS De; χN] * C_northeast[χN; χNE] *
        E_east[χNE De; χSE] * C_southeast[χSE; χS]

    #checked 

    return numerator
end

function end_correlator_denominator(
    j::CartesianIndex{2}, 
    Vn::TensorMap,
    env::CTMRGEnv)

    E_east = env.edges[PEPSKit.EAST]
    """
            out
            |
    --- E_east
    
            |
            in
    """

    C_northeast = env.corners[PEPSKit.NORTHEAST]
    """
            out- C_northeast
                         |
                        in
    """
    C_southeast = env.corners[PEPSKit.SOUTHEAST]
    """
           out
            |       
    in- C_southeast

    """

    # contract with the northeast and southeast corners 
     @tensor denominator = 
        Vn[χS De; χN] * C_northeast[χN; χNE] *
        E_east[χNE De; χSE] * C_southeast[χSE; χS]

    #checked 
    return denominator
end

function apply_horiz_transfer_matrix!(
    V::TensorMap, 
    E_north::TensorMap, 
    Tz::TensorMap, 
    E_south::TensorMap)

     @tensor V[χSE De; χNE] := 
        V[χS Dw; χN] *
        Tz[Dw Ds; Dn De] * 
        E_north[χN Dn; χNE] *
        E_south[χSE Ds; χS]
    
    return V
end

# function below should be in src/ algorithms / 
function correlator_horizontal(
    Z::InfinitePartitionFunction, 
    Oi, Oj,
    i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}},
    env::CTMRGEnv)


    # checks: 
    size(Z) == (1,1) ||
        throw(ArgumentError("Currently only implemented for unit cell size (1,1)"))
    typeof(Oi) <: TensorMap ||
        throw(ArgumentError("Oi must be a TensorMap"))
    typeof(Oj) <: TensorMap ||
        throw(ArgumentError("Oj must be a TensorMap"))
    all(==(i[1]) ∘ first ∘ Tuple, js) ||
        throw(ArgumentError("Not a horizontal correlation function"))
    issorted(vcat(i, js); by = last ∘ Tuple) ||
        throw(ArgumentError("Not an increasing sequence of coordinates"))


    # preallocate with the correct scalar type 
    G = similar(
        js, 
        TensorOperations.promote_contract(
            scalartype(env), scalartype(Oi)
        ),
    )

    Vn, Vo = start_correlator(i, Z, Oi, env) # to implement above
    i+= CartesianIndex(0,1)
    for (k,j) in enumerate(js)
        # transfer matrix until left of site j
        while j > i 
            E_north = env.edges[PEPSKit.NORTH]
            E_south= env.edges[PEPSKit.SOUTH]

            Vo = apply_horiz_transfer_matrix!(Vo, E_north, Z[1+mod(i[1],length(Z))], E_south)
            Vn = apply_horiz_transfer_matrix!(Vn, E_north, Z[1+mod(i[1],length(Z))], E_south)


            # TM = edge_transfer_matrix(E_north, Z[1+mod(i[1],length(Z))], E_south) # this is a very bad idea
            # 
            #Vo = Vo * TM;
            #Vn = Vn * TM; 
            
            i += CartesianIndex(0,1)
        end 

        # compute the overlap with  Oj 
        numerator = end_correlator_numerator(j, Vo, Oj, env) # to implement above

        # transfer right of size j 
        E_north = env.edges[PEPSKit.NORTH]
        E_south = env.edges[PEPSKit.SOUTH]
        
        if k < length(js)
            Vo = apply_horiz_transfer_matrix!(Vo, E_north, Z[1+mod(i[1],length(Z))], E_south) # update the left Vo 
        end
        Vn = apply_horiz_transfer_matrix!(Vn, E_north, Z[1+mod(i[1],length(Z))], E_south) # always update Vn
        i += CartesianIndex(0,1)

        denominator = end_correlator_denominator(j, Vn, env) # to implement above. note that the transfer matrix has already been applied to Vn
        G[k] = numerator / denominator
    end 
    return G
end