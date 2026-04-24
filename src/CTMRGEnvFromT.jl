
# ---------------------------------------------------------
# Author: Jeanne Colbois
# Year: 2026
# Description: Initiazation of CTMRG environment from the local tensor
# ---------------------------------------------------------

function CTMENVfromT(Z)

    Venv = space(Z[1])[1] # starting point

    envref =  CTMRGEnv(ones,ComplexF64,Z,Venv) # build the space

    T1 = Z[1] # get the local tensor

    # remember that the order of legs in T is WEST, SOUTH, NORTH, EAST
    eyew = ones(space(T1)[1]',one(space(T1)[1]))/sqrt(2);
    @show eyew
    eyes = ones(space(T1)[2]',one(space(T1)[2]))/sqrt(2);
    eyen = ones(space(T1)[3]',one(space(T1)[3]))/sqrt(2);
    eyee =ones(space(T1)[4]',one(space(T1)[4]))/sqrt(2);

     """
        C_nowrthwest --- in
            |
            out
    """
    @tensor C_nw[χs; χe] := eyew[χw]*eyen[χn]*T1[χw,χs,χn,χe]
    @show C_nw
    envref.corners[PEPSKit.NORTHWEST] =C_nw; 



    """
        in 
        | 
        C_southwest --- out
    """
    @tensor C_sw[χe; χn] :=  eyew[χw]*eyes[χs]*T1[χw,χs,χn,χe];
    envref.corners[PEPSKit.SOUTHWEST] = flip(C_sw,[1]);


      """
            out- C_northeast
                         |
                        in
    """
    @tensor C_ne[χw; χs] :=  eyen[χn]*eyee[χe]*T1[χw,χs,χn,χe];
    envref.corners[PEPSKit.NORTHEAST] = flip(C_ne,[2]);
    """             out 
                    |       
            in- C_southeast
    """
    @tensor C_se[χn; χw] :=  eyes[χs]*eyee[χe]*T1[χw,χs,χn,χe];
    envref.corners[PEPSKit.SOUTHEAST] =flip(C_se,[1,2]);
    


    """
        out --- E_north -- in 
                  |
                 out 
    """
    @tensor E_north[χw, χs; χe] :=  eyen[χn]*T1[χw,χs,χn,χe];
    envref.edges[PEPSKit.NORTH] = E_north;


    
     """
        in
        | 
        E_west --- out
        | 
        out 
    """
    @tensor E_west[χs, χe; χn] :=  eyew[χw]*T1[χw,χs,χn,χe];
    envref.edges[PEPSKit.WEST] = E_west;


    """
                 out 
                 |
        in --- E_south -- out         
    """
    @tensor E_south[χe, χn; χw] :=  eyes[χs]*T1[χw,χs,χn,χe];
    envref.edges[PEPSKit.SOUTH] = flip(E_south, [1,3]);


    """
               out
                | 
        out --- E_east 
                | 
                in 
    """
    @tensor E_east[χn, χw; χs] :=  eyee[χe]*T1[χw,χs,χn,χe];
    envref.edges[PEPSKit.EAST] = flip(E_east, [1,3]);

    return envref
end