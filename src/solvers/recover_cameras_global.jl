function recover_camera_SpanningTree(F_multiview::AbstractSparseMatrix; kwargs...) 
    num_cams = size(F_multiview, 1)
    Adj = sparse(zeros(num_cams,num_cams))
    Adj[findall(F_multiview .!= 0)] .= 1
    
    G = Graph(Adj)
    if num_cams < 40
        Ps = SizedVector{num_cams, Camera{Float64}}(repeat([Camera_canonical], num_cams))
    else
        Ps = Vector{Camera{Float64}}(repeat([Camera_canonical], num_cams))
    end

    ST = prim_mst(G)
    root_node = setdiff(collect(1:num_cams), [v.dst for v in ST])[1]
   
    for e in ST
        src = e.src
        dst = e.dst
        if Ps[dst] != Camera_canonical
            continue
        end
        
        #loop
        path = []
        while src != root_node && Ps[dst] == Camera_canonical
            append!(path, src)
            src = ST[src-1].src
        end
        for p in reverse(path)
            # X[p] = inv(Z[ ST[p-1].src , p])*X[ST[p-1].src]
            # Ps[p] = get_canonical_cameras
        end

        # println(src, "\t", dst, "\t", path)
        Ps[dst] = get_canonical_cameras(F_multiview[dst, e.src])[2]

    end 
    return Cameras{Float64}(Ps)

end

function get_canonical_cameras(F::FundMat{T}) where T<:AbstractFloat
    # Returns {P1, P2} with input F_21 
    P₁ = Camera_canonical
    F_svd = svd(F)
    e′ = F_svd.U[:,end] #left null space of F         
    e′ₓ = make_skew_symmetric(SVector{3,T}(e′))
    v = rand(3)
    λ = rand()
    
    P₂ = zeros(3,4)  
    P₂[1:3,1:3] = e′ₓ*F + e′*v' 
    P₂[:,end] = λ*e′
    
    return Cameras{T}([P₁,P₂])
end 

