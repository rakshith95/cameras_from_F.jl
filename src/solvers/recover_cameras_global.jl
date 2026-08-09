function recover_camera_SpanningTree(F_multiview::AbstractSparseMatrix{FundMat{T}}, weights=ones(size(F_multiview)...); kwargs...) where T<:AbstractFloat
    num_cams = size(F_multiview, 1)
    Adj = sparse(zeros(num_cams,num_cams))
    for i=1:num_cams-1
        for j=i+1:num_cams
            if iszero(F_multiview[i,j])
                continue
            end
            Adj[i,j] = 1;
        end
    end
    Adj = Adj + transpose(Adj);
    Adj = Adj .* weights;

    # G = Graph(Adj)
    G = SimpleWeightedGraphs.SimpleWeightedGraph(Adj);

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
        Ps[dst] = get_camera(Ps[src], F_multiview[dst, src])

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

function get_camera(P₁::Camera{T}, F::FundMat{T}) where T<:AbstractFloat
    # From: A closed form solution for viewing graph construction in uncalibrated vision
    A₁ = @view P₁[:,1:3];
    a₁ = @view P₁[:,end]
    e₂₁ = get_NullSpace_svd(F);
    e₂₁ₓF = make_skew_symmetric(e₂₁)*F;
    
    P₂ = Camera{T}([e₂₁ₓF*A₁ e₂₁ₓF*a₁+e₂₁])
    return P₂ 
end