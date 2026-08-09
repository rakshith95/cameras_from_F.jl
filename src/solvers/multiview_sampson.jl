    # using ForwardDiff: JacobianConfig, Chunk, jacobian!
function preprocess_VG(A_wt::AbstractSparseMatrix{T}, n_trees::Int) where T<:AbstractFloat
    G = SimpleWeightedGraphs.SimpleWeightedGraph(A_wt);
    @assert Graphs.is_connected(G)
    
    Vg_rob =  Graphs.Graph( Graphs.prim_mst(G) )    
    Vg_A = Graphs.adjacency_matrix(Vg_rob)  

    for i=1:n_trees-1
        X = setdiff( [Graphs.Edge(ed.src, ed.dst) for ed in  Graphs.edges(G)] ,  Graphs.edges(Vg_rob));
        if size(A_wt,1) > size(Graphs.adjacency_matrix(Graphs.Graph(X)),1)
            break
        end
    
        G = SimpleWeightedGraphs.SimpleWeightedGraph( Graphs.adjacency_matrix(Graphs.Graph(X)).*A_wt)
        Vg_rob =  Graphs.Graph( Graphs.prim_mst(G) )    
        A_vg_rob = Graphs.adjacency_matrix(Vg_rob)
        if Graphs.nv(Vg_rob)<size(A_wt,1)
            println("Not a ST")
            A_vg_rob = sparse(zeros(size(A_wt)...))
        end
        # println(size(Graphs.adjacency_matrix(Vg_rob)))
        Vg_A = Vg_A + A_vg_rob
    end
    return Vg_A
end

function add_edges!(A_existing::AbstractSparseMatrix, A_wt::AbstractSparseMatrix{T}, num_edges_new::Int) where T<:AbstractFloat
    rvals = rowvals(A_wt)
    Wtvals = nonzeros(A_wt)

    candidates = Vector{Tuple{eltype(Wtvals), Int, Int}}()

    for j in 1:size(A_wt, 2)
        for r in nzrange(A_wt, j)
            i = rvals[r]

            # Work only on the upper triangle
            if i >= j || !iszero(A_existing[i,j])
                continue
            end

            w = Wtvals[r]
            push!(candidates, (w, i, j))
        end
    end

    # sort!(candidates, by = first)
    partialsort!(candidates, 1:min(num_edges_new,length(candidates)), by = first)
    for k in 1:min(num_edges_new, length(candidates))
        _, i, j = candidates[k]
        A_existing[i, j] = 1
        A_existing[j, i] = 1
    end
    return min(num_edges_new, length(candidates))
end

function mv_sampson_δ_track(track::track2D{point_id2D{T}}, F_multiview::AbstractSparseMatrix, EdgeWeights::AbstractSparseMatrix{T}; limit_constraints=Inf) where T<:AbstractFloat
    # Assume there is only 1 point in the track corresponding to any image.

    num_pts = length(track)
    img_ids = track.image_id
    num_cams = length(img_ids)
    # Get the subgraph for the track 
    F_sub = F_multiview[img_ids, img_ids]
    WeightsSub = EdgeWeights[img_ids, img_ids]
    num_constraints = div(nnz(F_sub),2)

    if num_constraints > limit_constraints
        num_constraints = Int(div(limit_constraints,(num_cams-1))*(num_cams-1))
        Adj_filtered = preprocess_VG(WeightsSub, Int(div(limit_constraints,(num_cams-1))) )
        # num_remaining = limit_constraints % (num_cams-1)
        # n_added = add_edges!(Adj_filtered, WeightsSub, num_remaining)
        # num_constraints += n_added
        F_sub .= F_sub .* Adj_filtered
    end

    C = zeros(T, num_constraints)
    J = zeros(T, num_constraints , 2*num_pts) #Sparse?
    # JJT = zeros(T, num_constraints, num_constraints)
    # JJT_cpy = copy(JJT)
    δ = zeros(T, 2*num_pts)

    # preprocess_VG

    nzvals = nonzeros(F_sub)
    rvals = rowvals(F_sub)
    xᵢ = zeros(T,3)
    xⱼ = zeros(T,3)
    xᵢ[end] = 1
    xⱼ[end] = 1
    
    Fij = zeros(T,3,3)
    Fji = zeros(T,3,3)
    
    Fjixᵢ = zeros(T,3)

    constraint_ct = 1;    
    # Start from a minimial spanning tree (constraints as weights) and add edges, instead of traversing and adding constraints this way 
    for j=1:size(F_sub,2)
        rangeCol = nzrange(F_sub,j)
        for r in rangeCol
            i = rvals[r]
            # skip lower triangle of matrix
            if j<=i  || constraint_ct > num_constraints
                continue
            end
            copyto!(Fij, nzvals[r])
            # copyto!(Fji, transpose(Fij))
            Fji .= transpose(Fij)
            xᵢ[1] = track[i].point[1];
            xᵢ[2] = track[i].point[2];
            
            xⱼ[1] = track[j].point[1];
            xⱼ[2] = track[j].point[2];

            mul!(Fjixᵢ, Fji, xᵢ) 
            C[constraint_ct] =  dot(xⱼ,Fjixᵢ)
    
            J[constraint_ct, (i-1)*2+1] = dot(xⱼ, Fji[:,1])
            J[constraint_ct, i*2] = dot(xⱼ, Fji[:,2])

            J[constraint_ct, (j-1)*2+1] = dot(xᵢ, Fij[:,1])
            J[constraint_ct, j*2] = dot(xᵢ, Fij[:,2])
           
            constraint_ct += 1
        end
    end
    # δ = -pinv(J)*C # correct formula
    δ = -J\C # correct formula
    return δ
end

function mv_sampson_δ_tracks(fs::AbstractVector{T}, fs_svd::AbstractVector{S},tracks::AbstractVector{track2D}, F_info::Tuple{AbstractSparseMatrix{FundMat{T1}}, Vector{Int}}, subgraphs::Vector{AbstractSparseMatrix{Int}}; disp=false) where {S,T,T1<:AbstractFloat}
    F_mv, F_rvals = F_info
    Fnz = [apply_update(fs[(i-1)*7+1:i*7], fs_svd[i]) for i in eachindex(fs_svd)]
    nconstr_prev = div(nnz(subgraphs[1]),2)

    C = zeros(T, nconstr_prev)
    δ = Vector{T}()
    
    for (k,track) in enumerate(tracks) 
        nconstr = div(nnz(subgraphs[k]),2)
        npts = length(track)
        img_ids = track.image_id
        main_sub_ind = Dict{Int,Int}(zip(img_ids, collect(1:npts)))    
        resize!(C, nconstr)
        J = zeros(T, nconstr, 2*npts)
        xᵢ = zeros(T,3)
        xᵢ[end] = 1
        xⱼ = zeros(T,3)
        xⱼ[end] = 1
        Fij = zeros(T,3,3)
        Fji = zeros(T,3,3)    
        Fjixᵢ = zeros(T,3)
    
        constraint_ct = 1;
    
        for j in img_ids
            rangeCol = nzrange(F_mv,j)
            for r in rangeCol
                i = F_rvals[r]
                if !haskey(main_sub_ind,i)
                    continue
                end
                # skip lower triangle 
                if j<=i  || iszero(subgraphs[k][main_sub_ind[i],main_sub_ind[j]]) || constraint_ct > nconstr ### COntinue here 
                    continue
                end
                copyto!(Fij, Fnz[r])
                # copyto!(Fji, transpose(Fij))
                Fji .= transpose(Fij)
                xᵢ[1] = track[main_sub_ind[i]].point[1];
                xᵢ[2] = track[main_sub_ind[i]].point[2];

                xⱼ[1] = track[main_sub_ind[j]].point[1];
                xⱼ[2] = track[main_sub_ind[j]].point[2];

                mul!(Fjixᵢ, Fji, xᵢ) 
                C[constraint_ct] =  dot(xⱼ,Fjixᵢ)
            
                J[constraint_ct, (main_sub_ind[i]-1)*2+1] = dot(xⱼ, Fji[:,1])
                J[constraint_ct, main_sub_ind[i]*2] = dot(xⱼ, Fji[:,2])

                J[constraint_ct, (main_sub_ind[j]-1)*2+1] = dot(xᵢ, Fij[:,1])
                J[constraint_ct, main_sub_ind[j]*2] = dot(xᵢ, Fij[:,2])
                
                constraint_ct += 1
            end
        end

        # append!(δ,-J\C)) # correct formula
        append!(δ,transpose(J)*(cholesky!(J*J')\C)) # minus can be ignored
        # append!(δ,-pinv(J)*C) # correct formula
    end
    return δ
end

# Need to speed up 
function refineF_trackwise!(F_mult::AbstractSparseMatrix{FundMat{T}}, keypoints::Vector{Pts2D{T}}, CorresMat::AbstractSparseMatrix{correspondences2D{keypoint_id}}, ConstraintGraph::AbstractSparseMatrix{T}) where T<:AbstractFloat
    F_nz = nonzeros(triu!(F_mult,1)); #### scale to make svdvals[1] = 1
    F_nz_svd = [ svd(F/svdvals(F)[1]) for F in F_nz  ]
    fvecs_init = zeros(T, length(F_nz_svd)*7);
    F_rvals = copy(rowvals(F_mult))
    A_curr = spzeros(Int, size(F_mult)...)
    for j=1:size(F_mult,2)
        F_col_nz = nzrange(F_mult,j)
        for r in F_col_nz
            i = F_rvals[r]
            A_curr[i,j] = 1
        end
    end
    tracks = correspondences_to_tracks(keypoints, CorresMat, F_mult);
    WeightsGraph = 1 ./ ConstraintGraph;
    filtered_subGraphs = Vector{AbstractSparseMatrix{Int}}(undef, length(tracks))
    
    for (i,track) in enumerate(tracks)
        subG_wts = WeightsGraph[track.image_id, track.image_id];
        filt = preprocess_VG(subG_wts,1);
        filtered_subGraphs[i] = filt; 
        num_remaining = (2*length(track) - 3) - div(nnz(filtered_subGraphs[i]),2)
        n_added = add_edges!(filtered_subGraphs[i], subG_wts, num_remaining)
        filtered_subGraphs[i] = dropzeros!(filtered_subGraphs[i])
        # filtered_subGraphs[i] = A_curr[track.image_id, track.image_id] 
        # filtered_subGraphs[i] += filtered_subGraphs[i]' 
    end
    opt = LeastSquaresOptim.optimize(x->mv_sampson_δ_tracks( x, F_nz_svd, tracks, (F_mult, F_rvals), filtered_subGraphs), fvecs_init, LeastSquaresOptim.LevenbergMarquardt(), iterations=100, autodiff=:forward)

    # f(x) = mv_sampson_δ_tracks(x, F_nz_svd, tracks, (F_mult, F_rvals), filtered_subGraphs)

    # x0 = fvecs_init
    # y0 = f(x0)
    # jac_cfg = JacobianConfig(nothing, y0, x0, Chunk(x0))

    # function g!(J, x)
    #     jacobian!(J, (out, x) -> copyto!(out, f(x)), similar(y0), x, jac_cfg)
    #     if any(isnan, J)
    #         @warn "NaN in J" x findall(isnan, J)
    #     end
    # end

    # prob = LeastSquaresOptim.LeastSquaresProblem(x = x0, f! = (out,x) -> copyto!(out, f(x)),
    #                                               y = y0, g! = g!)
    # opt = LeastSquaresOptim.optimize!(prob, LeastSquaresOptim.LevenbergMarquardt(); show_trace = true, iterations=20)


    println(opt.converged," ", opt.iterations," ", opt.ssr)

    for j=1:size(A_curr,2)
        A_col_nz = nzrange(A_curr,j)
        for r in A_col_nz
            i = F_rvals[r]
            F_mult[i,j] = apply_update(opt.minimizer[ (r-1)*7+1 : r*7], F_nz_svd[r])
            F_mult[j,i] = transpose(F_mult[i,j])
        end
    end
end