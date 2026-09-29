# epipolar_constraint_residual(corr::correspondence2D{T}, F_ji::FundMat{T}) where T<:AbstractFloat = transpose(homogenize(corr[2].point))*F_ji*homogenize(corr[1].point); 
function constraint_residual(keypoints::Vector{Pts2D{T}}, CorresMat::AbstractSparseMatrix{ correspondences2D{T1}} , F_mult::AbstractSparseMatrix{FundMat{T}}; p_norm=1) where {T1,T<:AbstractFloat}
    ConstraintGraph = spzeros(T,size(F_mult,1),size(F_mult,2))

    C_rvals = rowvals(CorresMat)
    Cvals = nonzeros(CorresMat)

    for j=1:size(F_mult,2)
        C_nzCol = nzrange(CorresMat,j)
        for r in C_nzCol
            i = C_rvals[r]
            Corr = Cvals[r]
            if j<=i
                # Iterate only over upper triangle
                continue
            end
            xᵢ = keypoints[i][Corr.point1.keypoint]  
            xⱼ = keypoints[j][Corr.point2.keypoint]

            ConstraintGraph[i,j] = norm( [homogenize(xᵢ[k])'*F_mult[i,j]*homogenize(xⱼ[k]) for k in eachindex(xᵢ)], p_norm ) 
        end
    end
    return ConstraintGraph
end

unwrap_track(track::track2D{point_id2D{T}}) where T<:AbstractFloat = reduce(vcat, track.point) 
function unwrap_track!(X::AbstractVector{T}, track::track2D{point_id2D{T}}) where T<:AbstractFloat
    @assert length(X) == 2*length(track)
    for (i,p) in enumerate(track.point)
        X[(i-1)*2+1] = p[1]
        X[i*2] = p[2]
    end
    return X
end

function wrap_track!(track::track2D{point_id2D{T}}, X::AbstractVector{T}) where T<:AbstractFloat
    for i in eachindex(track)
        track[i] = point_id2D{T}(X[(i-1)*2+1:i*2], track[i].image_id, track[i].keypoint)
    end
end

function correspondences_to_tracks(keypoints::Vector{Pts2D{T}} , CorresMat::AbstractSparseMatrix{ correspondences2D{T1}}, F_mult::AbstractSparseMatrix{FundMat{T}}; min_track_length=2, max_track_length=Inf ) where {T1,T<:AbstractFloat}
    # Add epipolar verification later 
    # Fill this when moving to real data.
    nImg = size(CorresMat, 1)
    L = length.(keypoints)
    Lsum = cumsum(L)

    # build the list of nodes (keypoint) with index
    vals = nonzeros(CorresMat)
    rvals = rowvals(CorresMat)
    # keypts_dict = Bijection{keypoint_id, Integer}()

    function get_point_id(kp::keypoint_id)
        return point_id2D{T}(keypoints[kp.image_id][kp.keypoint], kp.image_id)
    end

    function get_keypoint_index(kp::keypoint_id)
        if kp.image_id==1
            return kp.keypoint
        else
            return Lsum[kp.image_id-1] + kp.keypoint
        end
            
    end

    function get_kp_img_id(node_id::Integer)
        img_id = searchsortedfirst(Lsum, node_id)
        if img_id == 1
            kp = node_id
        else
            kp = node_id - Lsum[img_id-1]
        end  
        return img_id, kp
    end

    X = reduce(vcat, keypoints)
    num_keypts = length(X)
    # Build adjacency matrix
    Adj = spzeros(Bool, num_keypts, num_keypts)
    for j=1:nImg
        rowJ = nzrange(CorresMat, j)
        for r in rowJ
            i = rvals[r]
            if j<=i 
                continue
            end
            C = vals[r]
            for k in 1:length(C)
                point1 = keypoint_id(C.point1[k]...)
                point2 = keypoint_id(C.point2[k]...)
                # Adj[keypts_dict[point1], keypts_dict[point2]] = 1;
                idx1 = get_keypoint_index(point1)
                idx2 = get_keypoint_index(point2)
                Adj[idx1, idx2] = 1;
            end
        end
    end
    Adj .+= transpose(Adj)
    G_pts = SimpleGraph(Adj);
    cc = connected_components(G_pts); # Vector of vector of node indices
    tracks = Vector{track2D}()
    for comp in cc
        Xcomp = Vector{eltype(X)}(undef, length(comp))
        img_ids_comp = Vector{Int}(undef, length(comp))
        kps_comp = Vector{Int}(undef, length(comp))

        if length(comp) < min_track_length || length(comp) > max_track_length
            continue
        end 
        # push!(tracks, track2D( get_point_id.(keypts_dict.(comp)) ))
        for i in eachindex(comp)
            Xcomp[i] = X[comp[i]]
            imgId_kp = get_kp_img_id(comp[i])
            img_ids_comp[i] = imgId_kp[1]
            kps_comp[i] = imgId_kp[2] 
        end
        track = track2D{point_id2D{T}}( ( Xcomp, img_ids_comp, kps_comp ))
        sort!(track, by=x->x.image_id)
        if allunique(track.image_id) # Don't add tracks where there is more than 1 point per image.
            push!(tracks,  track) # Add epipolar verification later
        end
    end
    return tracks
end

function tracks_to_points!(tracks::Vector{track2D}, X::AbstractVector{Pts2D{T}}) where T<:AbstractFloat
    for track in tracks
        for pt_id in track
            # push!(X[pt_id.image_id], pt_id.point)
            X[pt_id.image_id][pt_id.keypoint] = pt_id.point
        end
    end
end

function track_constraint_residual(track::track2D, F_mult::AbstractSparseMatrix{FundMat{T}}) where T<:AbstractFloat
    img_ids = track.image_id
    # Get the subgraph for the track 
    F_sub = F_mult[img_ids, img_ids]
    num_constraints = div(nnz(F_sub),2)
    
    C = zeros(T, num_constraints)
    xᵢ = zeros(T,3)
    xⱼ = zeros(T,3)
    xᵢ[end] = 1
    xⱼ[end] = 1

    constr_ct=1;
    for i=1:size(F_sub,1)-1
        for j=i+1:size(F_sub,2)
            if iszero(F_sub[i,j])
                continue
            end
            Fji = transpose(F_sub[i,j])
            xᵢ[1] = track[i].point[1];
            xᵢ[2] = track[i].point[2];
            
            xⱼ[1] = track[j].point[1];
            xⱼ[2] = track[j].point[2];

            C[constr_ct] = transpose(xⱼ)*Fji*xᵢ
            constr_ct += 1
        end
    end
    return C 
end

import PolynomialRoots

function line_search_constraint(X::AbstractVector{T},  δ::AbstractVector{T}, image_ids::AbstractVector{Int}, F_mult::AbstractSparseMatrix{FundMat{T}}) where T
    F_sub = @views F_mult[image_ids, image_ids]
    pq = zero(T);
    p² = zero(T);
    q² = zero(T);
    s² = zero(T);
    sp = zero(T);
    sq = zero(T);

    xᵢ = MVector{2,T}(zeros(T,2));
    xⱼ = MVector{2,T}(zeros(T,2));
    δᵢ = MVector{2,T}(zeros(T,2));
    δⱼ = MVector{2,T}(zeros(T,2));

    for i=1:size(F_sub,1)-1
        for j=i+1:size(F_sub,2)
            if iszero(F_sub[i,j])
                continue
            end
            xᵢ[1:2] = X[(i-1)*2+1:i*2]
            xⱼ[1:2] = X[(j-1)*2+1:j*2]
            
            δᵢ[1:2] = δ[(i-1)*2+1:i*2]
            δⱼ[1:2] = δ[(j-1)*2+1:j*2]
            
            Fji = transpose(F_sub[i,j])
            A = Fji[1:2,1:2]
            b = Fji[end,1:2]
            c = Fji[1:2,end]
            f33 = Fji[end,end]

            p = transpose(xⱼ)*A*δᵢ + transpose(δⱼ)*A*xᵢ + transpose(δⱼ)*c + transpose(b)*δᵢ    
            q = transpose(xⱼ)*A*xᵢ + transpose(xⱼ)*c + transpose(b)*xᵢ + f33
            s = transpose(δⱼ)*A*δᵢ

            p² += abs2(p)
            q² += abs2(q)
            s² += abs2(s)
            
            pq += p*q
            sp += s*p 
            sq += s*q 
        end
    end

    function constr_res(λᵢ)
        return s²*(λᵢ^4) + p²*(λᵢ^2) + q² + 2*λᵢ*pq + 2*(λᵢ^3)*sp + 2*(λᵢ^2)*sq 
    end

    λ_roots = PolynomialRoots.roots( [pq, 2*sq+p², 3*sp, 2*s²] )
    min_c=Inf; λ_best=missing;
    for λ in λ_roots
        if abs(imag(λ)) > 1e-12
            continue
        end
        λ = real(λ)
        if constr_res(λ) < min_c 
            min_c = constr_res(λ)
            λ_best = λ
        end
    end     
    # λ = -pq/p²
    return λ_best
end
   
function line_search_sampson(δ::AbstractVector{T}, track::track2D, F::AbstractSparseMatrix{FundMat{T}}, Wts::AbstractSparseMatrix{T}) where {T<:AbstractFloat}
    λ₀ = 0.5*ones(T,1)
    track_og = deepcopy(track)
    track_mod = deepcopy(track)
    opt = LeastSquaresOptim.optimize(x->line_search_sampson( x, δ, track_og, track_mod, F, Wts), λ₀, LeastSquaresOptim.LevenbergMarquardt(), iterations=500, autodiff=:central)
    # println(opt.converged," ", opt.iterations)
    return opt.minimizer[1]
end

function line_search_sampson(λ::AbstractVector{T} , δ::AbstractVector{TF}, track_og::track2D, track::track2D, F::AbstractSparseMatrix{FundMat{TF}}, Wts::AbstractSparseMatrix{TF}) where {T,TF<:AbstractFloat}
    X1 = unwrap_track(track_og) + λ[1]*δ
    wrap_track!(track, X1)
    return mv_sampson_δ_track(track, F, Wts) 
end

function algebraic_err(F::SMatrix{3,3,T}, pts1::Pts2D{T2}, pts2::Pts2D{T2}) where {T, T2<:AbstractFloat} 
    e = Vector{T}()
    @assert length(pts1) == length(pts2)
    for i in eachindex(pts1)
        push!(e, transpose(homogenize(pts2[i]))*F*homogenize(pts1[i]))
    end
    return e
end

function min_algebraic_err_nonlin(fvec::AbstractVector{T}, F₀_svd::StaticArrays.SVD{S}, pts1::Pts2D{T2}, pts2::Pts2D{T2}) where {S,T, T2<:AbstractFloat} 
    F_curr = apply_update(fvec, F₀_svd)
    # return two_view_sampson_err(F_curr, pts1, pts2)
    return algebraic_err(F_curr, pts1, pts2)
end