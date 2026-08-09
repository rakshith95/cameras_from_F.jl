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
    # keypt_ct = 1;
    # for j=1:nImg
    #     rowJ = nzrange(CorresMat, j)
    #     for r in rowJ
    #         i = rvals[r]                
    #         if j<=i 
    #             continue
    #         end
    #         C = vals[r]
    #         for k in 1:length(C)
    #             point1 = keypoint_id(C.point1[k]...)
    #             if !haskey(keypts_dict, point1)
    #                 keypts_dict[point1] = keypt_ct;
    #                 # push!(kpts, point1)
    #                 keypt_ct += 1;
    #             end
                
    #             point2 = keypoint_id(C.point2[k]...)
    #             if !haskey(keypts_dict, point2)
    #                 keypts_dict[point2] = keypt_ct;
    #                 # push!(kpts, point1)
    #                 keypt_ct += 1;
    #             end
    #         end
    #     end
    # end
    # num_keypts = length(keys(keypts_dict))
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
