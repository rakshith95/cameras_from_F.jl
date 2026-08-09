function alternation(CMat::AbstractSparseMatrix{ correspondences2D{T1}} , F_mult::AbstractSparseMatrix{FundMat{T}} , ConstraintGraph::AbstractSparseMatrix{T}, xₙ::Vector{Pts2D{T}} , tracks::AbstractVector{track2D}; τ=1e-2, max_its=100, limit_constraints=Inf, normalize_Pts=true, normalize_F=true, λ_points=1.0, λₑ=0.0) where {T1,T<:AbstractFloat}
    tracks_og = deepcopy(tracks)
    C_rvals = rowvals(CMat)
    C_nzvals = nonzeros(CMat)
    Fₖ = deepcopy(F_mult)
    WeightsGraph = ConstraintGraph;
    replace!(WeightsGraph.nzval, Inf => 0.0, -Inf => 0.0)
    dropzeros!(WeightsGraph)

    function update_points(λₚ=1.0)
        for (i,track) in enumerate(tracks_og)
        # for (i,track) in enumerate(tracks)
            δ = mv_sampson_δ_track(track, Fₖ, WeightsGraph; limit_constraints=2*length(track)-3)
            # δ = mv_sampson_δ(track, Fₖ, WeightsGraph; limit_constraints=100)
            X1 = unwrap_track(track) + λₚ*δ
            track_update = tracks[i]
            wrap_track!(track_update, X1)
        end
        tracks_to_points!(tracks, xₙ)
    end 

    function update_Fs()
        for j=1:size(CMat,2)
            C_nzCol = nzrange(CMat,j)
            for r in C_nzCol
                i = C_rvals[r]
                if j<=i
                    continue
                end
                corres = C_nzvals[r];
                kp_i = corres.point1.keypoint;
                kp_j = corres.point2.keypoint;
                xᵢ = xₙ[i][kp_i]
                xⱼ = xₙ[j][kp_j]
                
                if normalize_Pts
                    Fnew = F_8pt(xᵢ, xⱼ; normalize=normalize_F);
                else
                    Fnew = F_8ptNorm(xᵢ, xⱼ; scale_normalize=normalize_F);
                end
                Fₖ[j,i] = Fline_search(Fₖ[j,i], Fnew, λₑ) # 70% of F1 and 30% of F2
                Fₖ[i,j] = Fₖ[j,i]';
            end
        end
    end

    xₖ₋₁ = deepcopy(xₙ)
    diff_vector = zeros(length(xₙ))
    k=1;

    while k<=max_its  
        # update_points(λ_points)
        update_points(λ_points)
        refineF_pairwise!(Fₖ, xₙ, CMat; max_its=1)
        # update_Fs() # Gradient step with sweeney parameterization on 1:Algebraic error, 2: sampson error

        for i=1:length(xₙ)
            diff_vector[i] = norm(xₖ₋₁[i] - xₙ[i])
        end
        if norm(diff_vector) < τ
            println("points converged in ",k," iterations")
            break
        end
        xₖ₋₁ .= deepcopy(xₙ)
        # copyto!(xₖ₋₁, xₙ) 
        k+=1;
    end

    println(k)
    return Fₖ
end

function Fline_search(F1::FundMat{T}, F2::FundMat{T}, λ=1.0; normalize=true) where T<:AbstractFloat
    if iszero(λ)
        return F1;
    elseif isone(λ)
        return F2;
    end
    f1 = vec(F1)/norm(F1); f2 = vec(F2)/norm(F2);
    f = (1-λ)*f1 + λ*f2
    F = FundMat{T}(reshape(f,(3,3)));
    F_svd = svd(F)
    D = diagm([F_svd.S[1:end-1];0])
    F = FundMat{T}(F_svd.U*D*F_svd.Vt)
    return F/norm(F)
end