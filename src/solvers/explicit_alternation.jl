function alternation!(Fₖ::AbstractSparseMatrix{FundMat{T}} , xₙ::Vector{Pts2D{T}} , CMat::AbstractSparseMatrix{ correspondences2D{T1}} , ConstraintGraph::AbstractSparseMatrix{T}, tracks::AbstractVector{track2D}; τ=1e-2, max_its=100, limit_constraints=Inf, normalize_Pts=true, λ_points=ones(length(tracks)), λₑ=0.0) where {T1,T<:AbstractFloat}
    tracks_og = deepcopy(tracks)
    C_rvals = rowvals(CMat)
    C_nzvals = nonzeros(CMat)
    WeightsGraph = ConstraintGraph;
    replace!(WeightsGraph.nzval, Inf => 0.0, -Inf => 0.0)
    dropzeros!(WeightsGraph)

    function update_points()
        # randomly update some tracks to see how it does with fiunding a oslution
        # Translation averaging
        for (i,track) in enumerate(tracks_og)
    #    for (i,track) in enumerate(tracks)
            δ,C1 = mv_sampson_δ_track(track, Fₖ, WeightsGraph)
            # print(norm(C1),"\t")
            # λ = line_search_constraint(unwrap_track(track), δ, track.image_id, Fₖ)
            # λ = line_search_sampson(δ, track, Fₖ, WeightsGraph)
            λ = 1
            # λ = 0.01
            X1 = unwrap_track(track) + λ*δ
            track_update = tracks[i]
            wrap_track!(track_update, X1)
        end
        tracks_to_points!(tracks, xₙ)
        # for (i,track) in enumerate(tracks)
        #     update_points!(track, Fₖ, update_point_projection_average)
        # end
        # tracks_to_points!(tracks, xₙ)
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
                Fₖ[j,i] = Fₖ[j,i]/svdvals(Fₖ[j,i])[1]
                F_svd = svd(Fₖ[j,i])
                opt = LeastSquaresOptim.optimize( x->min_algebraic_err_nonlin(x, F_svd, xᵢ, xⱼ), zeros(7), LeastSquaresOptim.LevenbergMarquardt(), autodiff=:forward, iterations=3)
                Fnew = apply_update(opt.minimizer, F_svd)
                Fₖ[j,i] = Fnew 
                Fₖ[i,j] = Fₖ[j,i]';
            end
        end
    end

    xₖ₋₁ = deepcopy(xₙ)
    # xog = deepcopy(xₙ)
    F_prev = copy(Fₖ)
    diff_vector = zeros(length(xₙ))
    k=1;

    
        S = 0;
        for (i,track) in enumerate(tracks)
            δ,C1 = mv_sampson_δ_track(track, Fₖ, WeightsGraph)
            S += norm(C1)^2
        end
        println(S,"\t")

    while k<=max_its  
        update_points()
        # update_Fs()
        refineF_trackwise!(Fₖ, tracks, ConstraintGraph; max_its=1)

        for i=1:length(xₙ)
            diff_vector[i] = norm(xₖ₋₁[i] - xₙ[i])
        end
        
        if norm(diff_vector) < τ 
            # println("points converged in ",k," iterations")
            break
        end

        if mean(get_error(F_prev, Fₖ )) < 1e-3
            # println("Fs converged in ",k," iterations")
            # break
        end

        xₖ₋₁ = deepcopy(xₙ)
        F_prev = deepcopy(Fₖ)
        k+=1;
    end
 
       S = 0;
        for (i,track) in enumerate(tracks)
            δ,C1 = mv_sampson_δ_track(track, Fₖ, WeightsGraph)
            S += norm(C1)^2
        end
        print(S,"\n")


    println(k)
    return Fₖ
end
