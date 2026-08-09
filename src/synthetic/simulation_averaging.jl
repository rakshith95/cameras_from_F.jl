
function create_synthetic_environment_averaging(σₓ, σₑ; kwargs...)
    normalize_cameras = get(kwargs, :normalize_cams, true)
    normalize_Pts = get(kwargs, :normalize_Pts, true)
    normalize_F_scale = get(kwargs, :scale_normalize, true)
    nCams = get(kwargs, :num_cams, 25)
    nPts = get(kwargs,:num_points, 100)
    ρ = get(kwargs, :holes_density, 0.0)
    # Ρ = get(kwargs, :outliers_density, 0.0)
    sampson_constraint_limit = get(kwargs, :limit_constraints_sampson, Inf)

    gt_cameras = Cameras{Float64}(repeat([Camera(zeros(3,4))], nCams))
    create_cameras!(gt_cameras;normalize = normalize_cameras)
    F_gt = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],nCams,nCams))
    compute_multiviewF_from_cams!(0.0, F_gt, gt_cameras)

    F_multiview = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],nCams,nCams))
    CorresMat = spzeros(correspondences2D{keypoint_id}, nCams, nCams) # define zero for corres, and proceed with that. 
    NormMats = Vector{SMatrix{3,3,Float64}}(undef, nCams)
    ConstraintGraph = spzeros(Float64,nCams,nCams)

    X = rand(3,nPts);
    X_hom = vcat(X, ones(1,nPts));
    
    x_hom = Vector{Pts2D_homo{Float64}}(undef, nCams)
    x_h = zeros(3, nPts)
    tmp = similar(x_h)

    x_euc = Vector{Pts2D{Float64}}(undef, nCams)
    x_e = zeros(2, nPts)

    for i=1:nCams
        P = gt_cameras[i]#view(gt_cameras, i)        
        mul!(x_h, P, X_hom) # x_h  = P*X_hom
        @views x_e .= (x_h./x_h[end,:]')[1:2,:]; #Euclideanize 
        NormMats[i] = get_normalization_mat(Pts2D{Float64}(eachcol(x_e)))
        if normalize_Pts
            mul!(tmp, NormMats[i], x_h)
            copyto!(x_h, tmp) # x_h = Normalized homogeneous points 
            @views x_e .= (x_h./x_h[end,:]')[1:2,:]; #Euclideanize  pts 
        end
        axpy!(1.0, rand(Distributions.Normal(0.0,σₓ), 2, nPts),  x_e); # Add noise to x_e  
        # re-Homogenize
        copyto!(view(x_h,1:2,:), x_e); 
        fill!(view(x_h, 3,:), 1.0);
        x_hom[i] = Pts2D_homo{Float64}(eachcol(x_h))
        x_euc[i] = Pts2D{Float64}(eachcol(x_e))
    end

    # errs = zeros(n, 1)
    # times = zeros(length(methods))

    A = spzeros(nCams, nCams)
    O = sparse(ones(nCams,nCams))
    G = Graph(A)
    while !Graphs.is_connected(G)
        copyto!(A, sprand(nCams,nCams, ρ) )
        A[A.!=0] .= 1.0
        A .= O - A
        A .= triu(A,1) + triu(A,1)'
        G = Graph(A)
    end
    A_rvals = rowvals(A)
    for j=1:size(A,2)
        A_nzCol = nzrange(A,j)
        for r in A_nzCol
            i = A_rvals[r]
            if j<=i
                # Iterate only over upper triangle
                continue
            end
            nCorres = rand(8:(nPts>20 ? 20 : nPts))
            corres_IDs = StatsBase.sample(1:nPts, nCorres; replace=false)
            xᵢ_hom = x_hom[i][corres_IDs]
            # xᵢ_euc = x_euc[i][corres_IDs]
            xⱼ_hom = x_hom[j][corres_IDs]
            # xⱼ_euc = x_euc[j][corres_IDs]

            pt_id1 = keypoints(( keypoint=corres_IDs, image_id=repeat([i], length(corres_IDs)) ))
            pt_id2 = keypoints(( keypoint=corres_IDs, image_id=repeat([j], length(corres_IDs)) ))
            CorresMat[i,j] = correspondences2D{keypoint_id}( (point1= pt_id1  , point2=pt_id2) )

            if normalize_Pts
                F_multiview[j,i] = noise_F_angular(σₑ, F_8pt(xᵢ_hom, xⱼ_hom); normalize=normalize_F_scale) ; # Returns a normalized F computed with normalized points 
            else
                F_multiview[j,i] = noise_F_angular(σₑ, F_8ptNorm(xᵢ_hom, xⱼ_hom); normalize=normalize_F_scale); # F is computed with normalized points, but the unnormalized F is returned
            end
            F_multiview[i,j] = F_multiview[j,i]'
            ConstraintGraph[i,j] = norm( [xᵢ_hom[k]'*F_multiview[i,j]*xⱼ_hom[k] for k in eachindex(xᵢ_hom)], 1) 
            ConstraintGraph[j,i] = ConstraintGraph[i,j]
        end
    end
    trax = correspondences_to_tracks(x_euc, CorresMat, F_multiview; min_track_length=3);
    x_og = deepcopy(x_euc)
    F2v_sampson = deepcopy(F_multiview)
    Fmv_sampson = deepcopy(F_multiview)
    refineF_pairwise!(F2v_sampson, x_euc, CorresMat)
    
    Fmv_sampson = alternation(CorresMat, F_multiview, ConstraintGraph, x_euc, trax;max_its=250, τ=1e-6, λ_points=0.01, λₑ=1.0, normalize_Pts=normalize_Pts, normalize_F=normalize_F_scale);
    # refineF_trackwise!(Fmv_sampson, x_euc, CorresMat, ConstraintGraph)
    CG2 = constraint_residual(x_euc, CorresMat, Fmv_sampson)
    return NormMats, F_gt, F_multiview, F2v_sampson, Fmv_sampson, CorresMat, ConstraintGraph, CG2, x_og, x_euc

end

# N, F_gt, F_in, F_2S, FmvS, Cmat, CG1, CG2, x_in, x_final = create_synthetic_environment_averaging(0.001, deg2rad(0.0); normalize_Pts=true,num_cams=3, num_points=20, holes_density=0.2);

# e1=[]; e2 = []; e3=[];
# for i=1:size(F_in,1)-1
#     for j=i+1:size(F_in,1)
#         if iszero(F_in[i,j])
#             continue
#         end
#         # push!(e1, rad2deg(projective_synchronization.angular_distance(F_gt[i,j], F_in[i,j] )) );
#         push!(e1, rad2deg(projective_synchronization.angular_distance(F_gt[i,j], N[i]'*F_in[i,j] *N[j] )) );
# #         push!(e2, rad2deg(projective_synchronization.angular_distance(F_gt[i,j], F2[i,j] )) );
#         push!(e2, rad2deg(projective_synchronization.angular_distance(F_gt[i,j], N[i]'*F_2S[i,j]*N[j] )) );
# #         push!(e2, rad2deg(projective_synchronization.angular_distance(F_gt[i,j], F2[i,j] )) );
#         push!(e3, rad2deg(projective_synchronization.angular_distance(F_gt[i,j], N[i]'*FmvS[i,j]*N[j] )) );
#     end
# end
# mean(e1)
# mean(e2)
# mean(e3)

# # any(iszero, e1-e3)

# # # # Parameterization of points as directions, instead of inhomo points.
# # # Look at adding a ΔF term to the objective. 


# # # # # v1 = nonzeros(triu(CG1));
# # # # # # v2 = nonzeros(CG2);
# # # # # # norm(v1,1)
# # # # # norm(v2,1)