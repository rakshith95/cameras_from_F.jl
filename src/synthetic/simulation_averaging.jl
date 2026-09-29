function get_error(F_gt::AbstractSparseMatrix{FundMat{T}}, F::AbstractSparseMatrix{FundMat{T}} ) where T<:AbstractFloat
    E = zeros(T, div(nnz(F),2) )
    ct = 1;
    for i=1:size(F,1)-1
        for j=i+1:size(F,2)
            if iszero(F[i,j])
                continue
            end
            E[ct] = rad2deg( projective_synchronization.angular_distance(F_gt[i,j],  F[i,j] ) )
            ct += 1
        end
    end            
    return E
end

function get_error(F_gts::AbstractSparseMatrix{FundMat{T}}, F_methods::AbstractVector{<:AbstractSparseMatrix{FundMat{T}}}, NormMats::Vector{SMatrix{3,3,T}} ;normalized=true) where T<:AbstractFloat
    E = Vector{Vector{T}}( repeat([[0.0]], length(F_methods)) )

    for i=1:size(F_methods[1],1)-1
        for j=i+1:size(F_methods[1],2)
            if iszero(F_methods[1][i,j])
                continue
            end
            for k in eachindex(F_methods)
                if normalized
                    E[k] = vcat(E[k], rad2deg( projective_synchronization.angular_distance(F_gts[i,j],  NormMats[i]'*F_methods[k][i,j]*NormMats[j] ) )  ) 
                else
                    E[k] = vcat(E[k], rad2deg( projective_synchronization.angular_distance(F_gts[i,j],  F_methods[k][i,j] ) )  ) 
                end
            end
        end
    end            
    for k in eachindex(F_methods)
        E[k] = E[k][2:end]
    end
    return E
end




function sensitivity_epipolar(param_type::String, param_range::Vector{T}, test_methods::Vector{String}; σ_fixed=0.0, num_trials=1e3, kwargs...) where T<:AbstractFloat
    E = Vector{Vector{Vector{T}}}(undef, length(param_range))
    
    if occursin("noise", param_type)
        i=1
        for σ=tqdm(param_range)
            Eᵢ = Vector{Vector{T}}(undef, num_trials)
            for j=tqdm(1:num_trials)
                Eⱼ = create_synthetic_environment_averaging(σ, 0.0; kwargs...)
                Eᵢ[j] = mean.(Eⱼ)
                    # E = create_synthetic_environment_averaging(0.001, deg2rad(0.0); normalize_Pts=true,num_cams=3, num_points=30, holes_density=0.0);
            end
            E[i] = Eᵢ
            i += 1
        end
        return E
    end
end

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
    
    CameraCenters = get_camera_center.(gt_cameras);
    X = randn(3,nPts);
    X_hom = vcat(X, ones(1,nPts));

    for (c,threeD_hom) in enumerate(eachcol(X_hom))
        if any( norm(Ref(euclideanize(threeD_hom)) .- euclideanize.(CameraCenters)) <1e-5)
            X_hom[1:3, c] = rand(3);
        end
    end

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
            nCorres = rand(8: nPts)
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
    trax_alt = deepcopy(trax)
    x_alt = deepcopy(x_euc)
    xnew = deepcopy(x_euc)
    F2v_sampson = deepcopy(F_multiview)
    Fmv_sampson = deepcopy(F_multiview)
    Fmv_alternation = deepcopy(F_multiview)
    
    refineF_pairwise!(F2v_sampson, x_euc, CorresMat)
    
    alternation!(Fmv_alternation, x_alt, CorresMat,  ConstraintGraph, trax_alt; max_its=100, τ=1e-6, λ_points=ones(length(trax)), λₑ=1.0, normalize_Pts=normalize_Pts);
    
    refineF_trackwise!(Fmv_sampson, x_euc, CorresMat, ConstraintGraph; disp=true)

    Es = get_error(F_gt, [F_multiview, F2v_sampson, Fmv_alternation, Fmv_sampson], NormMats; normalized=normalize_Pts)
    # println( get_error(Fmv_sampson, [F2v_sampson, Fmv_alternation], NormMats; normalized=false) )
    return Es, F_multiview, x_euc, x_alt, Fmv_alternation
    # return Es
end

# Compare with GPSFM
# Time complexity investigation of ours vs gpsfm 
# Possibly introduce algebraic consistency during the optimization. 
# Parameterize with cameras, 1) triplets like GPSFM 2) overall 

# E = create_synthetic_environment_averaging(0.005, deg2rad(0.0); normalize_Pts=true, normalize_F_scale=true, num_cams=3, num_points=30, holes_density=0.0);
# mean.(E)

# E, F_in, x_og, x_new, Fmv = create_synthetic_environment_averaging(0.01, deg2rad(0.0); normalize_Pts=true, normalize_F_scale=true, num_cams=3, num_points=28, holes_density=0.0);
# mean.(E)

# # # # Parameterization of points as directions, instead of inhomo points.
# # # Look at adding a ΔF term to the objective. 


# P = Camera{Float64}(rand(3,4));
# @allocations SVector{4,Float64}(nullspace(P))
# @allocations SVector{4,Float64}( [ det(@SMatrix [P[1,2] P[1,3] P[1,4]; P[2,2] P[2,3] P[2,4]; P[3,2] P[3,3] P[3,4]]) , -det(P[:,[1,3,4]]) , det(P[:, [1,2,4]]), -det(P[:,1:end-1]) ] )
# @allocations SVector{4,Float64}( [ det(view(P,:,2:4) )  , -det(view(P,:,[1,3,4])) , det(view(P,:, [1,2,4])), -det(view(P,:,1:3) ) ] )


# minors()



    # gt_cameras = Cameras{Float64}(repeat([Camera(zeros(3,4))], 3));
    # create_cameras!(gt_cameras);
    # F_gt = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],3,3));
    # compute_multiviewF_from_cams!(0.0, F_gt, gt_cameras);

    # e21 = SVector{3,Float64}(nullspace(F_gt[1,2]));
    # e31 = SVector{3,Float64}(nullspace(F_gt[1,3]));
    # e32 = SVector{3,Float64}(nullspace(F_gt[2,3]));

    # P1 = Camera_canonical
    # P2 = Camera([make_skew_symmetric(e21)*F_gt[2,1] e21]);
    
    # P3′ = Camera{Float64}( [make_skew_symmetric(e31)*F_gt[3,1] zeros(3)] )
    # B = kron(pinv(P2)', (make_skew_symmetric(e32)*e31)) 
    # D = ones(9,5);
    # D[:,1:end-1] = -B;
    # D[:,end] = vec(F_gt[3,2])
    # a = vec(make_skew_symmetric(e32)*P3′*pinv(P2) ) 
    # v = D\a
    # # D*v - a

    # P3 = P3′ + e31*v[1:4]'
    # F32 = (1/v[end])*make_skew_symmetric(e32)*P3*pinv(P2);
    
    # F32/norm(F32)
    # F_gt[3,2]/norm(F_gt[3,2])
    

    # rad2deg(mean( compute_error(gt_cameras, Cameras{Float64}([Camera_canonical, P2, P3]), projective_synchronization.angular_distance) ) )

    # F = F_from_cams(P2,P3);
    # F'/norm(F')
    # F32/norm(F32)

    # P2'*F_gt[2,1]*P1 + (P2'*F_gt[2,1]*P1)'
    # P3'*F_gt[3,1]*P1 + (P3'*F_gt[3,1]*P1)'
    # P3'*F_gt[3,2]*P2 + (P3'*F_gt[3,2]*P2)'

    # Ps, Fs = get_cams_from_triplet_sinha( FundMats{Float64}([F_gt[2,1], F_gt[3,1], F_gt[3,2]]) );

    # Ps[3]
    # P3




# K1 = [ rand() 0 rand(); 0 rand() rand(); 0 0 1  ]
# K2 = [ rand() 0 rand(); 0 rand() rand(); 0 0 1  ]

# A = rand(3,3);
# R1 = Matrix(qr(A).Q);
# t1 = rand(3);
# C1 = -R1'*t1;

# B = rand(3,3);
# R2 = Matrix(qr(B).Q);
# t2 = rand(3);
# C2 = -R2'*t2;

# P1 = Camera{Float64}( K1*[R1 t1] );
# P2 = Camera{Float64}( K2*[R2 t2] );

# F1 = F_from_cams(P1,P2);
 
# F2 = inv(K2)' *R2*R1' *make_skew_symmetric( SVector{3,Float64}(t1 - (R1*R2'*t2) ) )  *inv(K1)

# X = rand(4);
# x1 = P1*X;
# x2 = P2*X;

# x2'*F2*x1