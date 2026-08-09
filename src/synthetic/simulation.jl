import Base.reverse

function reverse(a::CartesianIndex)
    return CartesianIndex(Base.reverse(a.I))
end

function is_affine_F(F::SMatrix{3,3,T}, threshold=1e-5) where T<:AbstractFloat
    if all(abs.(F[1:2,1:2]) .<= threshold)
        return true
    else
        return false
    end
end

function check_if_all_nodes_in_triplets(A::AbstractSparseMatrix) 
    n = size(A,1)
    for i=tqdm(1:n)
        rel = findall(x->x>0, view(A,i,1:n)) 
        in_triplet = false
        if length(rel) < 2
            return false
        end
        possiblePairs = Combinatorics.combinations(rel, 2)
        for pair in possiblePairs
            if !iszero(A[pair...])
                in_triplet = true
                break
            end
        end
        if !in_triplet
            return false
        end
    end
    return true
end

function get_triplets(A::AbstractSparseMatrix) 
    L = Graphs.laplacian_matrix(Graph(A))
    n = size(A,1)
    max_trips = Integer(round(factorial(big(n))/(factorial(3)*factorial(big(n-3)))))
    triplets = Vector{Vector{Integer}}(undef, max_trips)
    ct=1
    for i=1:n
        rel = findall(x->x<0, L[i,i+1:n]) .+ i
        if length(rel) < 2
            continue
        end
        possiblePairs = Combinatorics.combinations(rel, 2)
        for pair in possiblePairs
            if !iszero(L[pair...])
                triplets[ct] = [i,pair[1],pair[2]]
                ct += 1 
            end
        end
    end
    return triplets[1:ct-1]
end

function get_triplet_cover(A::AbstractSparseMatrix; max_size=typemax(Int)) 
    triplets = get_triplets(A)
    ntriplets = length(triplets)
    # println(ntriplets)
    if ntriplets > max_size
        triplets = StatsBase.sample(triplets, max_size; replace=false)
        ntriplets = max_size
    end
    A = spzeros(ntriplets,ntriplets)
    for i=1:ntriplets
        for j=i+1:ntriplets
            if length(intersect(triplets[i], triplets[j])) == 2
                A[i,j] = 1
                A[j,i] = 1
            end
        end
    end
    G = Graph(A)
    cc = Graphs.connected_components(G)
    if length(cc) == 0
        return nothing
    end
    largest_cc = cc[findmax(length.(cc))[2]]
    return Graphs.induced_subgraph(G, largest_cc), triplets
end

function noise_cameras( σ::T, Ps::Cameras{T}) where T<:AbstractFloat
    θ = abs(rand(Distributions.Normal(0,σ)))
    return Cameras{T}([ Camera{T}(reshape(projective_synchronization.rotate_vector(vec(Ps[i]), θ) , 3, 4)) for i=1:length(Ps) ])
end

function noise_F_from_points(σ, P₁::Camera{T}, P₂::Camera{T}, resolution=(1280,720); normalize=false) where T<:AbstractFloat
    # Think about how best  to do this
    X = Pts3D{Float64}([rand(3),rand(3),rand(3),rand(3),rand(3),rand(3),rand(3),rand(3)])
    X_homo = homogenize.(X)
    x₁_homo = Pts2D_homo{Float64}([P₁*X_homo[i] for i=1:length(X)]) #+ rand(Distributions.Normal(0, σ), 3) for i=1:length(X)])
    x₂_homo = Pts2D_homo{Float64}([P₂*X_homo[i] for i=1:length(X)]) #+ rand(Distributions.Normal(0, σ), 3) for i=1:length(X)])
    s₁ = resolution[1]/max(maximum(euclideanize.(x₁_homo))...    )
    s₂ = resolution[2]/max(maximum(euclideanize.(x₂_homo))...    )
    x₁_homo = Pts2D_homo{Float64}([s₁*P₁*X_homo[i] + rand(Distributions.Normal(0, σ), 3) for i=1:length(X)])
    x₂_homo = Pts2D_homo{Float64}([s₂*P₂*X_homo[i] + rand(Distributions.Normal(0, σ), 3) for i=1:length(X)])

    F = F_8ptNorm(x₁_homo, x₂_homo)
    if normalize
        return F/norm(F)
    else
        return F
    end
end


function noise_F_gaussian(σ, P₁::Camera{T}, P₂::Camera{T}; F_estimation_fn=F_from_cams, normalize=true) where T<:AbstractFloat
    F = F_estimation_fn(P₁, P₂)
    F_noised = F + rand(Distributions.Normal(0,σ), 3, 3) 
    # rank-2 approximation
    F_noisy_svd = svd(F_noised)
    F_noisy = FundMat{T}(F_noisy_svd.U*diagm([F_noisy_svd.S[1:2];0.0])*F_noisy_svd.Vt)
    if normalize
        return F_noisy/norm(F_noisy)
    else
        return F_noisy
    end
end
    

function  noise_F_angular(σ::T, P₁::Camera{T}, P₂::Camera{T}; F_estimation_fn=F_from_cams, normalize=true) where T<:AbstractFloat
    F = F_estimation_fn(P₁, P₂)
    θ = abs(rand(Distributions.Normal(0,σ)))
    if iszero(θ)
        if normalize
            return F/norm(F)
        else
            return F
        end
    end    
    F_noisy = FundMat{T}(reshape(projective_synchronization.rotate_vector(vec(F), θ), 3, 3))
    #Rank 2 approximation
    F_noisy_svd = svd(F_noisy)
    F_noisy = FundMat{T}(F_noisy_svd.U*diagm([F_noisy_svd.S[1:2];0.0])*F_noisy_svd.Vt)
    if normalize
        return F_noisy/norm(F_noisy)
    else
        return F_noisy
    end
end

function noise_F_angular(σ::T, F::FundMat{T}; normalize=true) where T<:AbstractFloat
    θ = abs(rand(Distributions.Normal(0,σ)))
    if iszero(θ)
        if normalize
            return F/norm(F)
        else
            return F
        end
    end    
    F_noisy = FundMat{T}(reshape(projective_synchronization.rotate_vector(vec(F), θ), 3, 3))
    #Rank 2 approximation
    F_noisy_svd = svd(F_noisy)
    F_noisy = FundMat{T}(F_noisy_svd.U*diagm([F_noisy_svd.S[1:2];0.0])*F_noisy_svd.Vt)
    if normalize
        return F_noisy/norm(F_noisy)
    else
        return F_noisy
    end
end

function create_cameras!(cameras::Cameras; normalize=true, affine=false)
    num_cams = size(cameras,1)
    for i=1:num_cams
        if !affine
            cameras[i] = Camera{Float64}(randn(3,4))
            if normalize
                cameras[i] = cameras[i]/norm(cameras[i],2)
            end
        else
            cameras[i] = AffineCamera(SVector{8,Float64}(randn(8)))
        end
    end
end 

# function F_from_cams(Pᵢ::Camera{T}, Pⱼ::Camera{T}) where T
#     # This works only if 1st 3x3 block of cameras is non-singular
#     # Returns Fⱼᵢ
#     Qᵢ = @views Pᵢ[1:3,1:3]
#     Qⱼ = @views Pⱼ[1:3,1:3]
#     Pⱼ_svd = svd(Pⱼ, full=true)
#     Cⱼ = Pⱼ_svd.V[:,end]
#     eᵢ = Pt2D_homo{Float64}(Pᵢ*Cⱼ)
#     eᵢₓ = make_skew_symmetric(eᵢ)
#     Fⱼᵢ = FundMat{T}(inv(Qⱼ)'*Qᵢ'*eᵢₓ)
#     return Fⱼᵢ
# end

function F_from_cams(Pᵢ::Camera{T}, Pⱼ::Camera{T}) where T<:AbstractFloat
    # Returns Fⱼᵢ
    Fⱼᵢ = zeros(3,3)
    for i=1:3
        for j=1:3
            Fⱼᵢ[j,i] = ((-1)^(i+j))*det([Pᵢ[1:end .!= i, :]; Pⱼ[1:end .!= j,:]])
        end
    end
    return FundMat{T}(Fⱼᵢ)
end

function F_from_cams_gpsfm(Pᵢ::Camera{T}, Pⱼ::Camera{T}) where T<:AbstractFloat
    # Returns Fⱼᵢ
    # NOTE: IMPLEMENTED WITH GPSFM CONVENTION. REFER TO PAPER
    Vᵢ = inv(Pᵢ[1:3,1:3])'
    tᵢ = SVector{3,Float64}(-Vᵢ'*Pᵢ[:,end])

    Vⱼ = inv(Pⱼ[1:3,1:3])'
    tⱼ =  SVector{3,Float64}(-Vⱼ'*Pⱼ[:,end])

    Fᵢⱼ = Vᵢ*(make_skew_symmetric(tᵢ) - make_skew_symmetric(tⱼ))*Vⱼ';
    
    return Fᵢⱼ' # Remeber this returns Fji, which is why we transpose
end
    
function affine_split_error(cam1_vec::SVector{8,T}, cam2_vec::SVector{8,T}, error) where T<:AbstractFloat
    return [error(cam1_vec[1:6], cam2_vec[1:6]), error(cam1_vec[7:8], cam2_vec[7:8]) ]
end

function compute_error(cams1::Cameras{T}, cams2::Cameras{T}, error; affine=false, split=false) where T<:AbstractFloat
    H = relative_projectivity(cams2, cams1; affine=affine)
    Ps_transformed = [cams2[i]*H for i = 1:length(cams2)];
    if affine
        if split
            # display(Recovered_cameras[1])
            # for i=1:length(cams2)
                # println(vec_aff(Ps_transformed[i]),"\t", vec_aff(cams1[i]))
            # end            
            
            return [ affine_split_error( vec_aff(Ps_transformed[i]), vec_aff(cams1[i]), error) for i=1:length(cams2) ]
        else
            return [error(vec_aff(Ps_transformed[i]), vec_aff(cams1[i])) for i=1:length(cams2) ]
        end
    else
        return [error(vec(Ps_transformed[i]), vec(cams1[i])) for i=1:length(cams2) ]
    end
end

function compute_multiviewF_from_cams!(σ, F_multiview::AbstractSparseMatrix, cams::Cameras{T}; F_estimation=F_from_cams, noise_type="angular", normalize=true) where T<:AbstractFloat
    n = length(cams)
    for i=1:n
        for j=1:i 
            if i==j
                continue
            end
            if occursin("angular", noise_type)
                if isequal(cams[j],Camera_canonical) && isequal(cams[i], Camera_canonical)
                    F_multiview[i,j] = FundMat{T}(zeros(3,3))
                else
                    F_multiview[i,j] = noise_F_angular(σ, cams[j], cams[i]; F_estimation_fn=F_estimation, normalize=normalize)
                end
            elseif occursin("points", noise_type)
                F_multiview[i,j] = noise_F_from_points(σ, cams[j], cams[i]; normalize=normalize)
            elseif occursin("gaussian", noise_type)
                F_multiview[i,j] = noise_F_gaussian(σ, cams[j], cams[i]; F_estimation_fn=F_estimation, normalize=normalize)
            end
            F_multiview[j,i] = F_multiview[i,j]'
        end
    end
end

function create_synthetic_environment(σ, methods; affine=false, noise_type="angular", error=projective_synchronization.angular_distance, kwargs...)
    normalize_cameras = get(kwargs, :normalize_cams, true)
    normalize_F = get(kwargs, :normalize_Fs, true)
    split_error = get(kwargs, :split_err, false)
    n = get(kwargs, :num_cams, 25)
    ρ = get(kwargs, :holes_density, 0.0)
    Ρ = get(kwargs, :outliers_density, 0.0)
    init = get(kwargs, :initialize, false)
    missing_initials = get(kwargs, :missing_initial, [0.0])
    init_methods = get(kwargs, :init_methods, ["gpsfm"])

    gt_cameras = Cameras{Float64}(repeat([Camera(zeros(3,4))], n))
    create_cameras!(gt_cameras;normalize = normalize_cameras, affine=affine)
    # TRY WITH 0 BLOCKS FOR GT 
    F_multiview = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],n,n))
    compute_multiviewF_from_cams!(σ, F_multiview, gt_cameras, noise_type=noise_type; normalize=normalize_F)
        
    errs = zeros(n, 1)
    times = zeros(length(methods))
    recovered_cams = Vector{Float64}(undef, length(methods))

    A = missing
    G = missing
    nonTriplet_cams = []
    t = missing
    tG = missing
    trips_time = missing
    while true
        A = sprand(n,n, ρ)
        A[A.!=0] .= 1.0
        A = sparse(ones(n,n)) - A
        A = triu(A,1) + triu(A,1)'
        G = Graph(A)
        if Graphs.is_connected(G)
            if affine
                # println("HERE")
                if solvability_affine(A)
                    break
                else
                    continue
                end
            end
            # Get nodes not covered by triplets 
            trips_time = @elapsed T = get_triplet_cover(A)
            if isnothing(T)
                continue
            else
                tG, t = T
            end
            covered_nodes = unique(reduce(hcat,t[tG[2][1:nv(tG[1])]]))
            nonTriplet_cams = setdiff( collect(1:n), covered_nodes)
            if length(nonTriplet_cams) == 0
                break
            else
            # Check solvability
            # UNCOMMENT THIS FOR NON TRIPLET EXPERIMENTS
            # C = rand(4,n);
            # solvable = MATLAB.mxcall(:is_finite_solvable, 1, Matrix{Float64}(A), C, "eigs")
            # if solvable
                # break
            # end
            end
        end
    end
    recovered_cams_trips = n - length(nonTriplet_cams)
    F_multiview = SparseMatrixCSC{FundMat{Float64}, Int64}(F_multiview .* A)
    UT_outliers = missing
    if Ρ > 0
        num_UT =  length(findall(triu(A,1) .!= 0))
        num_outliers = Int(round(Ρ*num_UT))
        while true
            A′ = copy(A)
            UT_outliers = StatsBase.sample(findall(triu(A,1).!=0), num_outliers, replace=false)
            if length(UT_outliers) == 0
                break
            end
            A′[UT_outliers] .= 0
            A′[reverse.(UT_outliers)] .= 0.0
            if affine
                if solvability_affine(A′)
                    break
                else
                    continue
                end
            end
            tG2 , t2 = get_triplet_cover(A′)
            covered_nodes2 = unique(reduce(hcat,t2[tG2[2][1:nv(tG2[1])]]))
            nonTriplet_cams2 = setdiff( collect(1:n), covered_nodes2)
            if length(nonTriplet_cams2) == 0
                break
            end
            
            C = rand(4,n);
            solvable = MATLAB.mxcall(:is_finite_solvable, 1, Matrix{Float64}(A′), C, "eigs")
            if solvable
                break
            end
        end
        
        for ind in UT_outliers
            if affine
                F_out = SMatrix{3,3,Float64}([[0 0 randn()];[0 0 randn()];randn(1,3)])
            else
                F_out = randn(3,3)
                F_out_svd = svd(F_out)
                F_out = F_out_svd.U*diagm([F_out_svd.S[1:end-1];0])*F_out_svd.Vt
                if normalize_F
                    F_out = F_out/norm(F_out)
                end
            end 
            F_multiview[ind] = FundMat{Float64}(F_out)
            F_multiview[CartesianIndex(reverse(ind.I))] = F_multiview[ind]'
        end
    end

    # if affine
    #     Ps_est_affine = AffineCams_from_F(F_multiview; lad=true);
    #     err = compute_error(gt_cameras, Ps_est_affine, error; affine=affine);
    #     println("FINISHED AFF")
    #     errs =  hcat(errs,err);
    #     return gt_cameras, F_multiview, errs[:,2:end]
        # return UT_outliers, F_multiview, gt_cameras
    # end

    F_multiview_gpsfm = missing
    recovered_cameras_gpsfm = missing
    gpsfm_results = missing
    t₀ = 0
    cam_init_vec = []
    for init_method in init_methods
        P_init = Vector{Camera{Float64}}(repeat([Camera_canonical], n))
        if occursin("gpsfm", lowercase(init_method)) || occursin("sinha", lowercase(init_method)) || occursin("colombo", lowercase(init_method)) 
            # Remove these nodes and input to GPSFM
            F_multiview_gpsfm = F_multiview[1:end .∉ Ref(nonTriplet_cams), 1:end .∉ Ref(nonTriplet_cams)]
            F_unwrap = unwrap(F_multiview_gpsfm);
            if occursin("gpsfm", lowercase(init_method))
                gpsfm_results = MATLAB.mxcall(:runProjectiveSim, 2, F_unwrap, "gpsfm");
                t₀ = gpsfm_results[2]          
                recovered_cameras_gpsfm = Cameras{Float64}(gpsfm_results[1]);
                P_init[intersect(collect(1:n), unique(reduce(hcat,t[tG[2][1:nv(tG[1])]])) )] = recovered_cameras_gpsfm            
            elseif occursin("sinha", lowercase(init_method))
                t₀ = @elapsed recovered_cameras = recover_cameras_baselines(F_multiview_gpsfm, "sinha"; triplet_cover=(tG,t))
                P_init[intersect(collect(1:n), unique(reduce(hcat,t[tG[2][1:nv(tG[1])]])) )] = recovered_cameras 
            else
                t₀ = @elapsed recovered_cameras = recover_cameras_baselines(F_multiview_gpsfm, "colombo"; triplet_cover=(tG,t))
                P_init[intersect(collect(1:n), unique(reduce(hcat,t[tG[2][1:nv(tG[1])]])) )] = recovered_cameras
            end  
        elseif occursin("rand", lowercase(init_method))
            P_init = Vector{Camera{Float64}}(repeat([rand(3,4)], n))
            for i=1:n
                P_init[i] = Camera{Float64}(rand(3,4))
            end
        elseif occursin("spanning", lowercase(init_method)) || occursin("tree", lowercase(init_method))
            P_init = recover_camera_SpanningTree(F_multiview)
        end

        for missing_initial in missing_initials
            P_init_cpy = copy(P_init)
            init_missing = StatsBase.sample(collect(1:length(P_init)), Int(round(missing_initial*length(P_init))), replace=false )
            for i=1:length(P_init)
                if i in init_missing
                    P_init_cpy[i] = Camera_canonical
                end
                cam_init_vec = [cam_init_vec; vec(P_init_cpy[i]/norm(P_init_cpy[i]) )]
            end
            recovered_cameras = nothing
            Wts = nothing
            for (ct,method) in enumerate(methods)
                if occursin("afflin", lowercase(method))
                    # p = [[1;0;0;1;0;0;0;0];4;15;17;18];
                    if occursin("lad", lowercase(method))
                        Ps_est_affine = AffineCams_from_F_vectorized(copy(F_multiview); lad=true);
                    elseif occursin("irls", lowercase(method))
                        if occursin("outer", lowercase(method))
                            Ps_est_affine, wts = lsq_irls((wts)->AffineCams_from_F_vectorized(F_multiview,wts; irls=false), copy(F_multiview); weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy,  max_it=30, δ=deg2rad(1e-3));
                        else
                            wts_window=1;wts_set_last=0;extend_wts=false; regularize=false
                            occursin("filter", lowercase(method))   ? wts_window=4 : missing
                            occursin("set_last", lowercase(method)) ? wts_set_last=12 : missing
                            occursin("only_t", lowercase(method)) ? extend_wts=true : missing 
                            occursin("regularize", lowercase(method)) ? regularize=true : missing

                            Ps_est_affine = AffineCams_from_F_vectorized(copy(F_multiview); irls=true, wts_window=wts_window, wts_set_last=wts_set_last, extend_wts=extend_wts, regularize=regularize);
                        end
                    else
                        Ps_est_affine = AffineCams_from_F_vectorized(copy(F_multiview); );
                    end
                    err = compute_error(gt_cameras, Ps_est_affine, error; split=split_error, affine=affine);
                    # err = compute_error(Ps_est_affine, gt_cameras,  error; split=split_error, affine=affine);
                    errs =  hcat(errs,err);
                elseif occursin("synch", lowercase(method))
                    # synch_results = MATLAB.mxcall(:runProjectiveSim, 2, F_unwrap, "synch")
                    matches = ones(div(size(F_unwrap,1),3), div(size(F_unwrap,1),3))

                    tSynch = @elapsed recovered_cameras_synch, tS = projective_synchronization.matlab_interface(F_unwrap, matches, []; sim=true);
                    # recovered_cameras_synch = Cameras{Float64}(synch_results[1]);
                    recovered_cameras_synch = Cameras{Float64}( [ recovered_cameras_synch[i] for i =1:size(recovered_cameras_synch,2) ]    );
                    err = compute_error(gt_cameras[1:n .∉ Ref(nonTriplet_cams)], recovered_cameras_synch, error);
                    errs =  hcat(errs,[err;ones(length(nonTriplet_cams))*mean(err)] );
                    # times[ct] = synch_results[2]
                    times[ct] = tSynch
                    recovered_cams[ct] = recovered_cams_trips
                elseif occursin("gpsfm", lowercase(method)) 
                    if init && any(occursin.("gpsfm", lowercase.(init_methods)) )
                        err = compute_error(gt_cameras[1:n .∉ Ref(nonTriplet_cams)], recovered_cameras_gpsfm, error);
                        # if iszero(missing_initial)
                            # errs =  hcat(errs,[err;ones(length(nonTriplet_cams))*mean(err)] );
                        # end
                        errs =  hcat(errs,[err;ones(length(nonTriplet_cams))*mean(err)] );
                        # println(rad2deg(mean(err)),"\t", rad2deg(mean(compute_error(recovered_cameras_gpsfm, gt_cameras[1:n .∉ Ref(nonTriplet_cams)], error))))
                        times[ct] = gpsfm_results[2]
                        recovered_cams[ct] = recovered_cams_trips
                    else
                        F_unwrap = unwrap(F_multiview);
                        gpsfm_results = MATLAB.mxcall(:runProjectiveSim, 2, F_unwrap, "gpsfm")
                        recovered_cameras_gpsfm = Cameras{Float64}(gpsfm_results[1]);
                        err = compute_error(gt_cameras[1:n .∉ Ref(nonTriplet_cams)], recovered_cameras_gpsfm, error);
                        errs =  hcat(errs,[err;ones(length(nonTriplet_cams))*mean(err)] );
                        times[ct] = gpsfm_results[2]
                        recovered_cams[ct] = recovered_cams_trips
                    end
                elseif occursin("baseline", lowercase(method))
                    if occursin("colombo", lowercase(method))
                        if length(nonTriplet_cams) < 1
                            t_colombo = @elapsed recovered_cameras_colombo = recover_cameras_baselines(F_multiview_gpsfm, "colombo"; triplet_cover=(tG,t))
                            times[ct] = t_colombo + trips_time
                        else
                            times[ct] = @elapsed recovered_cameras_colombo = recover_cameras_baselines(F_multiview_gpsfm, "colombo")
                        end
                        err = compute_error(gt_cameras[1:n .∉ Ref(nonTriplet_cams)], recovered_cameras_colombo, error);
                        errs =  hcat(errs,[err;ones(length(nonTriplet_cams))*mean(err)] );
                        recovered_cams[ct] = recovered_cams_trips
                    elseif occursin("sinha", lowercase(method))
                        if length(nonTriplet_cams) < 1
                            times[ct] = @elapsed recovered_cameras_sinha = recover_cameras_baselines(F_multiview, "sinha"; triplet_cover=(tG,t))
                            times[ct] += trips_time
                            err = compute_error(gt_cameras, recovered_cameras_sinha, error)
                            errs =  hcat(errs, err);
                            recovered_cams[ct] = n
                        else
                            times[ct] = @elapsed recovered_cameras_sinha, covered_nodes = recover_cameras_baselines_general(F_multiview, "sinha")
                            if count(!iszero,covered_nodes) < 1
                                err = compute_error(gt_cameras, recovered_cameras_sinha, error)
                                errs = hcat(errs, err)
                                recovered_cams[ct] = 0
                            else
                                err = compute_error(gt_cameras[covered_nodes], recovered_cameras_sinha[covered_nodes], error)
                                errs =  hcat(errs, [err;ones(size(errs,1) - length(err))*mean(err)]);
                                recovered_cams[ct] = count(!iszero,covered_nodes)
                            end
                        end
        
                    end
                elseif occursin("global", method)
                    t = @elapsed x_est = MATLAB.mxcall(:global_optimizer, 1, Vector{Float64}(cam_init_vec), unwrap(F_multiview) )
                    est_cameras = Cameras{Float64}(repeat([Camera(zeros(3,4))],n ))
                    for i=1:n
                        est_cameras[i] = reshape(x_est[(i-1)*12+1:i*12], 3,4)
                    end
                    times[ct] = t₀ + t
                    errs = hcat(errs, compute_error(gt_cameras, est_cameras, error))                    
                else
                    if occursin("irls", method)
                        ti = @elapsed recovered_cameras, Wts = outer_irls(recover_cameras_iterative, F_multiview, P_init_cpy, method, compute_error, max_iter_init=50, error_measure=projective_synchronization.angular_distance, inner_method_max_it=5, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, max_iterations=50, δ_irls=1.0, update_init="all", update="order-weights-update-all",  set_anchor="fixed");
                        recovered_cams[ct] = n
                        times[ct] = t₀ + ti
                    else
                        ti = @elapsed recovered_cameras = recover_cameras_iterative(F_multiview; X₀=P_init_cpy, method=method, kwargs...);
                        recovered_cams[ct] = n
                        times[ct] = t₀ + ti
                    end
                    errs = hcat(errs, compute_error(gt_cameras, recovered_cameras, error))
                end
            end
        end
    end
    return errs[:,2:end], F_multiview
    # return errs[:,2:end], F_multiview, gt_cameras
    # return A, F_multiview, gt_cameras, errs[:,2:end]
    # return times
end

# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability/Finite_solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM/3rdparty/fromPPSFM/')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM/3rdparty/vgg_code/')"
 
# test_mthds = ["gpsfm", "baseline sinha", "subspace_angular", "subspace", "skew_symmetric_v4ectorized"]
# test_mthds = ["skew_symmetric_vectorized", "subspace", "subspace-svd", "subspace_angular",] ;
# test_mthds = ["gpsfm", "skew_symmetric_vectorized", "subspace_angular", "l2_kkt" ] ;
# test_mthds = ["gpsfm", "synch", "skew_symmetric_vectorized"]
# test_mthds = ["gpsfm"]
# Err = create_synthetic_environment(0.0, test_mthds;  outliers_density=0.0, holes_density=0.4, update_init="all", initialize=true, init_methods=["gpsfm"], num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);
# Err, F_mult = create_synthetic_environment(0.0, test_mthds;  outliers_density=0.0, holes_density=0.4, update_init="all", initialize=true, init_methods=["gpsfm"], num_cams=10, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);
# println(rad2deg.(mean.(eachcol(Err))))
# println(rad2deg.((Err[:,2])))








# n=4;
# A = sprand(n,n, 0.0);
# A[A.!=0] .= 1.0;
# A = sparse(ones(n,n)) - A;
# A = triu(A,1) + triu(A,1)';

# P1 = Camera{Float64}(rand(3,4));
# P2 = Camera{Float64}(rand(3,4));
# P3 = Camera{Float64}(rand(3,4));
# P4 = Camera{Float64}(rand(3,4));
# gt_cams =Cameras{Float64}([P1,P2,P3,P4]);

# F_mult = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],n,2))
# compute_multiviewF_from_cams!(0.0, F_mult, gt_cams[1:2], noise_type="angular"; normalize=true);
# F_mult = SparseMatrixCSC{FundMat{Float64}, Int64}(F_mult .* A)

# X = Pt3D{Float64}(rand(3));
# X_hom = Pt3D_homo(homogenize(X));

# x1 = euclideanize(P1*X_hom);
# x1_noised = x1 + 1e-1*rand(2);
# x2 = euclideanize(P2*X_hom);
# x2_noised = x2 + 1e-1*rand(2);
# x3 = euclideanize(P3*X_hom);
# x3_noised = x3 + 1e-1*rand(2);
# x4 = euclideanize(P4*X_hom);
# x4_noised = x4 + 1e-1*rand(2);


# track_gt = track2D{point_id2D{Float64}}((point=[x1, x2, x3, x4], image_id=[1,2,3,4]));
# track = track2D{point_id2D{Float64}}((point=[x1_noised, x2_noised], image_id=[1,2]));


# X_og, δ =  mv_sampson_δ(track, F_mult);
# X̂ = X_og + δ;

# norm(unwrap_track(track_gt) - X_og)
# norm(unwrap_track(track_gt) - unwrap_track(track))
# norm(unwrap_track(track_gt) - X̂)

# e_geom = norm(unwrap_track(track_gt) - unwrap_track(track))^2
# e_sampson = dot(δ,δ)



# e_sampson_CF = (homogenize(x2_noised)'*F_mult[2,1]*homogenize(x1_noised))^2 * inv( (F_mult[2,1]*homogenize(x1_noised))[1]^2 + (F_mult[2,1]*homogenize(x1_noised))[2]^2 + (F_mult[1,2]*homogenize(x2_noised))[1]^2 + (F_mult[1,2]*homogenize(x2_noised))[2]^2 )
