import Base.reverse

function reverse(a::CartesianIndex)
    return CartesianIndex(Base.reverse(a.I))
end

function get_triplets(A::AbstractSparseMatrix) 
    triplets = zeros(Int64, 1,3 )
    n = size(A,1)
    for i=1:n
        rel = findall(x->x>0, view(A,i,i+1:n)) .+ i
        if length(rel) < 2
            continue
        end
        possiblePairs = Combinatorics.combinations(rel, 2)
        for pair in possiblePairs
            if A[pair...] > 0
                triplets = [triplets; [i, pair[1], pair[2]]']
            end
        end
    end
    return triplets[2:end,:]'
end

function get_triplet_cover(A::AbstractSparseMatrix) 
    triplets = get_triplets(A)
    # remove_colinear!(triplets, F;tts_thresh=3e-4)
    ntriplets = size(triplets, 2)
    A = sparse(zeros(ntriplets,ntriplets))
    for i=1:ntriplets
        for j=i+1:ntriplets
            t1_pairs = Combinatorics.combinations(view(triplets,:,i), 2)
            t2_pairs = Combinatorics.combinations(view(triplets,:,j), 2)
            if any(t1_pairs .∈ Ref(t2_pairs))
                A[i,j] = 1
                A[j,i] = 1
            end
        end
    end

    cc = Graphs.connected_components(Graph(A))
    largest_cc = cc[findmax(length.(cc))[2]]
    return triplets[:, largest_cc]
end

function noise_cameras( σ::T, Ps::Cameras{T}) where T<:AbstractFloat
    θ = abs(rand(Distributions.Normal(0,σ)))
    return Cameras{T}([ Camera{T}(reshape(projective_synchronization.rotate_vector(vec(Ps[i]), θ) , 3, 4)) for i=1:length(Ps) ])
end

function noise_F_from_points(σ, P₁::Camera{T}, P₂::Camera{T}, resolution=(1280,720)) where T<:AbstractFloat
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
    return F
end

# noise_F_from_points(1, Camera{Float64}(rand(3,4)), Camera{Float64}(rand(3,4)) );

function noise_F_gaussian(σ, P₁::Camera{T}, P₂::Camera{T}) where T<:AbstractFloat
    F = F_from_cams(P₁, P₂)
    F_noised = F + rand(Distributions.Normal(0,σ), 3, 3) 
    # rank-2 approximation
    F_noisy_svd = svd(F_noised)
    F_noisy = FundMat{T}(F_noisy_svd.U*diagm([F_noisy_svd.S[1:2];0.0])*F_noisy_svd.Vt)
    return F_noisy/norm(F_noisy)
end
    

function  noise_F_angular(σ::T, P₁::Camera{T}, P₂::Camera{T}) where T<:AbstractFloat
    F = F_from_cams(P₁, P₂)
    θ = abs(rand(Distributions.Normal(0,σ)))
    F_noisy = FundMat{T}(reshape(projective_synchronization.rotate_vector(vec(F), θ), 3, 3))
    #Rank 2 approximation
    F_noisy_svd = svd(F_noisy)
    F_noisy = FundMat{T}(F_noisy_svd.U*diagm([F_noisy_svd.S[1:2];0.0])*F_noisy_svd.Vt)
    return F_noisy/norm(F_noisy)
end

function create_cameras!(cameras::Cameras, normalize)
    num_cams = size(cameras,1)
    for i=1:num_cams
        cameras[i] = Camera{Float64}(rand(3,4))
        if normalize
            cameras[i] = cameras[i]/norm(cameras[i],2)
        end
    end
end 

function F_from_cams(Pᵢ::Camera{T}, Pⱼ::Camera{T}) where T
    # This works only if 1st 3x3 block of cameras is non-singular
    # Returns Fⱼᵢ
    Qᵢ = @views Pᵢ[1:3,1:3]
    Qⱼ = @views Pⱼ[1:3,1:3]
    Pⱼ_svd = svd(Pⱼ, full=true)
    Cⱼ = Pⱼ_svd.V[:,end]
    eᵢ = Pt2D_homo{Float64}(Pᵢ*Cⱼ)
    eᵢₓ = make_skew_symmetric(eᵢ)
    Fⱼᵢ = FundMat{T}(inv(Qⱼ)'*Qᵢ'*eᵢₓ)
    return Fⱼᵢ
end

function F_from_cams2(Pᵢ::Camera{T}, Pⱼ::Camera{T}) where T
    Fⱼᵢ = zeros(3,3)
    for i=1:3
        for j=1:3
            Fⱼᵢ[j,i] = ((-1)^(i+j))*det([Pᵢ[1:end .!= i, :]; Pⱼ[1:end .!= j,:]])
        end
    end
    return FundMat{T}(Fⱼᵢ)
end

function compute_error(GT_cameras::Cameras{T}, Recovered_cameras::Cameras{T}, error) where T<:AbstractFloat
    H = relative_projectivity(Recovered_cameras, GT_cameras)
    Ps_transformed = [Recovered_cameras[i]*H for i = 1:length(Recovered_cameras)];

    return [error(vec(Ps_transformed[i]), vec(GT_cameras[i])) for i=1:length(Recovered_cameras) ]
end

function compute_multiviewF_from_cams!(σ, F_multiview::AbstractSparseMatrix, cams::Cameras{T}; noise_type="angular") where T<:AbstractFloat
    n = length(cams)
    for i=1:n
        for j=1:i 
            if i==j
                continue
            end
            if occursin("angular", noise_type)
                F_multiview[i,j] = noise_F_angular(σ, cams[j], cams[i])
            elseif occursin("points", noise_type)
                F_multiview[i,j] = noise_F_from_points(σ, cams[j], cams[i])
            elseif occursin("gaussian", noise_type)
                F_multiview[i,j] = noise_F_gaussian(σ, cams[j], cams[i])
            end
            F_multiview[j,i] = F_multiview[i,j]'
        end
    end
end

function create_synthetic_environment(σ, methods; noise_type="angular", error=projective_synchronization.angular_distance, kwargs...)
    normalize_cameras = get(kwargs, :normalize, true)
    n = get(kwargs, :num_cams, 25)
    ρ = get(kwargs, :holes_density, 0.4)
    Ρ = get(kwargs, :outliers_density, 0.0)
    init = get(kwargs, :initialize, false)
    camera_noise = get(kwargs, :camera_noise, 0.0)
    missing_initial = get(kwargs, :missing_initial, 0.0)
    init_method = get(kwargs, :init_method, "gpsfm")

    gt_cameras = Cameras{Float64}(repeat([Camera(zeros(3,4))], n))
    create_cameras!(gt_cameras, normalize_cameras)
    F_multiview = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],n,n)) # Relative projectivities
    compute_multiviewF_from_cams!(σ, F_multiview, gt_cameras, noise_type=noise_type)
        
    errs = zeros(n, 1)
    A = missing
    G = missing
    nonTriplet_cams = []
    t = missing
    while true
        A = sprand(n,n, ρ)
        A[A.!=0] .= 1.0
        A = sparse(ones(n,n)) - A
        A = triu(A,1) + triu(A,1)'
        G = Graph(A)
        if Graphs.is_connected(G)
            # Check solvability
            # Get nodes not covered by triplets 
            t = get_triplet_cover(A)
            nonTriplet_cams = setdiff(  collect(1:n), unique(t))
            if length(nonTriplet_cams) == 0
                break
            else

            C = rand(4,n);
            solvable = MATLAB.mxcall(:is_finite_solvable, 1, Matrix{Float64}(A), C, "rank")
            if solvable
                break
            end
            end
        end
    end
    F_multiview = SparseMatrixCSC{FundMat{Float64}, Int64}(F_multiview .* A)
    # num_UT = (n*(n-1))/2
    num_UT =  length(findall(triu(A,1) .!= 0))
    num_outliers = Int(round(Ρ*num_UT))
    UT_outliers = missing
    while true
        A′ = copy(A)
        UT_outliers = StatsBase.sample( findall(triu(A,1).!=0), num_outliers, replace=false)
        if length(UT_outliers) == 0
            break
        end
        # println(UT_outliers)
        A′[UT_outliers] .= 0
        A′[reverse.(UT_outliers)] .= 0.0
        C = rand(4,n);
        t2 = get_triplet_cover(A′)
        nonTriplet_cams2 = setdiff(  collect(1:n), unique(t2))
        if length(nonTriplet_cams2) == 0
            break
        end

        solvable = MATLAB.mxcall(:is_finite_solvable, 1, Matrix{Float64}(A′), C, "rank")
        if solvable
            break
        end
    end

    for ind in UT_outliers
        F_out = rand(3,3)
        F_out_svd = svd(F_out)
        F_out = F_out_svd.U*diagm([F_out_svd.S[1:end-1];0])*F_out_svd.Vt
        F_multiview[ind] = FundMat{Float64}(F_out)
        F_multiview[CartesianIndex(reverse(ind.I))] = F_multiview[ind]'
    end
    if init
        if occursin("gt", lowercase(init_method)) || occursin("ground truth", lowercase(init_method))
            P_init = noise_cameras(camera_noise, gt_cameras)
        elseif occursin("gpsfm", lowercase(init_method)) 
            # recovered_cameras = Cameras{Float64}(MATLAB.mxcall(:runProjectiveSim, 1, F_unwrap, "synch"));
            
            # Remove these nodes and input to GPSFM
            
            F_multiview_gpsfm = F_multiview[1:end .∉ Ref(nonTriplet_cams), 1:end .∉ Ref(nonTriplet_cams)]
            F_unwrap = unwrap(F_multiview_gpsfm);


            recovered_cameras_gpsfm = Cameras{Float64}(MATLAB.mxcall(:runProjectiveSim, 1, F_unwrap, "gpsfm"));
            
            P_init = Vector{Camera{Float64}}(repeat([Camera_canonical], n))
            P_init[intersect(collect(1:n), unique(t))] = recovered_cameras_gpsfm            
            
            # recovered_cameras_gpsfm = P_init

        elseif occursin("rand", lowercase(init_method))
            P_init = Vector{Camera{Float64}}(repeat([rand(3,4)], n))
            for i=1:n
                P_init[i] = Camera{Float64}(rand(3,4))
            end
        elseif occursin("spanning", lowercase(init_method)) || occursin("tree", lowercase(init_method))
            P_init = recover_camera_SpanningTree(F_multiview)
        end
        init_missing = StatsBase.sample(collect(1:length(P_init)), Int(round(missing_initial*length(P_init))) )
        for i=1:length(P_init)
            if i in init_missing
                P_init[i] = Camera_canonical
            end
        end
    else
        P_init = nothing
    end
    recovered_cameras = nothing
    Wts = nothing
    for method in methods
        if occursin("gpsfm", lowercase(method)) 
            if init && occursin("gpsfm", lowercase(init_method)) 
                errs =  hcat(errs, compute_error(gt_cameras, recovered_cameras_gpsfm, error));
            else
                F_unwrap = unwrap(F_multiview);
                recovered_cameras_gpsfm = Cameras{Float64}(MATLAB.mxcall(:runProjectiveSim, 1, F_unwrap, "gpsfm"));
                errs =  hcat(errs, compute_error(gt_cameras, recovered_cameras_gpsfm, error));
            end
        else
            if occursin("irls", method)
                recovered_cameras, Wts = outer_irls(recover_cameras_iterative, F_multiview, P_init, method, compute_error, max_iter_init=50, error_measure=projective_synchronization.angular_distance, inner_method_max_it=5, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, max_iterations=50, δ_irls=1.0, update_init="all", update="order-weights-update-all",  set_anchor="fixed");
        else
                recovered_cameras = recover_cameras_iterative(F_multiview; X₀=P_init, method=method, gt=gt_cameras, kwargs...);
            end
            errs = hcat(errs, compute_error(gt_cameras, recovered_cameras, error))
        end
    end
    return errs[:,2:end]
    # return UT_outliers, gt_cameras, P_init, F_multiview, errs[:,2:end]
end

# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability/Finite_solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM')"

# test_mthds = ["gpsfm", "skew_symmetric_vectorized_irls", "subspace_irls"]
# test_mthds = ["gpsfm", "skew_symmetric_vectorized", "subspace", "subspace_angular"]
# test_mthds = ["gpsfm", "skew_symmetric_vectorized", "subspace", "subspace_angular"]
# Err = create_synthetic_environment(0.01, test_mthds; outliers_density=0.0, holes_density=0.0, update_init="all", initialize=true, init_method="gpsfm",  num_cams=25, noise_type="angular", update="order-random-update-all", set_anchor="fixed", max_iterations=100);
# rad2deg.(mean.(eachcol(Err)))

# UT_outliers, P_gt, P_gpsfm, F, Err = create_synthetic_environment(0.02, test_mthds; outliers_density=0.3, holes_density=0.3, update_init="all", initialize=true, init_method="gpsfm",  num_cams=25, noise_type="angular", update="order-weights-update-all", set_anchor="fixed", max_iterations=100);

# recovered_cameras, Wts = outer_irls(recover_cameras_iterative, F, P_gpsfm, "subspace", compute_error, max_iter_init=55, inner_method_max_it=5, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, max_iterations=50, δ_irls=1.0, update_init="all", update="order-weights-update-all",  set_anchor="fixed");
# recovered_cameras = recover_cameras_iterative(F; X₀=P_gpsfm, weights=wts, method="subspace", update="order-weights-update-all", gt=P_gt, max_iterations=50, δ=1e-5);


# n = size(P_gpsfm,1);
# Ẑ = SparseMatrixCSC{eltype(F), Integer}(repeat([zero(eltype(F))], n,n))
# compute_multiviewF_from_cams!(0.0,Ẑ,P_gpsfm)

# wts = compute_weights(F, Ẑ, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy);

# err = compute_error(P_gt, recovered_cameras, projective_synchronization.angular_distance);
# rad2deg.(mean.(eachcol(err)))



# println(CartesianIndex(2,4) in UT_outliers)
# n = size(P_gt,1);
# Ẑ = SparseMatrixCSC{eltype(F), Integer}(repeat([zero(eltype(F))], n,n));
# compute_multiviewF_from_cams!(0.0,Ẑ,P_gpsfm)
# 
# E_UT = projective_synchronization.angular_distance.(UpperTriangular(F),UpperTriangular(Ẑ));
# E = E_UT + E_UT';
# M = ones(Bool, size(F)...);
# M[diagind(M)] .= false;
# s = StatsBase.mad( E[.!isinf.(E) .&& M] )
# wts = projective_synchronization.cauchy.(E/(1*projective_synchronization.c_cauchy*s)) 


# F[1,4]
# Ẑ[1,4]

# n = size(recovered_cameras,1);
# Ẑ = SparseMatrixCSC{eltype(F), Integer}(repeat([zero(eltype(F))], n,n));
# compute_multiviewF_from_cams!(0.0,Ẑ,recovered_cameras)
# E_UT = projective_synchronization.angular_distance.(UpperTriangular(F),UpperTriangular(Ẑ));
# E = E_UT + E_UT';
# M = ones(Bool, size(F)...);
# M[diagind(M)] .= false;
# s = StatsBase.mad( E[.!isinf.(E) .&& M] )
# wts = projective_synchronization.welsch.(E/(1*projective_synchronization.c_cauchy*s)) 







# W,F,E = create_synthetic_environment(0.01, test_mthds; outliers_density=0.2, holes_density=0.0, update_init="all", initialize=true, init_method="gpsfm",  num_cams=15, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=100);
 