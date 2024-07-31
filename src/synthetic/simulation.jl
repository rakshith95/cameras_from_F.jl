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

# function remove_colinear!(triplets::AbstractMatrix{T}, F::AbstractSparseMatrix) where T


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
   # covered = []
    # for j=1:size(triplets,2)
        # if j in covered
            # continue
        # end
        # for k=j+1:size(triplets,2)
            # t1_pairs = Combinatorics.combinations(view(triplets,:,j))
            # t2_pairs = Combinatorics.combinations(view(triplets,:,k))
            # if any(t1_pairs .∈ Ref(t2_pairs))
                # covered = [covered;j;k]
                # break
            # end
        # end
    # end
    # println(setdiff(covered, collect(1:size(triplets,2))))
    # return triplets[:,covered]


end

function noise_cameras( σ::T, Ps::Cameras{T}) where T<:AbstractFloat
    θ = abs(rand(Distributions.Normal(0,σ)))
    return Cameras{T}([ Camera{T}(reshape(projective_synchronization.angular_noise(vec(Ps[i]), θ) , 3, 4)) for i=1:length(Ps) ])
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
    F_noised = FundMat{T}(F_noisy_svd.U*diagm([F_noisy_svd.S[1:2];0.0])*F_noisy_svd.Vt)
    return F_noised
end
    

function  noise_F_angular(σ::T, P₁::Camera{T}, P₂::Camera{T}) where T<:AbstractFloat
    F = F_from_cams(P₁, P₂)
    θ = abs(rand(Distributions.Normal(0,σ)))
    F_noisy = FundMat{T}(reshape(projective_synchronization.angular_noise(vec(F), θ), 3, 3))
    #Rank 2 approximation
    F_noisy_svd = svd(F_noisy)
    F_noisy = FundMat{T}(F_noisy_svd.U*diagm([F_noisy_svd.S[1:2];0.0])*F_noisy_svd.Vt)
    return F_noisy
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

function F_from_cams(P₁::Camera{T}, P₂::Camera{T}) where T
    # This works only if 1st 3x3 block of cameras is non-singular
    # Returns F_21
    Q₁ = @views P₁[1:3,1:3]
    Q₂ = @views P₂[1:3,1:3]
    P₂_svd = svd(P₂, full=true)
    C₂ = P₂_svd.V[:,end]
    e₁ = Pt2D_homo{Float64}(P₁*C₂)
    e₁ₓ = make_skew_symmetric(e₁)
    F = FundMat{T}(inv(Q₂)'*Q₁'*e₁ₓ)
    
    return F
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
    while true
        A = sprand(n,n, ρ)
        A[A.!=0] .= 1.0
        A = sparse(ones(n,n)) - A
        A = triu(A,1) + triu(A,1)'
        G = Graph(A)
        if Graphs.is_connected(G)
            # Check solvability
            C = rand(4,n);
            solvable = MATLAB.mxcall(:is_finite_solvable, 1, Matrix{Float64}(A), C, "rank")
            if solvable
                break
            end
        end
    end
    F_multiview = SparseMatrixCSC{FundMat{Float64}, Int64}(F_multiview .* A)
    # num_UT = (n*(n-1))/2
    num_UT =  length(findall(triu(A,1) .!= 0))
    num_outliers = Int(round(Ρ*num_UT))
    UT_outliers = StatsBase.sample( findall(triu(A,1).!=0), num_outliers, replace=false  )
    for ind in UT_outliers
        F_out = rand(3,3)
        F_out_svd = svd(F_out)
        F_out = F_out_svd.U*diagm([F_out_svd.S[1:end-1];0])*F_out_svd.Vt
        F_multiview[ind] = FundMat{Float64}(F_out)
    end
    if init
        if occursin("gt", lowercase(init_method)) || occursin("ground truth", lowercase(init_method))
            P_init = noise_cameras(camera_noise, gt_cameras)
        elseif occursin("gpsfm", lowercase(init_method)) 
            # recovered_cameras = Cameras{Float64}(MATLAB.mxcall(:runProjectiveSim, 1, F_unwrap, "synch"));
            
            # Get nodes not covered by triplets 
            t = get_triplet_cover(A)
            nonTriplet_cams = setdiff(  collect(1:n), unique(t))
            # Remove these nodes and input to GPSFM
            F_multiview_gpsfm = F_multiview[1:end .∉ Ref(nonTriplet_cams), 1:end .∉ Ref(nonTriplet_cams)]
            F_unwrap = unwrap(F_multiview_gpsfm);

            # file = MAT.matopen("F_test.mat", "w")
            # write(file, "F", F_unwrap)
            # savegraph("problem_graph.lgz", G)
            # close(file)

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
    for method in methods
        if occursin("gpsfm", lowercase(method)) 
            if init && occursin("gpsfm", lowercase(init_method)) 
                errs =  hcat(errs, compute_error(gt_cameras, recovered_cameras_gpsfm, error));
            else
                F_unwrap = unwrap(F_multiview);
                recovered_cameras_gpsfm = Cameras{Float64}(MATLAB.mxcall(:runProjectiveSim, 1, F_unwrap, "gpsfm"));
                errs =  hcat(errs, compute_error(gt_cameras, recovered_cameras_gpsfm, error));
            end
        elseif occursin("skew", lowercase(method))
            recovered_cameras = recover_cameras_iterative(F_multiview; X₀=P_init, method=method, kwargs...);
            errs = hcat(errs, compute_error(gt_cameras, recovered_cameras, error))
        end
    end
    return errs[:,2:end]
end

# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability/Finite_solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM')"

# test_mthds = ["gpsfm", "skew_symmetric_vectorized", "skew_symmetric-l1"]
# test_mthds = ["skew-symmetric"]
# Err = create_synthetic_environment(0.03, test_mthds; holes_density=0.0, update_init="all", initialize=true, init_method="gpsfm", missing_initial=0.0, num_cams=20, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=100);
# rad2deg.(mean.(eachcol(Err)))

# F = create_synthetic_environment(0.1, test_mthds; holes_density=0.65, update_init="all", initialize=false, init_method="gpsfm", missing_initial=0.0, num_cams=20, noise_type="angular", update="random", set_anchor="fixed", max_iterations=1000);

# X₀ = Vector{Camera{Float64}}(repeat([Camera_canonical], 20))
# irls(recover_cameras_iterative, F, X₀, "skew_symmetric_vectorized", compute_error, update_init="all", update="start-centrality-update-all",  set_anchor="fixed")

# A = rand(3,3)
# B = rand(3,3)
# projective_synchronization.angular_distance(SMatrix{3,3,Float64}(A), SMatrix{3,3,Float64}(B))

# IF HD <= 0.5, ALWAYS COVERED BY TRIPLETS?

# G = loadgraph("problem_graph.lgz")
# A = adjacency_matrix(G)

