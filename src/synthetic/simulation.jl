import Base.reverse

function reverse(a::CartesianIndex)
    return CartesianIndex(Base.reverse(a.I))
end

function get_max_component(G::SimpleGraph, components::AbstractVector)
    max_comp = missing
    max_num = 0 
    for el in unique(components)
        g = findall(components .== el);
        if length(g) > max_num
            max_num = length(g)
            max_comp = g
        end
    end
    return induced_subgraph(G, collect(edges(G))[max_comp])
end

function remove_fraction_edges(M::AbstractSparseMatrix; remove_frac=10, sample=false)
    n = size(M,1)
    A_orig = sparse(ones(n,n))
    A_orig[findall(SparseMatrixCSC{Bool, Integer}(iszero.(M)))] .= 0
    matches_ut = sort([(M[i,j], CartesianIndex(i,j)) for i=1:n for j=i+1:n if M[i,j]>0])
    remove_num = Int(round((remove_frac/100)*length(matches_ut)))
    if sample
        wts = [(1/match[1]) for match in matches_ut]
        remove_inds = StatsBase.sample(collect(1:length(matches_ut)), StatsBase.Weights(wts), remove_num, replace=false )
    else
        remove_inds = 1:remove_num
    end

    for el in matches_ut[remove_inds]
        A_orig[el[2]] = 0 
        A_orig[reverse(el[2])] = 0
    end
    dropzeros!(A_orig)
    return A_orig
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
    largest_cc = cc[findmax(length.(cc))[2]]
    return Graphs.induced_subgraph(G, largest_cc), triplets
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

function F_from_cams(Pᵢ::Camera{T}, Pⱼ::Camera{T}) where T
    # Returns Fⱼᵢ
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
    ρ = get(kwargs, :holes_density, 0.0)
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
    tG = missing
    while true
        A = sprand(n,n, ρ)
        A[A.!=0] .= 1.0
        A = sparse(ones(n,n)) - A
        A = triu(A,1) + triu(A,1)'
        G = Graph(A)
        if Graphs.is_connected(G)
            # Get nodes not covered by triplets 
            tG, t = get_triplet_cover(A)
            covered_nodes = unique(reduce(hcat,t))
            nonTriplet_cams = setdiff( collect(1:n), covered_nodes)
            if length(nonTriplet_cams) == 0
                break
            else
                # Check solvability
                # UNCOMMENT THIS FOR NON TRIPLET EXPERIMENTS
            # C = rand(4,n);
            # solvable = MATLAB.mxcall(:is_finite_solvable, 1, Matrix{Float64}(A), C, "rank")
            # if solvable
                # break
            # end
            end
        end
    end
    F_multiview = SparseMatrixCSC{FundMat{Float64}, Int64}(F_multiview .* A)
    if Ρ > 0
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
            tG2, t2 = get_triplet_cover(A′)
            n2 = unique(reduce(hcat,t2))

            nonTriplet_cams2 = setdiff(  collect(1:n), n2)
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
    end
    
    if init
        if occursin("gt", lowercase(init_method)) || occursin("ground truth", lowercase(init_method))
            P_init = noise_cameras(camera_noise, gt_cameras)
        elseif occursin("gpsfm", lowercase(init_method)) 
            # Remove these nodes and input to GPSFM
            F_multiview_gpsfm = F_multiview[1:end .∉ Ref(nonTriplet_cams), 1:end .∉ Ref(nonTriplet_cams)]
            F_unwrap = unwrap(F_multiview_gpsfm);
            recovered_cameras_gpsfm = Cameras{Float64}(MATLAB.mxcall(:runProjectiveSim, 1, F_unwrap, "gpsfm"));
            
            P_init = Vector{Camera{Float64}}(repeat([Camera_canonical], n))
            P_init[intersect(collect(1:n), unique(reduce(hcat,t)))] = recovered_cameras_gpsfm            
            
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
        if occursin("synch", lowercase(method))
            recovered_cameras_synch = Cameras{Float64}(MATLAB.mxcall(:runProjectiveSim, 1, F_unwrap, "synch"));
            errs =  hcat(errs, compute_error(gt_cameras, recovered_cameras_synch, error));
        elseif occursin("gpsfm", lowercase(method)) 
            if init && occursin("gpsfm", lowercase(init_method)) 
                errs =  hcat(errs, compute_error(gt_cameras, recovered_cameras_gpsfm, error));
            else
                F_unwrap = unwrap(F_multiview);
                recovered_cameras_gpsfm = Cameras{Float64}(MATLAB.mxcall(:runProjectiveSim, 1, F_unwrap, "gpsfm"));
                errs =  hcat(errs, compute_error(gt_cameras, recovered_cameras_gpsfm, error));
            end
        elseif occursin("baseline", lowercase(method))
            if occursin("colombo", lowercase(method))
                recovered_cameras_colombo = recover_cameras_baselines(F_multiview, "colombo"; triplet_cover = (tG, t))
                errs =  hcat(errs, compute_error(gt_cameras, recovered_cameras_colombo, error));
            elseif occursin("sinha", lowercase(method))
                recovered_cameras_sinha = recover_cameras_baselines(F_multiview, "sinha"; triplet_cover = (tG, t))
                errs =  hcat(errs, compute_error(gt_cameras, recovered_cameras_sinha, error));
            end
        else
            if occursin("irls", method)
                recovered_cameras, Wts = outer_irls(recover_cameras_iterative, F_multiview, P_init, method, compute_error, max_iter_init=50, error_measure=projective_synchronization.angular_distance, inner_method_max_it=5, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, max_iterations=50, δ_irls=1.0, update_init="all", update="order-weights-update-all",  set_anchor="fixed");
            else
                recovered_cameras = recover_cameras_iterative(F_multiview; X₀=P_init, method=method, kwargs...);
            end
            errs = hcat(errs, compute_error(gt_cameras, recovered_cameras, error))
        end
    end
    return errs[:,2:end]
    # return gt_cameras, F_multiview, errs[:,2:end]
end

# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability/Finite_solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM')"
 
# test_mthds = ["gpsfm", "synch"];
# test_mthds = ["gpsfm", "skew_symmetric_vectorized", "subspace", "subspace_angular"]
# test_mthds = ["gpsfm", "skew_symmetric_vectorized",  "subspace_angular", "baseline colombo", "baseline sinha"]
# Err = create_synthetic_environment(0.01, test_mthds; outliers_density=0.0, holes_density=0.2, update_init="all", initialize=false, init_method="gpsfm",  num_cams=10, noise_type="angular", update="order-random-update-all", set_anchor="fixed", max_iterations=100);
# rad2deg.(mean.(eachcol(Err)))



# folder_name = "MatFiles/nerf_bonsai"

# file = MAT.matopen(folder_name*"/Adj.mat");
# Adj = sparse(read(file, "A"))
# close(file)
# num_v = nv(Graph(Adj))
# HD = 1 - ne(Graph(Adj))/(num_v*(num_v-1)/2)

# match_file = MAT.matopen(folder_name*"/Matches.mat");
# Matches = SparseMatrixCSC{Float64, Int64}(read(match_file, "M"));
# close(match_file)

# F_file = MAT.matopen(folder_name*"/Fs.mat");
# F = read(F_file, "FN");
# close(F_file)
# F_multiview = cameras_from_F.wrap(F)
 
# Ps_file = MAT.matopen(folder_name*"/Ps.mat");
# Ps_gt = read(Ps_file);
# close(Ps_file)
# Ps_gt = sort(Dict(parse(Int,string(k))=>v  for (k,v) in pairs(Ps_gt)));
# Ps_gt = Cameras{Float64}(getindex.(Ref(Ps_gt), keys(Ps_gt)));


# Ps_gpsfm_file = MAT.matopen(folder_name*"/Ps_gpsfm.mat");
# Ps_gpsfm = read(Ps_gpsfm_file, "Ps_gpsfm");
# close(Ps_gpsfm_file)
# recovered_cameras_gpsfm = Cameras{Float64}([Camera{Float64}(Ps_gpsfm[i]) for i=1:size(Ps_gpsfm,1)]);

# F_unwrap = unwrap(F_multiview_new);
# recovered_cameras_gpsfm = Cameras{Float64}(MATLAB.mxcall(:runProjectiveSim, 0, F_unwrap, "gpsfm", Matches));
# mean(rad2deg.(compute_error(Ps_gt, recovered_cameras_gpsfm, projective_synchronization.angular_distance)))
# recovered_cameras = recover_cameras_iterative(F_multiview; X₀=recovered_cameras_gpsfm, method="subspace_angular",  update="order-centrality-update-all");
# recovered_cameras, Wts = outer_irls(recover_cameras_iterative, F_multiview, recovered_cameras_gpsfm, "subspace_angular", compute_error, max_iter_init=15, inner_method_max_it=5, weight_function=projective_synchronization.cauchy , c=projective_synchronization.c_cauchy, max_iterations=15, δ=1e-3, δ_irls=1e-1 , update_init="all", update="order-weights-update-all", set_anchor="fixed");
# mean(rad2deg.(compute_error(Ps_gt, recovered_cameras, projective_synchronization.angular_distance)))


# if !check_if_all_nodes_in_triplets(Adj)
    # any(degree(Graph(Adj)) .< 2)
# end


# Adj_new = remove_fraction_edges(Matches; remove_frac=85, sample=false)
# check_if_all_nodes_in_triplets(Adj_new)
# tg, trips = get_triplet_cover(Adj_new; max_size=5000);
# covered_nodes = unique(reduce(hcat,trips))
# nonTriplet_cams = setdiff( collect(1:size(Adj,1)), covered_nodes)



# issymmetric(Adj_new)
# is_connected(Graph(Adj_new))
# any(degree(Graph(Adj_new)) .< 2)

# C = rand(4,size(Adj_new,1))*100;
# solvable = MATLAB.mxcall(:is_finite_solvable, 1, Matrix{Float64}(Adj_new), C, "eigs")

# F_multiview_new = SparseMatrixCSC{FundMat{Float64}, Int64}(F_multiview .* Adj_new)


# recovered_cameras = recover_cameras_iterative(F_multiview_new; X₀=P_init, method="subspace_angular");
# recovered_cameras, Wts = outer_irls(recover_cameras_iterative, F_multiview_new, P_init, "subspace_angular", compute_error, max_iter_init=50, error_measure=projective_synchronization.angular_distance, inner_method_max_it=5, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, max_iterations=50, δ_irls=1.0, update_init="all", update="order-weights-update-all",  set_anchor="fixed");



# adj_mod_file = MAT.matopen(folder_name*"/Pruned_Adj.mat", "w")
# write(adj_mod_file, "A", Adj_new)
# close(adj_mod_file) 

# F_unwrap = unwrap(F_multiview_new);
# recovered_cameras_gpsfm = Cameras{Float64}(MATLAB.mxcall(:runProjectiveSim, 0, F_unwrap, "gpsfm", Matches));
# mean(rad2deg.(compute_error(Ps_gt, recovered_cameras_gpsfm, projective_synchronization.angular_distance)))
# recovered_cameras = recover_cameras_iterative(F_multiview; X₀=recovered_cameras_gpsfm, method="subspace_angular");
# recovered_cameras, Wts = outer_irls(recover_cameras_iterative, F_multiview, recovered_cameras_gpsfm, "subspace_angular", compute_error, max_iter_init=15, inner_method_max_it=5, weight_function=projective_synchronization.huber , c=projective_synchronization.c_huber, max_iterations=15, δ=1e-3, δ_irls=1e-1 , update_init="all", update="order-weights-update-all", set_anchor="fixed");
# mean(rad2deg.(compute_error(Ps_gt, recovered_cameras, projective_synchronization.angular_distance)))


# f_file = MAT.matopen(folder_name*"/F_unwrapped.mat", "w")
# write(f_file, "F", F_unwrap)
# close(f_file) 


# recovered_cameras_baseline_sinha = recover_cameras_baselines(F_multiview, "sinha");
# mean(rad2deg.(compute_error(Ps_gt, recovered_cameras_baseline_sinha, projective_synchronization.angular_distance)))
# recovered_cameras_baseline_colombo = recover_cameras_baselines(F_multiview_new, "colombo");
# mean(rad2deg.(compute_error(Ps_gt, recovered_cameras_baseline_colombo, projective_synchronization.angular_distance)))
