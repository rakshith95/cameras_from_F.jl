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

function remove_fraction_edges(M::AbstractSparseMatrix; remove_frac=10, sample=false, threshold=0.0)
    n = size(M,1)
    A_orig = sparse(ones(n,n))
    A_orig[findall(SparseMatrixCSC{Bool, Integer}(iszero.(M)))] .= 0
    matches_ut = sort([(M[i,j], CartesianIndex(i,j)) for i=1:n for j=i+1:n if M[i,j]>0])
    scores_ut = [m[1] for m in matches_ut]
    remove_num = Int(round((remove_frac/100)*length(matches_ut)))
    if sample
        wts = [(1/match[1]) for match in matches_ut]
        remove_inds = StatsBase.sample(collect(1:length(matches_ut)), StatsBase.Weights(wts), remove_num, replace=false )
    else
        if threshold > 0
            remove_inds = findall(scores_ut .< threshold)
        else
            remove_inds = 1:remove_num
        end
            
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


function noise_F_gaussian(σ, P₁::Camera{T}, P₂::Camera{T}; normalize=true) where T<:AbstractFloat
    F = F_from_cams(P₁, P₂)
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
    

function  noise_F_angular(σ::T, P₁::Camera{T}, P₂::Camera{T}; normalize=true) where T<:AbstractFloat
    F = F_from_cams(P₁, P₂)
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

function compute_multiviewF_from_cams!(σ, F_multiview::AbstractSparseMatrix, cams::Cameras{T}; noise_type="angular", normalize=true) where T<:AbstractFloat
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
                    F_multiview[i,j] = noise_F_angular(σ, cams[j], cams[i]; normalize=normalize)
                end
            elseif occursin("points", noise_type)
                F_multiview[i,j] = noise_F_from_points(σ, cams[j], cams[i]; normalize=normalize)
            elseif occursin("gaussian", noise_type)
                F_multiview[i,j] = noise_F_gaussian(σ, cams[j], cams[i]; normalize=normalize)
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
    missing_initials = get(kwargs, :missing_initial, [0.0])
    init_methods = get(kwargs, :init_methods, ["gpsfm"])

    gt_cameras = Cameras{Float64}(repeat([Camera(zeros(3,4))], n))
    create_cameras!(gt_cameras, normalize_cameras)
    F_multiview = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],n,n)) # Relative projectivities
    compute_multiviewF_from_cams!(σ, F_multiview, gt_cameras, noise_type=noise_type)
        
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
            A′[UT_outliers] .= 0
            A′[reverse.(UT_outliers)] .= 0.0
            
            tG2 , t2 = get_triplet_cover(A)
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
            F_out = rand(3,3)
            F_out_svd = svd(F_out)
            F_out = F_out_svd.U*diagm([F_out_svd.S[1:end-1];0])*F_out_svd.Vt
            F_multiview[ind] = FundMat{Float64}(F_out)
            F_multiview[CartesianIndex(reverse(ind.I))] = F_multiview[ind]'
        end
    end
    F_multiview_gpsfm = missing
    recovered_cameras_gpsfm = missing
    gpsfm_results = missing
    t₀ = 0
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
            end
            recovered_cameras = nothing
            Wts = nothing
            for (ct,method) in enumerate(methods)
                if occursin("synch", lowercase(method))
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
                        if iszero(missing_initial)
                            errs =  hcat(errs,[err;ones(length(nonTriplet_cams))*mean(err)] );
                        end
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

    return errs[:,2:end]
    # return errs[:,2:end], recovered_cams
    # return times
end

# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability/Finite_solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM/3rdparty/fromPPSFM/')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM/3rdparty/vgg_code/')"
 
# test_mthds = ["gpsfm", "baseline sinha", "subspace_angular", "subspace", "skew_symmetric_vectorized"]
# test_mthds = ["skew_symmetric_vectorized", "subspace", "subspace-svd", "subspace_angular",] ;
# test_mthds = ["gpsfm", "skew_symmetric_vectorized", "subspace_angular", "l2_kkt" ] ;

# test_mthds = ["skew_symmetric_vectorized"]
# Err = create_synthetic_environment(0.0, test_mthds; outliers_density=0.0, holes_density=0.4, update_init="none", initialize=false, init_methods=[""], num_cams=25, noise_type="angular", update="order-random-update-all", set_anchor="fixed", max_iterations=20);
# println(rad2deg.(mean.(eachcol(Err))))

# gt_cameras = Cameras{Float64}(repeat([Camera(zeros(3,4))], 4));
# create_cameras!(gt_cameras, true);
# F_multiview = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],4,4)) ;
# compute_multiviewF_from_cams!(0.0, F_multiview, gt_cameras; normalize=false);
# F = unwrap(F_multiview);


# F_multiview2 = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],4,4)) ;
# compute_multiviewF_from_cams!(0.0, F_multiview2, gt_cameras; normalize=true);
# F′ = unwrap(F_multiview2);
# rank(F′)
# F_multiview2[2,3] = 0.02368376472974777*F_multiview2[2,3];
# F_multiview2[3,2] = 0.02368376472974777*F_multiview2[3,2];
# norm.(F_multiview)./norm.(F_multiview2)
# F′ == F

# # gt_cameras = Cameras{Float64}(repeat([Camera(zeros(3,4))], 5));
# create_cameras!(gt_cameras, true);
# F_multiview = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],6,6)) ;
# # F_multiview = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],5,5)) ;
# compute_multiviewF_from_cams!(0.0, F_multiview, gt_cameras, noise_type="angular");




# Fs_file = MAT.matopen("MatFiles/tmp_new/street_database/Fs.mat")
# F_vars = MAT.read(Fs_file);
# close(Fs_file)
# F = F_vars["FN"]
# F_mv = wrap(F)

# Ps_gt_file = MAT.matopen("MatFiles/tmp_new/street_database/Ps.mat")
# Ps_gt_vars = MAT.read(Ps_gt_file);
# close(Ps_gt_file)
# Ps_gt = Ps_gt_vars
# Ps_gt = Cameras{Float64}([Ps_gt[string(i)] for i=1:size(F_mv,2)]);

# Ps_gt[2]'F_mv[2,15]*Ps_gt[15]

# Matches_file = MAT.matopen("MatFiles/tmp_new/street_database/Matches.mat")
# M_vars = MAT.read(Matches_file);
# close(Matches_file)
# M = Matrix{Float64}(M_vars["M"])

# Ns_file = MAT.matopen("MatFiles/tmp_new/street_database/Ns.mat")
# N_vars = MAT.read(Ns_file);
# close(Ns_file)
# N = N_vars["NormMat"]

# Ps_gpsfm, t, FN_norm, N = MATLAB.mxcall(:runProjective_direct, 4, F, "gpsfm", M, [], N );
# Ps_gpsfm = Cameras{Float64}(Ps_gpsfm);
# F_mv_norm = wrap(FN_norm)

# Ps_init = Cameras{Float64}([Camera{Float64}(inv(N[3*i-2:3*i, 3*i-2:3*i])*Ps_gpsfm[i]) for i=1:size(Ps_gpsfm,1)]);
# Ps, Wts = outer_irls(recover_cameras_iterative, F_mv, Ps_gt, "subspace-angular", compute_error, max_iter_init=15, inner_method_max_it=5, weight_function=projective_synchronization.huber , c=projective_synchronization.c_huber, max_iterations=15, δ=1e-3, δ_irls=1e-1 , update_init="all", update="order-weights-update-all", set_anchor="fixed");
# Ps = [N[3*i-2:3*i, 3*i-2:3*i]*Matrix(Ps[i]) for i=1:length(Ps) ];

# err = compute_error(Cameras{Float64}(Ps),Cameras{Float64}(Ps_gt), projective_synchronization.angular_distance);
# println(mean(rad2deg.(err)))


