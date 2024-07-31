function eccv_sim(σ; noise_type="angular", error=projective_synchronization.angular_distance, kwargs...)
    normalize_cameras = get(kwargs, :normalize, true)
    n = get(kwargs, :num_cams, 20)
    ρ = get(kwargs, :holes_density, 0.0)
    Ρ = get(kwargs, :outliers_density, 0.0)

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
            # solvabile = MATLAB.mxcall(:is_finite_solvable, 1, Matrix{Float64}(A), C, "rank")
            solvable=true
            if solvable
                break
            end
        end
    end
    F_multiview = SparseMatrixCSC{FundMat{Float64}, Int64}(F_multiview .* A)
    num_UT =  length(findall(triu(A,1) .!= 0))
    num_outliers = Int(round(Ρ*num_UT))
    UT_outliers = StatsBase.sample( findall(triu(A,1).!=0), num_outliers, replace=false  )
    for ind in UT_outliers
        F_out = rand(3,3)
        F_out_svd = svd(F_out)
        F_out = F_out_svd.U*diagm([F_out_svd.S[1:end-1];0])*F_out_svd.Vt
        F_multiview[ind] = FundMat{Float64}(F_out)
    end
    
    F_unwrap = unwrap(F_multiview);
    recovered_cameras_gpsfm = Cameras{Float64}(MATLAB.mxcall(:runProjectiveSim, 1, F_unwrap, "gpsfm"));
    errs =  hcat(errs, compute_error(gt_cameras, recovered_cameras_gpsfm, error));

    recovered_cameras = Cameras{Float64}(MATLAB.mxcall(:runProjectiveSim, 1, F_unwrap, "synch"));
    errs = hcat(errs, compute_error(gt_cameras, recovered_cameras, error))

    return errs[:,2:end]#, G
end

# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM')"
# E = eccv_exp(0.2)

function eccv_exp(param_range::Vector{T}; num_trials=1e2, kwargs...) where T<:AbstractFloat
    E = Vector{Vector{Vector{Float64}}}(undef, length(param_range))
    i=1
    for σ=tqdm(param_range)
        Eᵢ = Vector{Vector{Float64}}(undef, num_trials)
        for j=tqdm(1:num_trials)
            Eⱼ = eccv_sim(σ; kwargs...)
            Eᵢ[j] = mean.(eachcol(Eⱼ))
        end
        E[i] = Eᵢ
        i += 1
    end
    return E
end

# E_eccv = eccv_exp(collect(0:0.01:0.05), num_trials=50 );


# Errs_matrix = stack(stack.(E_eccv)');
# Errs_matrix = rad2deg.(Errs_matrix);
# Errs_matrix = dropdims(Errs_matrix, dims = tuple(findall(size(Errs_matrix) .== 1)...));;
# file = MAT.matopen("Noise_E_eccv.mat", "w")
# write(file, "E", Errs_matrix)   
# close(file)
