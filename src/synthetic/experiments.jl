function sensitivity(synthetic_env_creator, param_type::String, param_range::Vector{T}, test_methods::Vector{String}, error; σ_fixed=0.0, num_trials=1e3, kwargs...) where T<:AbstractFloat
    split = get(kwargs, :split_err, false)
    if split
        E = Vector{Vector{Vector{ Vector{Float64} }}}(undef, length(param_range))
    else
        E = Vector{Vector{Vector{Float64}}}(undef, length(param_range))
    end
    if occursin("noise", param_type)
        i=1
        for σ=tqdm(param_range)
            if split
                Eᵢ = Vector{Vector{Vector{Float64}}}(undef, num_trials)
            else
                Eᵢ = Vector{Vector{Float64}}(undef, num_trials)
            end
            for j=tqdm(1:num_trials)
                try
                    Eⱼ = synthetic_env_creator(σ, test_methods; kwargs...)
                    Eᵢ[j] = mean.(eachcol(Eⱼ))
                catch
                    if split
                        Eᵢ[j]  = zeros(length(test_methods),2)
                    else
                        Eᵢ[j]  = zeros(length(test_methods))
                    end
                end
            end
            E[i] = Eᵢ
            i += 1
        end
        return E
    elseif occursin("outlier", param_type)
        i=1
        for Ρ=tqdm(param_range)
            if split
                Eᵢ = Vector{Vector{Vector{Float64}}}(undef, num_trials)
            else
                Eᵢ = Vector{Vector{Float64}}(undef, num_trials)
            end
            for j=tqdm(1:num_trials)
                try
                    Eⱼ = synthetic_env_creator(σ_fixed, test_methods; outliers_density=Ρ, kwargs...)
                    Eᵢ[j] = mean.(eachcol(Eⱼ))
                catch
                    Eᵢ[j]  = zeros(length(test_methods))
                end
            end
            E[i] = Eᵢ
            i += 1
        end
        return E
    elseif occursin("holes", param_type)
        i=1
        for ρ=tqdm(param_range)
            Eᵢ = Vector{Vector{Float64}}(undef, num_trials)
            for j=tqdm(1:num_trials)
                try
                    Eⱼ = synthetic_env_creator(σ_fixed, test_methods; holes_density=ρ, kwargs...)
                    Eᵢ[j] = mean.(eachcol(Eⱼ))
                catch
                    Eᵢ[j] = zeros(length(test_methods))
                end
            end
            E[i] = Eᵢ
            i += 1
        end
        return E
    elseif occursin("missing", param_type)
        i = 1
        for missing_init=tqdm(param_range)
            Eᵢ = Vector{Vector{Float64}}(undef, num_trials)
            for j=tqdm(1:num_trials)
                try
                    Eⱼ = synthetic_env_creator(σ_fixed, test_methods; missing_initial=missing_init, kwargs...)
                    Eᵢ[j] = mean.(eachcol(Eⱼ))
                catch
                    Eᵢ[j] = zeros(length(test_methods))
                end
            end
            E[i] = Eᵢ
            i += 1
        end
        return E
    end
end

function general_graph_experiment(synthetic_env_creator, test_methods::Vector{String},param_type::String, param_range::Vector{T}, error; σ_fixed=0.0, num_trials=1e3, kwargs...)  where T<:AbstractFloat
    E = Vector{Vector{Vector{Float64}}}(undef, length(param_range))
    C = Vector{Vector{Vector{Float64}}}(undef, length(param_range))
    if occursin("noise", param_type)
        i=1
        for σ=tqdm(param_range)
            Eᵢ = Vector{Vector{Float64}}(undef, num_trials)
            Cᵢ = Vector{Vector{Float64}}(undef, num_trials)
            for j=tqdm(1:num_trials)
                try
                    Eⱼ, Cⱼ = synthetic_env_creator(σ, test_methods; kwargs...)
                    Eᵢ[j] = mean.(eachcol(Eⱼ))
                    Cᵢ[j] = Cⱼ
                catch
                    Eᵢ[j]  = zeros(length(test_methods))
                    Cᵢ[j] = 0
                end
            end
            E[i] = Eᵢ
            C[i] = Cᵢ
            i += 1
        end
        return E,C
    end
end

function get_curves(mat, xrange)
    curves = Vector{Vector{Float64}}(undef, size(mat,2))

    for (i,col) in enumerate(eachcol(mat))
        c = Vector{Float64}(undef, length(xrange))
        for (j,x) in enumerate(xrange)
            c[j] = (count(col .> x)/length(col))*100
        end
        curves[i] = c
    end
    return curves
end

function timer_experiment(synthetic_env_creator, param_range::Vector{T}, test_methods::Vector{String}; σ_fixed=0.0, num_trials=1e3, kwargs...) where T<:Integer
    t = Vector{Vector{Vector{Float64}}}(undef, length(param_range))
    i=1
    for n=tqdm(param_range)
        tᵢ = Vector{Vector{Float64}}(undef, num_trials)
        for j=tqdm(1:num_trials)
            try
                tⱼ = synthetic_env_creator(σ_fixed, test_methods; num_cams=n, kwargs...)
                tᵢ[j] = tⱼ
            catch
                tᵢ[j] = zeros(length(test_methods))
            end
        end
        t[i] = tᵢ
        i += 1
    end
    return t
end

function get_data(folder_path,dataset)
    dataset_file = folder_path*dataset*".mat" 
    file = MAT.matopen(dataset_file)
    vars = read(file);
    close(file)
    F = vars["FN"]
    tracks = vars["M"]
    matches = vars["pointMatchesInliers"]
    return F,tracks,matches
end

function process_data(datasets, methods=["gpsfm", "synch", "ours"])
    # dataset_paths = Dict( [""]  )
    folder_path = "/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/DataSet Proj/"
    dataset_paths = [folder_path*dataset*".mat" for dataset in datasets]
    errs = zeros(length(datasets), length(methods))
    times = zeros(length(datasets), length(methods))
    BA_times = zeros(length(datasets), 2, length(methods)) # for 2 rounds of BA

    for (i,dataset_file) in enumerate(dataset_paths)
        println(datasets[i])
        file = MAT.matopen(dataset_file)
        vars = read(file);
        close(file)
        F = vars["FN"]
        tracks = vars["M"]
        matches = vars["pointMatchesInliers"]
        
        for (ct,method) in enumerate(methods)
            if !occursin("synch", method)
                recovered_cameras_gpsfm, tGpsfm, F_norm, NormMat = MATLAB.mxcall(:runProjective_direct, 4, F, "gpsfm", matches, tracks );
            else
                # t1 = @elapsed recovered_cameras_synch, t, F, N = MATLAB.mxcall(:runProjective_direct, 4, F, "synch", matches, tracks );
                recovered_cameras_synch, tSynch, F_norm, NormMat = projective_synchronization.matlab_interface(F, matches, tracks;sim=false);
            end
            if occursin("ours", method)
                F_mv = wrap(F_norm)
                # F_mv = wrap(F)
                if occursin("initsynch", method)
                    P_init = Cameras{Float64}([Camera{Float64}(inv(NormMat[3*i-2:3*i, 3*i-2:3*i])*recovered_cameras_synch[1,i]) for i=1:size(recovered_cameras_synch,2)]);
                    # P_init = Cameras{Float64}([Camera{Float64}(recovered_cameras_synch[1,i]) for i=1:size(recovered_cameras_synch,2)]);
                    t_init = tSynch
                else
                    P_init = Cameras{Float64}([Camera{Float64}(inv(NormMat[3*i-2:3*i, 3*i-2:3*i])*recovered_cameras_gpsfm[i]) for i=1:size(recovered_cameras_gpsfm,1)]);
                    # P_init = Cameras{Float64}([Camera{Float64}(recovered_cameras_gpsfm[i]) for i=1:size(recovered_cameras_gpsfm,1)]);
                    t_init = tGpsfm
                end
                tOurs = @elapsed Ps, Wts = outer_irls(recover_cameras_iterative, F_mv, P_init, "subspace_angular", compute_error, max_iter_init=15, inner_method_max_it=5, weight_function=projective_synchronization.huber , c=projective_synchronization.c_huber, max_iterations=15, δ=1e-3, δ_irls=1e-1 , update_init="all", update="order-weights-update-all", set_anchor="fixed");
                Ps_mat = [NormMat[3*i-2:3*i, 3*i-2:3*i]*Matrix(Ps[i]) for i=1:length(Ps) ];
                # Ps_mat = [Matrix(Ps[i]) for i=1:length(Ps) ];
                err, BAt1, BAt2 = MATLAB.mxcall(:eval_from_julia, 3, Ps_mat, tracks );
                times[i,ct] = t_init + tOurs
                BA_times[i,1,ct] = BAt1 
                BA_times[i,2,ct] = BAt2
                
            elseif !contains("synch", method)
                err, BAt1, BAt2 = MATLAB.mxcall(:eval_from_julia, 3, recovered_cameras_gpsfm, tracks );
                times[i,ct] = tGpsfm
                BA_times[i,1,ct] = BAt1 
                BA_times[i,2,ct] = BAt2
            
            else
                err, BAt1, BAt2 = MATLAB.mxcall(:eval_from_julia, 3, recovered_cameras_synch, tracks );
                times[i,ct] = tSynch
                BA_times[i,1,ct] = BAt1 
                BA_times[i,2,ct] = BAt2
    
            end
            errs[i,ct] = err
        end

    end
    return BA_times, times,errs
end

function get_hist_gpsfm_data(datasets, methods=["gpsfm", "synch", "ours"]; gt_folder_path="/home/rakshith/PoliMi/Recovering Cameras/datasets/GPSFM_DATASETS/")
    # dataset_paths = Dict( [""]  )

    gt_dataset_paths = [gt_folder_path*dataset*"/data"*".mat" for dataset in datasets]    
    folder_path = "/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/DataSet Proj/"
    dataset_paths = [folder_path*dataset*".mat" for dataset in datasets]
    Es = Vector{Vector{Float64}}()
    for (i,dataset_file) in enumerate(dataset_paths)
        println(datasets[i])
        # Get GT F
        file = MAT.matopen(gt_dataset_paths[i])
        vars = read(file);
        close(file)
        Ps_gt = vars["P"]
        Ps_gt = Cameras{Float64}([Ps_gt[j] for j=1:size(Ps_gt,2)])
        F_gt = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],length(Ps_gt),length(Ps_gt)))
        compute_multiviewF_from_cams!(0.0, F_gt, Ps_gt, noise_type="angular"; normalize=true)    
    
        # Get gpsfm computed F
        file = MAT.matopen(dataset_file)
        vars = read(file);
        close(file)
        F = vars["FN"]
        F_mv_gpsfm = wrap(F)
        # edge_errors(F_gt, F_mv_gpsfm)
        push!(Es,rad2deg.(edge_errors(F_gt, F_mv_gpsfm)))
    end
    return Es
end


# datasets = ["Dino 319","Dino 4983","Corridor", "House", "Gustav Vasa", "Folke Filbyter", "Park Gate", "Nijo", "Drinking Fountain", "Golden Statue", "Jonas Ahls", "De Guerre", "Dome", "Alcatraz Courtyard", "Alcatraz Water Tower", "Cherub", "Pumpkin", "Sphinx", "Toronto University", "Sri Thendayuthapani", "Porta san Donato", "Buddah Tooth", "Tsar Nikolai I", "Smolny Cathedral", "Skansen Kronan"];
# Er = get_hist_gpsfm_data(datasets);
# maximum(Er[8])


# BAtimes, times, errors_normalized = process_data(datasets, ["gpsfm", "ours"]);
# BAtimes, times, errors_normalized = process_data(datasets, ["gpsfm","ours-initsynch"]);
# println(errors_huber[:,2])
# println(errors_huber[:,2])
# BAtimes, times, errors_NOINIT = process_data2(datasets; norm=false);
# println(errors_NOINIT[:,2])
# times[23:25,:]1
# println(errors)
# err_cauchy = errors;
# println(err_cauchy[:,3])
# println(errors_huber[:,2])

# BAtimes_gpsfm, times_gpsfm, errors_gpsfm = process_data(datasets, "gpsfm");
# BAtimes_synch, times_synch, errors_synch = process_data(datasets, "synch");
# BAtimes_ours, times_ours, errors_ours = process_data(datasets);
# println(errors_ours)


# folder_path = "/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/DataSet Proj/"
# F, tracks, matches = get_data(folder_path, datasets[1]);
# F_mv = wrap(F)

# res = threshold_and_eval(F,tracks,matches, 134;synch=true, norm=false);


# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability/Finite_solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM/3rdparty/fromPPSFM/')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM/3rdparty/vgg_code/')"

# test_mthds = ["gpsfm", "gpsfm-synch", "skew_symmetric_vectorized", "subspace_angular", "baseline_colombo", "baseline_sinha"]
# test_mthds = ["gpsfm", "skew_symmetric_vectorized-irls", "l1", "subspace_angular-irls", "baseline_colombo", "baseline_sinha"]
# init_mthds = ["gpsfm"]
# test_mthds = ["gpsfm", "global", "subspace_angular"];

# E_noise_init_F = sensitivity(create_synthetic_environment, "noise", collect(0.0:0.0075:0.05), test_mthds, projective_synchronization.angular_distance; missing_initial = [0.3], update_init="all", initialize=true, init_methods=init_mthds, num_trials=50, outliers_density=0.0, holes_density=0.4, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);
# E_missing_init_F = sensitivity(create_synthetic_environment,  "noise", collect(0.0:0.0075:0.01), test_mthds, projective_synchronization.angular_distance; missing_initial=collect(0.0:0.1:0.1),  update_init="all", initialize=true, init_methods=["gpsfm"], num_trials=2, outliers_density=0.0, holes_density=0.4, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);

# E_noise_F = sensitivity(create_synthetic_environment, "noise", collect(0.0:0.0075:0.05), test_mthds, projective_synchronization.angular_distance; update_init="all", initialize=true, init_method=init_mthds, num_trials=100, holes_density=0.4, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);
# E_outliers_F = sensitivity(create_synthetic_environment, "outlier", collect(0.0:0.1:0.5), test_mthds, projective_synchronization.angular_distance; update_init="all", initialize=true, init_methods=init_mthds, num_trials=20, holes_density=0.5, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed");
# E_holes_F = sensitivity(create_synthetic_environment, "holes", collect(0.0:0.16:0.8), test_mthds, projective_synchronization.angular_distance; σ_fixed=0.015, update_init="all", initialize=true, init_method="gpsfm", num_trials=20, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed");
# timers = timer_experiment(create_synthetic_environment, collect(10:10:50), test_mthds; σ_fixed=0.015, holes_density=0.4, outliers_density=0.0, update_init="all", initialize=true, init_method="gpsfm", num_trials=10, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);
# e,c = general_graph_experiment(create_synthetic_environment, test_mthds, "noise", collect(0.0:0.005:0.03), projective_synchronization.angular_distance; update_init="all", initialize=true, init_method="gpsfm", num_trials=100, holes_density=0.75, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);
# test_mthds2 = ["subspace", "subspace_angular"]
# e2,c2 = general_graph_experiment(create_synthetic_environment, test_mthds2, "noise", collect(0.0:0.005:0.03), projective_synchronization.angular_distance; update_init="all", initialize=true, init_method="gpsfm", num_trials=100, holes_density=0.75, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);



### Affine
# test_mthds = [ "afflin", "affline_irls-outer", "afflin_irls","afflin_irls-filter", "afflin_irls-filter-set_last", "afflin_irls-filter-set_last-only_t", "afflin_irls-filter-set_last-only_t-regularize"]
# test_mthds = [ "afflin", "afflin_irls-outer", "afflin_irls-filter"]
# E_outliers_F = sensitivity(create_synthetic_environment, "outlier", collect(0.0:0.01:0.07), test_mthds, norm_err; split_err=false, affine=true, initialize=false, init_methods=[""], num_trials=25, holes_density=0.0, num_cams=30, noise_type="angular");
# E_noise_F = sensitivity(create_synthetic_environment, "noise", collect(deg2rad.(0:0.5:8.0)), test_mthds, norm_err; split_err=true, affine=true, initialize=false, init_methods=[""], num_trials=50, holes_density=0.0, num_cams=30, noise_type="angular");

# for i in 1:length(E_missing_init_F[6])
    # if length(E_missing_init_F[6][i]) < 10
        # E_missing_init_F[6][i] = Inf*ones(10)
    # end
# end

# Errs_matrix = stack(stack(stack.(E_outliers_F_split)'));
# Errs_matrix = stack(stack.(E_outliers_F)');
# Errs_matrix = stack(stack(stack.(E_noise_F)')) ;
# Errs_matrix = Errs_matrix[1,:,:,1,:,:]
# Errs_matrix = dropdims(Errs_matrix, dims = tuple(findall(size(Errs_matrix) .== 1)...));;
# file = MAT.matopen("Outliers_gt&cam_rand.mat", "w")
# write(file, "E", Errs_matrix)   

# write(file, "E_R", Errs_matrix[1,:,:,:])   
# write(file, "E_t", Errs_matrix[2,:,:,:])   

# close(file)





# Errs_matrix = stack.(e)';
# Errs_matrix = rad2deg.(Errs_matrix);
# Errs_matrix = dropdims(Errs_matrix, dims = tuple(findall(size(Errs_matrix) .== 1)...));;
# file = MAT.matopen("Noise_with_general.mat", "w")
# write(file, "E", Errs_matrix)   
# close(file)

# times_matrix = stack(stack.(timers)');
# times_matrix = dropdims(times_matrix, dims = tuple(findall(size(times_matrix) .== 1)...));
# file = MAT.matopen("Times_numFrames_fixed_w_Global.mat", "w")
# write(file, "times", times_matrix)   
# close(file)

# Cams_matrix = stack(stack.(c)');
# Cams_matrix = dropdims(Cams_matrix, dims = tuple(findall(size(Cams_matrix) .== 1)...));;
# file = MAT.matopen("CamsRecovered_with_general.mat", "w")
# write(file, "C", Cams_matrix)   
# close(file)







# datasets = ["Dino 319","Dino 4983","Corridor", "House", "Gustav Vasa", "Folke Filbyter", "Park Gate", "Nijo", "Drinking Fountain", "Golden Statue", "Jonas Ahls", "De Guerre", "Dome", "Alcatraz Courtyard", "Alcatraz Water Tower", "Cherub", "Pumpkin", "Sphinx", "Toronto University", "Sri Thendayuthapani", "Porta san Donato", "Buddah Tooth", "Tsar Nikolai I", "Smolny Cathedral", "Skansen Kronan"];
# gt_folder_path="/home/rakshith/PoliMi/Recovering Cameras/datasets/GPSFM_DATASETS/"
# gt_dataset_paths = [gt_folder_path*dataset*"/data"*".mat" for dataset in datasets]    
# folder_path = "/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/DataSet Proj/"
# dataset_paths = [folder_path*dataset*".mat" for dataset in datasets]

# i = 24
# # Get GT F
# file = MAT.matopen(gt_dataset_paths[i])
# vars = read(file);
# close(file)
# Ps_gt = vars["P"]
# Ps_gt = Cameras{Float64}([Ps_gt[j]/norm(Ps_gt[j]) for j=1:size(Ps_gt,2)])
# F_gt = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],length(Ps_gt),length(Ps_gt)))
# compute_multiviewF_from_cams!(0.0, F_gt, Ps_gt, noise_type="angular"; normalize=true)    

# F_gt[rand(1:length(Ps_gt))]

# dropzeros(is_affine_F.(F_gt, 1e-5 ))
# all(is_affine_F.(F_gt, 1e-5))

# Get gpsfm computed F
# file = MAT.matopen(dataset_paths[i])
# vars = read(file);
# close(file)
# F = vars["FN"]
# F_mv_gpsfm = wrap(F)