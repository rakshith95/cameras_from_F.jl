function sensitivity(synthetic_env_creator, param_type::String, param_range::Vector{T}, test_methods::Vector{String}, error; σ_fixed=0.0, num_trials=1e3, kwargs...) where T<:AbstractFloat
    E = Vector{Vector{Vector{Float64}}}(undef, length(param_range))
    if occursin("noise", param_type)
        i=1
        for σ=tqdm(param_range)
            Eᵢ = Vector{Vector{Float64}}(undef, num_trials)
            for j=tqdm(1:num_trials)
                try
                    Eⱼ = synthetic_env_creator(σ, test_methods; kwargs...)
                    Eᵢ[j] = mean.(eachcol(Eⱼ))
                catch
                    Eᵢ[j]  = zeros(length(test_methods))
                end
            end
            E[i] = Eᵢ
            i += 1
        end
        return E
    elseif occursin("outlier", param_type)
        i=1
        for Ρ=tqdm(param_range)
            Eᵢ = Vector{Vector{Float64}}(undef, num_trials)
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

function process_data(datasets, method="ours")
    # dataset_paths = Dict( [""]  )
    folder_path = "/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/DataSet Proj/"
    dataset_paths = [folder_path*dataset*".mat" for dataset in datasets]
    errs = zeros(length(datasets))
    times = zeros(length(datasets))
    BA_times = zeros(length(datasets), 2) # for 2 rounds of BA

    for (i,dataset_file) in enumerate(dataset_paths)
        file = MAT.matopen(dataset_file)
        vars = read(file);
        close(file)
        F = vars["FN"]
        tracks = vars["M"]
        matches = vars["pointMatchesInliers"]
        
        if !contains("synch", method)
            t1 = @elapsed recovered_cameras_gpsfm, t, F, N = MATLAB.mxcall(:runProjective_direct, 4, F, "gpsfm", matches, tracks );
        else
            t1 = @elapsed recovered_cameras_synch, t, F, N = MATLAB.mxcall(:runProjective_direct, 4, F, "synch", matches, tracks );
        end

        if contains("ours", method)
            F_mv = wrap(F)
            P_init = Cameras{Float64}([Camera{Float64}(inv(N[3*i-2:3*i, 3*i-2:3*i])*recovered_cameras_gpsfm[i]) for i=1:size(recovered_cameras_gpsfm,1)]);
            t2 = @elapsed Ps, Wts = outer_irls(recover_cameras_iterative, F_mv, P_init, "subspace-angular", compute_error, max_iter_init=15, inner_method_max_it=5, weight_function=projective_synchronization.huber , c=projective_synchronization.c_huber, max_iterations=15, δ=1e-3, δ_irls=1e-1 , update_init="all", update="order-weights-update-all", set_anchor="fixed");
            Ps_mat = [N[3*i-2:3*i, 3*i-2:3*i]*Matrix(Ps[i]) for i=1:length(Ps) ];
            err, BAt1, BAt2 = MATLAB.mxcall(:eval_from_julia, 3, Ps_mat, tracks );
            times[i] = t1 + t2
            BA_times[i,1] = BAt1 
            BA_times[i,2] = BAt2
             
        elseif !contains("synch", method)
            err, BAt1, BAt2 = MATLAB.mxcall(:eval_from_julia, 3, recovered_cameras_gpsfm, tracks );
            times[i] = t1
            BA_times[i,1] = BAt1 
            BA_times[i,2] = BAt2
        
        else
            err, BAt1, BAt2 = MATLAB.mxcall(:eval_from_julia, 3, recovered_cameras_synch, tracks );
            times[i] = t1
            BA_times[i,1] = BAt1 
            BA_times[i,2] = BAt2

        end
        errs[i] = err
    end
    return BA_times, times,errs
end

# datasets = ["Dino 319","Dino 4983","Corridor", "House", "Gustav Vasa", "Folke Filbyter", "Park Gate", "Nijo", "Drinking Fountain", "Golden Statue", "Jonas Ahls", "De Guerre", "Dome", "Alcatraz Courtyard", "Alcatraz Water Tower", "Cherub", "Pumpkin", "Sphinx", "Toronto University", "Sri Thendayuthapani", "Porta san Donato", "Buddah Tooth", "Tsar Nikolai I", "Smolny Cathedral", "Skansen Kronan"];
# datasets = ["Alcatraz Courtyard"]
# BAtimes_gpsfm, times_gpsfm, errors_gpsfm = process_data(datasets, "gpsfm");
# BAtimes_synch, times_synch, errors_synch = process_data(datasets, "synch");
# BAtimes_ours, times_ours, errors_ours = process_data(datasets);
# println(errors_ours)


# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability/Finite_solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM/3rdparty/fromPPSFM/')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM/3rdparty/vgg_code/')"

# test_mthds = ["gpsfm", "gpsfm-synch", "skew_symmetric_vectorized-irls", "subspace-irls", "subspace_angular-irls", "baseline_colombo", "baseline_sinha"]
# init_mthds = ["gpsfm"]
# test_mthds = ["gpsfm", "subspace_angular"];

# E_noise_init_F = sensitivity(create_synthetic_environment, "noise", collect(0.0:0.0075:0.05), test_mthds, projective_synchronization.angular_distance; update_init="all", initialize=true, init_methods=init_mthds, num_trials=50, outliers_density=0.0, holes_density=0.4, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);
# E_missing_init_F = sensitivity(create_synthetic_environment, "noise", collect(0.0:0.0075:0.03), test_mthds, projective_synchronization.angular_distance; missing_initial=collect(0:0.1:0.7),  update_init="all", initialize=true, init_methods=["gpsfm"], num_trials=100, outliers_density=0.0, holes_density=0.4, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=100);

# E_noise_F = sensitivity(create_synthetic_environment, "noise", collect(0.0:0.005:0.03), test_mthds, projective_synchronization.angular_distance; update_init="all", initialize=true, init_method=init_mthds, num_trials=100, holes_density=0.4, num_cams=20, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);
# E_outliers_F = sensitivity(create_synthetic_environment, "outlier", collect(0.0:0.1:0.5), test_mthds, projective_synchronization.angular_distance; update_init="all", initialize=true, init_methods=init_mthds, num_trials=20, holes_density=0.5, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed");
# E_holes_F = sensitivity(create_synthetic_environment, "holes", collect(0.0:0.16:0.8), test_mthds, projective_synchronization.angular_distance; σ_fixed=0.015, update_init="all", initialize=true, init_method="gpsfm", num_trials=100, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed");
# timers = timer_experiment(create_synthetic_environment, collect(10:10:50), test_mthds; σ_fixed=0.01, holes_density=0.4, outliers_density=0.3, update_init="all", initialize=true, init_method="gpsfm", num_trials=10, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);

# e,c = general_graph_experiment(create_synthetic_environment, test_mthds, "noise", collect(0.0:0.005:0.03), projective_synchronization.angular_distance; update_init="all", initialize=true, init_method="gpsfm", num_trials=100, holes_density=0.75, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);
# test_mthds2 = ["subspace", "subspace_angular"]
# e2,c2 = general_graph_experiment(create_synthetic_environment, test_mthds2, "noise", collect(0.0:0.005:0.03), projective_synchronization.angular_distance; update_init="all", initialize=true, init_method="gpsfm", num_trials=100, holes_density=0.75, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);

# for i in 1:length(E_missing_init_F[6])
    # if length(E_missing_init_F[6][i]) < 10
        # E_missing_init_F[6][i] = Inf*ones(10)
    # end
# end

# Errs_matrix = stack(stack.(E_noise_init_F)');
# Errs_matrix = rad2deg.(Errs_matrix);
# Errs_matrix = dropdims(Errs_matrix, dims = tuple(findall(size(Errs_matrix) .== 1)...));;
# file = MAT.matopen("Noise_angleReworked.mat", "w")
# write(file, "E", Errs_matrix)   
# close(file)




# times_matrix = stack(stack.(timers)');
# times_matrix = dropdims(times_matrix, dims = tuple(findall(size(times_matrix) .== 1)...));
# file = MAT.matopen("Times_numFrames_fixed.mat", "w")
# write(file, "times", times_matrix)   
# close(file)