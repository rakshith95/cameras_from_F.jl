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
            if !contains("synch", method)
                recovered_cameras_gpsfm, tGpsfm, F_norm, NormMat = MATLAB.mxcall(:runProjective_direct, 4, F, "gpsfm", matches, tracks );
            else
                # t1 = @elapsed recovered_cameras_synch, t, F, N = MATLAB.mxcall(:runProjective_direct, 4, F, "synch", matches, tracks );
                recovered_cameras_synch, tSynch = projective_synchronization.matlab_interface(F, matches, tracks;sim=false);
            end
    
            if contains("ours", method)
                F_mv = wrap(F_norm)
                # F_opt = NormMat'*F_opt*NormMat
                # F_mv = wrap(F_opt)
                P_init = Cameras{Float64}([Camera{Float64}(inv(NormMat[3*i-2:3*i, 3*i-2:3*i])*recovered_cameras_gpsfm[i]) for i=1:size(recovered_cameras_gpsfm,1)]);
                # P_init = Cameras{Float64}([Camera{Float64}(recovered_cameras_gpsfm[i]) for i=1:size(recovered_cameras_gpsfm,1)]);
                tOurs = @elapsed Ps, Wts = outer_irls(recover_cameras_iterative, F_mv, P_init, "subspace_angular", compute_error, max_iter_init=30, inner_method_max_it=5, weight_function=projective_synchronization.huber , c=projective_synchronization.c_huber, max_iterations=15, δ=1e-3, δ_irls=1e-1 , update_init="all", update="order-weights-update-all", set_anchor="fixed");
                Ps_mat = [NormMat[3*i-2:3*i, 3*i-2:3*i]*Matrix(Ps[i]) for i=1:length(Ps) ];
                # Ps_mat = [Matrix(Ps[i]) for i=1:length(Ps) ];
                err, BAt1, BAt2 = MATLAB.mxcall(:eval_from_julia, 3, Ps_mat, tracks );
                times[i,ct] = tGpsfm + tOurs
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




function process_data2(datasets, methods=["gpsfm", "ours"]; norm=false)
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
            recovered_cameras_gpsfm, tGpsfm, F_norm, NormMat, F_opt = MATLAB.mxcall(:runProjective_direct, 5, F, "gpsfm", matches, tracks );
            F_opt_norm = NormMat'*F_opt*NormMat
            if contains("ours", method)
                if norm
                    F_mv = wrap(F_opt_norm)
                else
                    F_mv = wrap(F_opt)
                end
                P_init = Vector{Camera{Float64}}(repeat([Camera_canonical], length(recovered_cameras_gpsfm)));
                # tOurs = @elapsed Ps, Wts = outer_irls(recover_cameras_iterative, F_mv, P_init, "subspace-angular", compute_error, max_iter_init=15, inner_method_max_it=5, weight_function=projective_synchronization.huber , c=projective_synchronization.c_huber, max_iterations=15, δ=1e-3, δ_irls=1e-1 , update_init="all", update="order-weights-update-all", set_anchor="fixed");
                tOurs = @elapsed Ps = recover_cameras_iterative(copy(F_mv); method="subspace_angular", update_init="none", δ=1e-1, update="order-centrality-all");
                if norm
                    Ps_mat = [NormMat[3*i-2:3*i, 3*i-2:3*i]*Matrix(Ps[i]) for i=1:length(Ps) ];
                else
                    Ps_mat = [Matrix{Float64}(Ps[i]) for i=1:length(Ps) ];
                end
                err, BAt1, BAt2 = MATLAB.mxcall(:eval_from_julia, 3, Ps_mat, tracks );
                times[i,ct] = tGpsfm + tOurs
                BA_times[i,1,ct] = BAt1 
                BA_times[i,2,ct] = BAt2
                
            else 
                err, BAt1, BAt2 = MATLAB.mxcall(:eval_from_julia, 3, recovered_cameras_gpsfm, tracks );
                times[i,ct] = tGpsfm
                BA_times[i,1,ct] = BAt1 
                BA_times[i,2,ct] = BAt2
                
            end
            errs[i,ct] = err
        end

    end
    return BA_times, times,errs
end


function threshold_and_eval(F, tracks, matches, thresh; synch=true, norm=false)
    F_mv = wrap(F)
    Matches = spzeros(size(F_mv)...)
    for i=1:size(F_mv,1)-1
        for j=i+1:size(F_mv,1)
            Matches[i,j] = copy(matches[i,j,1])
            Matches[j,i] = copy(matches[i,j,1])
        end
    end
    Adj_new = remove_fraction_edges(Matches; threshold=thresh)
    C = rand(4,size(Adj_new,1))*100;
    solvable = MATLAB.mxcall(:is_finite_solvable, 1, Matrix{Float64}(Adj_new), C, "eigs")
    if !solvable
        return false
    end

    tG, trips = get_triplet_cover(Adj_new; max_size=6000);
    covered_nodes = unique(reduce(hcat,trips[tG[2][1:nv(tG[1])]]));
    nonTriplet_cams = setdiff( collect(1:size(Adj_new,1)), covered_nodes)
    if length(nonTriplet_cams)==0
        println("All triplet covered")
        return false
    end
    println(size(trips), "\t",length(nonTriplet_cams))
    F_mv = SparseMatrixCSC{FundMat{Float64}, Int64}(F_mv .* Adj_new)
    Matches = SparseMatrixCSC{Float64, Int64}(Matches .* Adj_new);
    return F_mv, Matches, tracks
    F_multiview_gpsfm = F_mv[1:end .∉ Ref(nonTriplet_cams), 1:end .∉ Ref(nonTriplet_cams)];
    # F_unwrap = copy(unwrap(F_multiview_gpsfm));
    Matches_gpsfm = Matches[1:end .∉ Ref(nonTriplet_cams), 1:end .∉ Ref(nonTriplet_cams)];
    tracks_gpsfm = zeros(size(tracks,1)-2*length(nonTriplet_cams), size(tracks,2));
    
    j=1;
    for i=1:size(tracks,1)
        if i in 2*nonTriplet_cams .- 1 || i in 2*nonTriplet_cams
            continue
        else
            tracks_gpsfm[j,:] = copy(tracks[i,:])
            j+=1 
        end
    end

    recovered_cameras_gpsfm, _, _, _ = MATLAB.mxcall(:runProjective_direct, 4, unwrap(F_multiview_gpsfm), "gpsfm", Matches_gpsfm, tracks_gpsfm );
    err_gpsfm = MATLAB.mxcall(:eval_from_julia, 1, recovered_cameras_gpsfm, tracks_gpsfm );
    if synch
        # recovered_cameras_synch, _, _, _ = MATLAB.mxcall(:runProjective_direct, 4, unwrap(F_multiview_gpsfm), "synch", Matches_gpsfm, tracks_gpsfm );
        recovered_cameras_synch, tS = projective_synchronization.matlab_interface(unwrap(F_multiview_gpsfm), Matches_gpsfm, tracks_gpsfm; sim=false);

        err_synch = MATLAB.mxcall(:eval_from_julia, 1, recovered_cameras_synch, tracks_gpsfm );
    else
        err_synch = 0.0
    end

    P_init = Vector{Camera{Float64}}(repeat([Camera_canonical], size(F_mv,1)));
    P_init[intersect(collect(1:size(F_mv,1)), unique(reduce(hcat,trips[tG[2][1:nv(tG[1])]])) ) ] = recovered_cameras_gpsfm;

    if (norm)
        Ns = MATLAB.mxcall(:getNormalizationMatBigCondition,1,sqrt(2),tracks)
        F_norm = Ns'*F*Ns;
        F_mv_norm = wrap(F_norm)
        F_mv_norm = SparseMatrixCSC{FundMat{Float64}, Int64}(F_mv_norm .* Adj_new)

        for i =1:length(P_init)
            if isequal(P_init[i], Camera_canonical)
                continue
            end
            P_init[i] = inv(Ns[3*i-2:3*i, 3*i-2:3*i] )*P_init[i]
        end 
    end

    if norm
        Ps , Wts = outer_irls(recover_cameras_iterative, F_mv_norm, Cameras{Float64}(P_init), "subspace_angular", compute_error, max_iter_init=50, inner_method_max_it=15, weight_function=projective_synchronization.cauchy , c=projective_synchronization.c_cauchy, max_iterations=15, δ=1e-3, δ_irls=1e-1 , update_init="all", update="order-weights-update-all", set_anchor="fixed");
        Ps = [Ns[3*i-2:3*i, 3*i-2:3*i]*Matrix(Ps[i]) for i=1:length(Ps) ];
    else
        Ps , Wts = outer_irls(recover_cameras_iterative, F_mv, Cameras{Float64}(P_init), "subspace_angular", compute_error, max_iter_init=50, inner_method_max_it=15, weight_function=projective_synchronization.cauchy , c=projective_synchronization.c_cauchy, max_iterations=15, δ=1e-3, δ_irls=1e-1 , update_init="all", update="order-weights-update-all", set_anchor="fixed");
    end
    err_ours_all = MATLAB.mxcall(:eval_from_julia, 1, [Matrix{Float64}(P) for (i,P) in enumerate(Ps) ], tracks );
    err_ours_trips = MATLAB.mxcall(:eval_from_julia, 1, [Matrix{Float64}(P) for P in Ps ][intersect(collect(1:size(F_mv,1)), unique(reduce(hcat,trips[tG[2  ][1:nv(tG[1])]])) )] , tracks_gpsfm );

    return err_gpsfm, err_synch, err_ours_all, err_ours_trips
end


# datasets = ["Dino 319","Dino 4983","Corridor", "House", "Gustav Vasa", "Folke Filbyter", "Park Gate", "Nijo", "Drinking Fountain", "Golden Statue", "Jonas Ahls", "De Guerre", "Dome", "Alcatraz Courtyard", "Alcatraz Water Tower", "Cherub", "Pumpkin", "Sphinx", "Toronto University", "Sri Thendayuthapani", "Porta san Donato", "Buddah Tooth", "Tsar Nikolai I", "Smolny Cathedral", "Skansen Kronan"];
# datasets = ["Dino 319"]

# BAtimes, times, errors = process_data(datasets, ["gpsfm","ours"]);
# BAtimes, times, errors_normalized = process_data(datasets);
# println(errors_huber[:,2])
# println(errors_huber[:,2])
# BAtimes, times, errors_NOINIT = process_data2(datasets; norm=false);
# println(errors_NOINIT[:,2])
# times[23:25,:]
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
# res = threshold_and_eval(F,tracks,matches, 134;synch=true, norm=false);


# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability/Finite_solvability')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM/3rdparty/fromPPSFM/')"
# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM/3rdparty/vgg_code/')"

# test_mthds = ["gpsfm", "gpsfm-synch", "skew_symmetric_vectorized", "l1", "subspace_angular", "baseline_colombo", "baseline_sinha"]
# test_mthds = ["gpsfm", "skew_symmetric_vectorized-irls", "l1", "subspace_angular-irls", "baseline_colombo", "baseline_sinha"]
# init_mthds = ["gpsfm"]
# test_mthds = ["gpsfm", "skew_symmetric_vectorized", "subspace_angular", "l1"];

# E_noise_init_F = sensitivity(create_synthetic_environment, "noise", collect(0.0:0.0075:0.05), test_mthds, projective_synchronization.angular_distance; update_init="all", initialize=true, init_methods=init_mthds, num_trials=50, outliers_density=0.0, holes_density=0.4, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);
# E_missing_init_F = sensitivity(create_synthetic_environment, "noise", collect(0.0:0.0075:0.03), test_mthds, projective_synchronization.angular_distance; missing_initial=collect(0:0.1:0.7),  update_init="all", initialize=true, init_methods=["gpsfm"], num_trials=100, outliers_density=0.0, holes_density=0.4, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=100);

# E_noise_F = sensitivity(create_synthetic_environment, "noise", collect(0.0:0.0075:0.05), test_mthds, projective_synchronization.angular_distance; update_init="all", initialize=true, init_method=init_mthds, num_trials=100, holes_density=0.4, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);
# E_outliers_F = sensitivity(create_synthetic_environment, "outlier", collect(0.0:0.1:0.5), test_mthds, projective_synchronization.angular_distance; update_init="all", initialize=true, init_methods=init_mthds, num_trials=20, holes_density=0.5, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed");
# E_holes_F = sensitivity(create_synthetic_environment, "holes", collect(0.0:0.16:0.8), test_mthds, projective_synchronization.angular_distance; σ_fixed=0.015, update_init="all", initialize=true, init_method="gpsfm", num_trials=20, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed");
# timers = timer_experiment(create_synthetic_environment, collect(10:10:50), test_mthds; σ_fixed=0.015, holes_density=0.4, outliers_density=0.0, update_init="all", initialize=true, init_method="gpsfm", num_trials=10, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);

# e,c = general_graph_experiment(create_synthetic_environment, test_mthds, "noise", collect(0.0:0.005:0.03), projective_synchronization.angular_distance; update_init="all", initialize=true, init_method="gpsfm", num_trials=100, holes_density=0.75, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);
# test_mthds2 = ["subspace", "subspace_angular"]
# e2,c2 = general_graph_experiment(create_synthetic_environment, test_mthds2, "noise", collect(0.0:0.005:0.03), projective_synchronization.angular_distance; update_init="all", initialize=true, init_method="gpsfm", num_trials=100, holes_density=0.75, num_cams=25, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=50);

# for i in 1:length(E_missing_init_F[6])
    # if length(E_missing_init_F[6][i]) < 10
        # E_missing_init_F[6][i] = Inf*ones(10)
    # end
# end

# Errs_matrix = stack(stack.(E_holes_F)');
# Errs_matrix = rad2deg.(Errs_matrix);
# Errs_matrix = dropdims(Errs_matrix, dims = tuple(findall(size(Errs_matrix) .== 1)...));;
# file = MAT.matopen("Holes_withL1_20.mat", "w")
# write(file, "E", Errs_matrix)   
# close(file)




# times_matrix = stack(stack.(timers)');
# times_matrix = dropdims(times_matrix, dims = tuple(findall(size(times_matrix) .== 1)...));
# file = MAT.matopen("Times_numFrames_fixed_w_L1.mat", "w")
# write(file, "times", times_matrix)   
# close(file)