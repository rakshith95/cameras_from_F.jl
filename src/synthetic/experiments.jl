function sensitivity(synthetic_env_creator, param_type::String, param_range::Vector{T}, test_methods::Vector{String}, error; num_trials=1e3, kwargs...) where T<:AbstractFloat
    E = Vector{Vector{Vector{Float64}}}(undef, length(param_range))
    if occursin("noise", param_type)
        i=1
        for σ=tqdm(param_range)
            Eᵢ = Vector{Vector{Float64}}(undef, num_trials)
            for j=tqdm(1:num_trials)
                Eⱼ = synthetic_env_creator(σ, test_methods; kwargs...)
                Eᵢ[j] = mean.(eachcol(Eⱼ))
            end
            E[i] = Eᵢ
            i += 1
        end
        return E
    end
end

# MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM')"

# test_mthds =  ["gpsfm", "skew_symmetric_vectorized", "skew_symmetric-l1"]
# E_noise_F = sensitivity(create_synthetic_environment, "noise", collect(0.0:0.01:0.05), test_mthds, projective_synchronization.angular_distance; update_init="all", initialize=true, init_method="gpsfm", num_trials=50, holes_density=0.2, num_cams=20, noise_type="angular", update="random-all", set_anchor="fixed", max_iterations=200);

# test_mthds = ["skew_symmetric_vectorized"]
# Errs_matrix = stack(stack.(E_noise_F)');
# Errs_matrix = rad2deg.(Errs_matrix);
# Errs_matrix = dropdims(Errs_matrix, dims = tuple(findall(size(Errs_matrix) .== 1)...));;
# file = MAT.matopen("Noise_Fs.mat", "w")
# write(file, "E", Errs_matrix)   
# close(file)

# GPSFM + Synchronization as a competitor 
# Test with gpsfm datasets  
# Get cameras covered by triplets with similar method to what GPSFM does.
# Weighted outliers
# Check L1 loss instead of least squares error for F estimation. 

# 1D SFM 

#ALL-RANDOM INSTEAD OF RANDOM UPDATE WORKS BETTER