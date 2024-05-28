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


# test_mthds = ["skew_symmetric","skew_symmetric_vector"]
# E_noise_F = sensitivity(create_synthetic_environment, "noise", collect(0.0:0.01:0.05), test_mthds, projective_synchronization.angular_distance; num_trials=100, holes_density=0.0, num_cams=20, noise_type="angular", update="all-random", set_anchor="centrality", max_iterations=2000);

# Errs_matrix = stack(stack.(E_noise_F)');
# Errs_matrix = rad2deg.(Errs_matrix)
# Errs_matrix = dropdims(Errs_matrix, dims = tuple(findall(size(Errs_matrix) .== 1)...));
# file = MAT.matopen("Noise_Fs.mat", "w")
# write(file, "E", Errs_matrix)
# close(file)
