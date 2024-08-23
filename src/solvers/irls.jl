
# function frobenius_norm(F1::FundMat{T}, F2::FundMat{T}) where T<:AbstractFloat
    # if iszero(F1) && iszero(F2)
        # return 0.0
    # elseif (!iszero(F1) && iszero(F2)) || (iszero(F1) && !iszero(F2))
        # return Inf
    # else
        # return norm( F1/norm(F1) - F2/norm(F2) )
    # end
# end

function compute_weights(Z::AbstractMatrix, Ẑ::AbstractMatrix;error_measure=projective_synchronization.angular_distance, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, h=projective_synchronization.h_robust)
    E_UT = error_measure.(UpperTriangular(Z),UpperTriangular(Ẑ))
    E_LT = error_measure.(LowerTriangular(Z),LowerTriangular(Ẑ))
    # E = min.(E_UT, E_LT')
    E = (E_UT + E_LT')/2
    E = E + E'
    s = StatsBase.mad(E[.!isinf.(E)])
    if iszero(s)
        s = 1e-3
        # s = std(E[.!isinf.(E)])
    end
    if iszero(s)
        return false
    end
    wts = weight_function.(E/(h*c*s)) 
    return wts
end

function outer_irls(iterative_fn, input_var::SparseMatrixCSC, X₀::AbstractVector{T}, iterative_method::String, error_fn; compute_Z_fn = (Z,X) -> compute_multiviewF_from_cams!(0.0,Z,X),  weights=nothing, inner_method_max_it=10, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, h=projective_synchronization.h_robust, error_measure=projective_synchronization.angular_distance, max_iterations=50, δ_irls=1e-6, kwargs...) where T
    max_iter_init = get(kwargs, :max_iter_int, 50)
    exit_loop = false
    iter = 0 
    n = size(input_var, 1)
    Ẑ = SparseMatrixCSC{eltype(input_var), Integer}(repeat([zero(eltype(input_var))], n,n))
    
    if isnothing(weights)
        compute_Z_fn(Ẑ, X₀)
        wts = compute_weights(input_var, Ẑ, error_measure=error_measure, weight_function=weight_function, c=c, h=h)
        # display(wts)
        # wts = ones(n,n)
    end

    X_prev = copy(X₀)

    while !exit_loop && iter <= max_iterations
        if iszero(iter)
            X = iterative_fn(copy(input_var); X₀=copy(X₀), method=iterative_method, weights=wts, max_iterations=max_iter_init , kwargs...)
        else
            X = iterative_fn(copy(input_var); X₀=copy(X_prev), min_updates=0, method=iterative_method, weights=wts, max_iterations=inner_method_max_it, kwargs...)
        end
        compute_Z_fn(Ẑ, X)
        wts_prev = copy(wts)
        wts = compute_weights(input_var, Ẑ, error_measure=error_measure, weight_function=weight_function, c=c, h=h)
        if typeof(wts) == Bool
            return X, wts_prev
        end
        iter += 1

        if rad2deg(mean(error_fn(X, X_prev, error_measure))) <= δ_irls
            X_prev = X
            exit_loop = true
        end
        X_prev = X
    end
    println(iter)
    return X_prev, wts
end


function l1_nullspace_irls(Ps::Cameras{T}, Fs::FundMats{T}, wts = ones(length(Ps)); recover_camera=recover_camera_subspace, max_iter=50, convergence_δ=1e-4, δ=1e-3) where T<:AbstractFloat
    converge = false
    iter = 0
    Ms = Vector{SMatrix{16,12,Float64}}(undef,length(Ps))

    for i=1:length(Ps)
        Ms[i] = ( kron( (Ps[i]'*Fs[i]') , I₄)*K₃₄) + (kron( I₄, Ps[i]'*Fs[i]' ))
    end

    x = vec(recover_camera(Ps,Fs, wts))
    while !converge && iter < max_iter
        x_prev = copy(x)
        iter += 1
        for i=1:length(Ps)
            # rᵢ = @views norm(obj_i(x, Ms[i]))
            wts[i] = @views 1/max(δ,norm(obj_i(x, Ms[i])))
        end
        x = vec(recover_camera(Ps,Fs, wts))
        if norm(x-x_prev) < convergence_δ
            converge = true
        end
    end
    # println(wts)
    return x
end