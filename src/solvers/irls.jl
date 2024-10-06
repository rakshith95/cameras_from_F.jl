function frobenius_norm_err(F₁::AbstractMatrix{T}, F₂::AbstractMatrix{T}) where T<:AbstractFloat
    if iszero(F₁) && iszero(F₂)
        return 0.0
    elseif (!iszero(F₁) && iszero(F₂)) || (iszero(F₁) && !iszero(F₂))
        return Inf
    else
        return norm(F₁/norm(F₁) - F₂/norm(F₂))
    end
end

function compute_weights(Z::AbstractMatrix, Ẑ::AbstractMatrix;error_measure=projective_synchronization.angular_distance, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, h=projective_synchronization.h_robust)
    E_UT = error_measure.(UpperTriangular(Z),UpperTriangular(Ẑ))
    E = E_UT + E_UT'
    M = ones(Bool, size(Z)...)
    M[diagind(M)] .= false
    s = StatsBase.mad(E[.!isinf.(E) .&& M])
    if iszero(s)
        s = 1e-10
        # s = std(E[.!isinf.(E)])
    end
    wts = weight_function.(E/(h*c*s)) 
    return wts
end

function outer_irls(iterative_fn, input_var::SparseMatrixCSC, X₀::AbstractVector{T}, iterative_method::String, error_fn; compute_Z_fn = (Z,X) -> compute_multiviewF_from_cams!(0.0,Z,X),  init_wts=nothing, inner_method_max_it=10, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, h=projective_synchronization.h_robust, error_measure=projective_synchronization.angular_distance, max_iterations=50, δ_irls=1e-6, kwargs...) where T
    max_iter_init = get(kwargs, :max_iter_init, 5)
    exit_loop = false
    iter = 0 
    n = size(input_var, 1)
    Ẑ = SparseMatrixCSC{eltype(input_var), Integer}(repeat([zero(eltype(input_var))], n,n))
    
    if isnothing(init_wts)
        compute_Z_fn(Ẑ, X₀)
        wts = compute_weights(input_var, Ẑ, error_measure=error_measure, weight_function=weight_function, c=c, h=h)
    else
        wts = init_wts
    end
    X_prev = copy(X₀)
    while !exit_loop && iter < max_iterations
        if iszero(iter)
            X = iterative_fn(input_var; X₀=copy(X₀), method=iterative_method, weights=wts, max_iterations=max_iter_init , kwargs...)
        else
            X = iterative_fn(input_var; X₀=X_prev, min_updates=0, method=iterative_method, weights=wts, max_iterations=inner_method_max_it, kwargs...)
        end
        compute_Z_fn(Ẑ, X)
        wts = compute_weights(input_var, Ẑ, error_measure=error_measure, weight_function=weight_function, c=c, h=h)
        iter += 1

        if rad2deg(mean(error_fn(X, X_prev, error_measure))) <= δ_irls
            exit_loop = true
        end
        X_prev = X
    end
    println(iter)
    return X_prev, wts
end


function l1_nullspace_irls(Ps::Cameras{T}, Fs::FundMats{T}, wts = ones(length(Ps)), P₀=nothing; weight_function=projective_synchronization.huber, c=projective_synchronization.c_huber, recover_camera=recover_camera_subspace, max_iter=50, convergence_δ=1e-4) where T<:AbstractFloat
    converge = false
    iter = 0
    Ns = [nullspace( (kron( (Ps[i]'*Fs[i]') , I₄)*K₃₄) + (kron( I₄, Ps[i]'*Fs[i]' )) ) for i=1:length(Ps)];
    x = vec(recover_camera(Ps,Fs, wts, P₀))
    

    while !converge && iter < max_iter
        x_prev = copy(x)
        iter += 1

        r = [norm(subspace_obj_i(x,Ns[i])) for i=1:length(Ps)]
        s = StatsBase.mad(r)
        if isapprox(s,0.0)
            s = 1e-10
        end
        wts = weight_function.(r./(c*s))
        x = vec(recover_camera(Ps,Fs, wts, x_prev))
        if norm(x-x_prev) < convergence_δ
            converge = true
        end
    end
    # println(wts)
    return x
end