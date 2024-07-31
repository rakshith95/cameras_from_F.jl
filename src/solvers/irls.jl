
function compute_weights(Z::AbstractMatrix, Ẑ::AbstractMatrix;error_measure=projective_synchronization.angular_distance, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, h=projective_synchronization.h_robust)
    E_UT = error_measure.(UpperTriangular(Z),UpperTriangular(Ẑ))
    E_LT = error_measure.(LowerTriangular(Z),LowerTriangular(Ẑ))
    E = min.(E_UT, E_LT')
    E = E + E'
    s = StatsBase.mad(E[.!isinf.(E)])
    if iszero(s)
        s = std(E[.!isinf.(E)])
    end
    if iszero(s)
        return false
    end
    wts = weight_function.(E/(h*c*s)) 
end

function outer_irls(iterative_fn, input_var::SparseMatrixCSC, X₀::AbstractVector{T}, iterative_method::String, error_fn; compute_Z_fn = (Z,X) -> compute_multiviewF_from_cams!(0.0,Z,X),  weights=nothing, inner_method_max_it=10, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, h=projective_synchronization.h_robust, error_measure=projective_synchronization.angular_distance, max_iterations=100, δ_irls=1e-6, kwargs...) where T
    max_iter_init = get(kwargs, :max_iter_int, 100)
    exit_loop = false
    iter = 0 
    n = size(input_var, 1)
    
    if isnothing(weights)
        wts = ones(n,n)
    end

    X_prev = copy(X₀)
    Ẑ = SparseMatrixCSC{eltype(input_var), Integer}(repeat([zero(eltype(input_var))], n,n))

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

        if mean(error_fn(X, X_prev, error_measure)) <= δ_irls
            X_prev = X
            exit_loop = true
        end
        X_prev = X
    end
    println(iter)
    return X_prev, wts
end


function l1_nullspace_irls(D::AbstractMatrix{T}, dimension::Integer;max_iter=100, convergence_δ=1e-6, δ=1e-4) where T<:AbstractFloat
    wts = ones(size(D,1))
    converge = false
    iter = 0

    x = get_NullSpace_svd(D)
    while !converge && iter < max_iter
        if norm(D*x, 1) < convergence_δ
            break
        end
        iter += 1
        x_prev = copy(x)
        for i=1:div(size(D,1), dimension)
            rᵢ = norm(D[(i-1)*dimension+1:i*dimension,:]*x_prev, 1)
            if iter>0
                wtsᵢ = 1/max(δ,rᵢ)
            else
                wtsᵢ = view(wts,i)
            end
            D[(i-1)*dimension+1:i*dimension,:] = sqrt(wtsᵢ)*D[(i-1)*dimension+1:i*dimension,:]
        end
        x = get_NullSpace_svd(D)
        if norm(x-x_prev, 1) < convergence_δ
            converge = true
        end
    end

    return x
end