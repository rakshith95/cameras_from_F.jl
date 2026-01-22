# Implement IRLS for affine system


# function norm_err(F₁::AbstractVecOrMat{T}, F₂::AbstractVecOrMat{T};p=2, normalize=false) where T<:AbstractFloat
#     if iszero(F₁) && iszero(F₂)
#         return 0.0
#     elseif (!iszero(F₁) && iszero(F₂)) || (iszero(F₁) && !iszero(F₂))
#         return Inf
#     else
#         if normalize
#             return norm(F₁/norm(F₁) - F₂/norm(F₂),p)
#         else
#             return norm(F₁-F₂,p)
#         end
#     end
# end

function compute_weights(Z::AbstractMatrix, Ẑ::AbstractMatrix;error_measure=projective_synchronization.angular_distance, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, h=projective_synchronization.h_robust)
    E_UT = error_measure.(UpperTriangular(Z),UpperTriangular(Ẑ))
    E = UpperTriangular(E_UT) + UpperTriangular(E_UT)'
    M = ones(Bool, size(Z)...)
    M[diagind(M)] .= false
    s = StatsBase.mad(E[.!isinf.(E) .&& UpperTriangular(M)])
    if iszero(s)
        s = 1e-10
        # s = std(E[.!isinf.(E)])
    end
    wts = weight_function.(E/(h*c*s)) 
    return wts
end

function edge_errors(Z::AbstractMatrix, Ẑ::AbstractMatrix; error_measure=projective_synchronization.angular_distance)
    E = error_measure.(UpperTriangular(Z),UpperTriangular(Ẑ))
    # E = E_UT + E_UT'
    M = ones(Bool, size(Z)...)
    M[diagind(M)] .= false
    return E[ .!isinf.(E) .&& UpperTriangular(M) ]
end

function outer_irls(iterative_fn, input_var::SparseMatrixCSC, X₀::AbstractVector{T}, iterative_method::String, error_fn; compute_Z_fn = (Z,X) -> compute_multiviewF_from_cams!(0.0,Z,X),  init_wts=nothing, inner_method_max_it=10, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, h=projective_synchronization.h_robust, error_measure=projective_synchronization.angular_distance, max_iterations=50, δ_irls=1e-6, kwargs...) where T
    max_iter_init = get(kwargs, :max_iter_init, 5)
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
    while iter < max_iterations
        if iszero(iter)
            X = iterative_fn(input_var; X₀=copy(X₀), method=iterative_method, weights=wts, max_iterations=max_iter_init , kwargs...)
        else
            X = iterative_fn(input_var; X₀=X_prev, min_updates=0, method=iterative_method, weights=wts, max_iterations=inner_method_max_it, kwargs...)
        end
        compute_Z_fn(Ẑ, X)
        wts = compute_weights(input_var, Ẑ, error_measure=error_measure, weight_function=weight_function, c=c, h=h)
        iter += 1

        if rad2deg(mean(error_fn(X, X_prev, error_measure))) <= δ_irls
            break
        end
        X_prev = X
    end
    # println(iter)
    return X_prev, wts
end

function norm_err(a::AbstractVecOrMat{T}, b::AbstractVecOrMat{T}) where T<:AbstractFloat
    return norm(a - b )
end

function compute_wts_aff(F_mv_in::AbstractSparseMatrix, Ps_est::Cameras{T}, error, weight_function, c, h; scale=nothing) where T<:AbstractFloat
    F_in_vec = [SVector{5,Float64}([ F_mv_in[i,j][1:2,end];F_mv_in[i,j][end,:]])  for i=1:size(F_mv_in,1)-1 for j=i+1:size(F_mv_in,1) if (!iszero(F_mv_in[i,j])) ];
    
    F_mv_est = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],size(F_mv_in)...))
    compute_multiviewF_from_cams!(0.0, F_mv_est, Ps_est, noise_type="angular"; normalize=false)
    F_est_vec = [SVector{5,Float64}([ F_mv_est[i,j][1:2,end];F_mv_est[i,j][end,:]])  for i=1:size(F_mv_est,1)-1 for j=i+1:size(F_mv_est,1) if (!iszero(F_mv_in[i,j])) ];
    
    errs = SVector{length(F_in_vec), T}([error(F_in_vec[i], F_est_vec[i]) for i=1:length(F_in_vec)] )
    if isnothing(scale)
        s = StatsBase.mad(errs)
        # s = std(errs)
        if s < 1e-10
            s = 1e-10
        end
    else
        s = scale
    end
    wts = weight_function.(errs/(h*c*s)) 
    return wts, StatsBase.mad(errs)
end

function lsq_irls(solve::Function, input_var=nothing; update_scale=false, compare_fn= (cams1, cams2) -> compute_error(cams1,cams2,projective_synchronization.angular_distance), max_it=50, δ=1e-6, error=projective_synchronization.angular_distance, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, h=projective_synchronization.h_robust, compute_wts= (output_var,s) -> compute_wts_aff(input_var,output_var, error, weight_function,c,h; scale=s))
    wts = ones(nnz(input_var))
    est_var = solve(wts)
    prev_est_var = copy(est_var)
    it = 0;
    wts,s = compute_wts(prev_est_var, nothing)
    while(it < max_it)
        est_var = solve(wts)
        if (mean(compare_fn(est_var, prev_est_var)) <= δ)
            break
        end
        prev_est_var = copy(est_var)
        if update_scale
            wts,_ = compute_wts(prev_est_var, nothing)
        else
            wts,_ = compute_wts(prev_est_var, s)
        end
        it += 1
    end
    # println(it)
    return est_var, wts
end

function wts_filter(wts::AbstractVector{Float64}; extend=false, window_size=1, set_last=0)
    if extend
        cnt = length(wts)
        wts_extended = zeros(cnt*window_size+set_last)
        for i=1:cnt
            wts_extended[(i-1)*window_size+1:i*window_size] .= wts[i]
        end
        wts = wts_extended
    else
        cnt = div(length(wts),window_size)
        for i=1:cnt
            window = @views wts[(i-1)*window_size+1:i*window_size]
            window .= minimum(window)
        end
    end    
    wts[end-set_last+1:end] .= 1    
    return wts
end 

function lsq_irls(A::AbstractMatrix, b::AbstractVector; max_it=50, δ=1e-3, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, h=projective_synchronization.h_robust, filter_weights=true, window=1, set_last=0, extend_wts=false, regularization=false)
    wts = ones(length(b))
    sol = inv(Symmetric(A'*A))*A'*b;
    # Q,R = qr(A)
    # sol = inv(UpperTriangular(R)) *(Matrix{Float64}(Q)'*b)

    closeness = 1e-4;
    if regularization
        α = 0.01
    else
        α = 0.0
    end

    prev_sol = copy(sol)
    it = 1;
    while(it <= max_it)
        r = abs.(A*prev_sol - b)
        # r = A*prev_sol - b
        if extend_wts
            s = StatsBase.mad(r[window:window:end-set_last], normalize=false)
        else
            s = StatsBase.mad(r[1:end-set_last], normalize=false)
        end
            
        if s<1e-10
            # s = std(r)
            s = 1e-10
        end
        if extend_wts
            wts = weight_function.(r[window:window:end-set_last]/(h*c*s)) 
        else
            wts = weight_function.(r/(h*c*s)) 
        end
        if filter_weights
            wts = wts_filter(wts; extend=extend_wts, window_size=window, set_last=set_last)
        end
        # println(wts[end-11+1:end])
        W = diagm(wts)

        try
            # sol = inv( Symmetric(A'*W*A - α*Matrix(I,size(A,2),size(A,2))) )*A'*W*b
            sol = (sqrt.(W)*A)\(sqrt.(W)*b)
        catch
            sol = (sqrt.(W)*A)\(sqrt.(W)*b)
        end

        if norm(sol - prev_sol)/norm(prev_sol) < δ
            break
        end
        prev_sol = copy(sol)
        it += 1
    end
    # println(it)
    # println(count(wts .< 1e-6))
    return sol
end

function gnc(solve::Function, input_var=nothing; γ = 1.0, σ_init=1e6, σ_final=0.0, compare_fn= (cams1, cams2) -> compute_error(cams1,cams2,projective_synchronization.angular_distance), max_it=50, δ=1e-6, error=projective_synchronization.angular_distance, weight_function=projective_synchronization.huber, c=projective_synchronization.c_huber, h=projective_synchronization.h_robust, compute_wts= (output_var, s) -> compute_wts_aff(input_var,output_var, projective_synchronization.angular_distance,weight_function,c,h;scale=s))
    wts = ones(nnz(input_var))
    est_var = solve(wts)
    prev_est_var = copy(est_var)
    it = 0;
    σ = σ_init;
    while(it < max_it)
        # println(σ," ", σ_final)
        if (σ < σ_final)
            break
        end
        wts, s = compute_wts(prev_est_var, σ)
        # if (it==0)
            # σ_final = s;
        # end
        est_var = solve(wts)
        # println(mean(compare_fn(est_var, prev_est_var)))
        if (mean(compare_fn(est_var, prev_est_var)) <= δ) 
            break
        end
        prev_est_var = copy(est_var)
        σ = σ/(1+γ)
        it += 1
    end
    # println(it)
    return est_var, wts
end
