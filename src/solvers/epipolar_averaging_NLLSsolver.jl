# For NLLS.jl -

struct EpipolarErr{T<:AbstractFloat} <: NLLSsolver.AbstractResidual
    Fs::FundMats{T}
    nconstraints::Int
end

struct PointDistanceErr{T<:AbstractFloat} <:NLLSsolver.AbstractResidual
    Xs::Vector{T}
    nPts::Int 
end


NLLSsolver.ndeps(::EpipolarErr) = static(1); # For now, depends only on 1 variable (the stacked 2D points X)
NLLSsolver.ndeps(::PointDistanceErr) = static(1)

NLLSsolver.nres(res::EpipolarErr) = res.nconstraints; # n constraints will be ∑ᵢ |Ne(i)| 
NLLSsolver.nres(r::PointDistanceErr) = 6*r.nPts #3*r.nPts 

NLLSsolver.varindices(::EpipolarErr) = SVector(1) # a bit confusing as to what this is..
NLLSsolver.varindices(::PointDistanceErr) = SVector(1)

NLLSsolver.getvars(::EpipolarErr{T}, vars::Vector) where T<:AbstractFloat = (vars[1]::Vector{T},)
NLLSsolver.getvars(::PointDistanceErr{T}, vars::Vector) where {T<:AbstractFloat} = (vars[1]::Vector{T},)

NLLSsolver.computeresidual(res::EpipolarErr{T1}, X::AbstractVector{T}) where {T1<:AbstractFloat, T} = triplet_coincidence_angle(X, res.Fs)
NLLSsolver.computeresidual(r::PointDistanceErr{T1}, X::AbstractVector{T}) where {T1<:AbstractFloat, T} = point_dist_cost(X, r.Xs)

Base.eltype(::EpipolarErr{T}) where T = T 
Base.eltype(::PointDistanceErr{T}) where {T<:AbstractFloat} = T

function epipolar_ls_triplet(x1::Pts2D_homo{T}, x2::Pts2D_homo{T}, x3::Pts2D_homo{T}, F21::FundMat{T}, F31::FundMat{T}, F32::FundMat{T}) where T<:AbstractFloat
    # problem = NLLSsolver.NLLSProblem(Vector{T}, EpipolarErr{T}) 
    problem = NLLSsolver.NLLSProblem(Vector{T}, Union{EpipolarErr{T}, PointDistanceErr{T}})

    X₀ = vcat(reduce(vcat, euclideanize.(x1)),reduce(vcat, euclideanize.(x2)), reduce(vcat, euclideanize.(x3)));
    ncons = length(X₀) # Note: this is only true for the very toy  triplet example. ONLY FOR POC
    nPts = div(ncons, 6)

    NLLSsolver.addvariable!(problem, X₀)
    Fs = FundMats{T}([F21, F31, F32])  # Your fundamental matrices
    
    epipolar_cost_block = EpipolarErr{T}(Fs, ncons)
    dist_cost_block = PointDistanceErr{T}(X₀, nPts)

    NLLSsolver.addcost!(problem, epipolar_cost_block)
    # NLLSsolver.addcost!(problem, dist_cost_block)

    opts = NLLSsolver.NLLSOptions(iterator = NLLSsolver.levenbergmarquardt, maxiters = 100, reldcost = 1e-10, absdcost = 1e-12 )
    result = NLLSsolver.optimize!(problem, opts)
    X_optimal = problem.variables[1]

    println("Final cost: $(result.bestcost)")
    println("Iterations: $(result.niterations)")
    return X_optimal
end

function unwrap_X(X::Vector{T}) where T<:AbstractFloat
 #Temporary for toy triplet case
    l = length(X)
    n = div(l,6)
    X1 = X[1:2*n]
    x1 = Pts2D_homo{T}([ homogenize(X1[ (i-1)*2+1:2*i ]) for i=1:n  ])
    
    X2 = X[2*n+1:4*n]
    x2 = Pts2D_homo{T}([ homogenize(X2[ (i-1)*2+1:2*i ]) for i=1:n  ])
    
    X3 = X[4*n+1:end]
    x3 = Pts2D_homo{T}([ homogenize(X3[ (i-1)*2+1:2*i ]) for i=1:n  ])

    return x1,x2,x3
end



# X_gt = vcat(reduce(vcat, euclideanize.(x1_pts)),reduce(vcat, euclideanize.(x2_pts)), reduce(vcat, euclideanize.(x3_pts)));
# # norm(triplet_coincidence_cost(X_gt, FundMats{Float64}([F21,F31,F32])))

# X_noised = vcat(reduce(vcat, euclideanize.(x1_noised)),reduce(vcat, euclideanize.(x2_noised)), reduce(vcat, euclideanize.(x3_noised)));
# norm(triplet_coincidence_cost(X_noised, FundMats{Float64}([F21,F31,F32])))


# X_opt = epipolar_ls_triplet(x1_noised, x2_noised, x3_noised, F21, F31, F32);
# norm(triplet_coincidence_angle(X_opt, FundMats{Float64}([F21,F31,F32])))
# # norm(X_opt - X_noised)
# # norm(triplet_coincidence_cost(X_opt, FundMats{Float64}([F21,F31,F32])))


# F21_noised = F_8ptNorm(x1_noised, x2_noised);
# F31_noised = F_8ptNorm(x1_noised, x3_noised);
# F32_noised = F_8ptNorm(x2_noised, x3_noised);

# Ps_noised = get_cams_from_triplet_sinha(FundMats{Float64}([F21_noised,F31_noised,F32_noised]));
# e = compute_error(Cameras{Float64}([P1,P2,P3]), Ps_noised[1], projective_synchronization.angular_distance );
# rad2deg(mean(e))

# x1_opt, x2_opt, x3_opt = unwrap_X(X_opt);

# F21_opt = F_8ptNorm(x1_opt, x2_opt);
# F31_opt = F_8ptNorm(x1_opt, x3_opt);
# F32_opt = F_8ptNorm(x2_opt, x3_opt);

# Ps_opt = get_cams_from_triplet_sinha(FundMats{Float64}([F21_opt,F31_opt,F32_opt]));
# e = compute_error(Cameras{Float64}([P1,P2,P3]), Ps_opt[1], projective_synchronization.angular_distance );
# rad2deg(mean(e))



# x3_opt[16]'*F31*x1_opt[16]



# norm( (F21_opt/norm(F21_opt)) - (F21/norm(F21)) )

# x1_gt = reduce(hcat, x1_pts);
# x2_gt = reduce(hcat, x2_pts)
# x3_gt = reduce(hcat, x3_pts)

# file = MAT.matopen("corres_pts.mat", "w")
# write(file, "x1_gt", x1_gt)   
# write(file, "x2_gt", x2_gt)   
# write(file, "x3_gt", x3_gt)   

# write(file, "x1_noised", reduce(hcat, x1_noised) )   
# write(file, "x2_noised", reduce(hcat, x2_noised) )   
# write(file, "x3_noised", reduce(hcat, x3_noised) )   
# # 
# write(file, "x1_opt", reduce(hcat, x1_opt) )    
# write(file, "x2_opt", reduce(hcat, x2_opt) )    
# write(file, "x3_opt", reduce(hcat, x3_opt) )    

# close(file)