module cameras_from_F

import StructArrays, Distributions, MAT, StatsBase, Random, projective_synchronization, Combinatorics, MATLAB, ForwardDiff
using LinearAlgebra, StaticArrays, Statistics, SparseArrays, Graphs, SimpleWeightedGraphs, Arpack, ProgressBars, TiledIteration, Static, LeastSquaresOptim, Bijections

function get_commutation_matrix(m::Integer, n::Integer)
    K = zeros(m*n, m*n)
    block_m = n
    block_n = m
    for i=1:m
        for j=1:n
            K[(i-1)*block_m+j, (j-1)*block_n+i] = 1 
        end
    end
    return SMatrix{m*n, m*n, Float64}(K)
end


#General
include("common/datatypes.jl")
include("common/cameras.jl")
include("common/multiview_utils.jl")
include("solvers/solvers.jl")
include("solvers/affine_solvers.jl")
include("solvers/baseline_solvers.jl")
include("solvers/recover_cameras_iterative.jl")
include("solvers/recover_cameras_global.jl")
include("solvers/irls.jl")
# include("solvers/epipolar_averaging_NLLSsolver.jl")
# include("solvers/epipolar_averaging_NonLinearSolve.jl")
include("solvers/explicit_alternation.jl")
include("solvers/multiview_sampson.jl")
include("solvers/sampson2view.jl")
#Special
# include("common/eccv_exp.jl")

#Synthetic
include("synthetic/simulation.jl")
include("synthetic/experiments.jl")
include("synthetic/simulation_averaging.jl")

end # module cameras_from_F
