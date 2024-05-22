module cameras_from_F

import StructArrays, Distributions, MAT, StatsBase, Random, projective_synchronization

using LinearAlgebra, StaticArrays, Statistics, SparseArrays, Graphs, Arpack, ProgressBars, TiledIteration

#General
include("common/datatypes.jl")
include("common/solvers.jl")

#Synthetic
include("synthetic/simulation.jl")


end # module cameras_from_F
