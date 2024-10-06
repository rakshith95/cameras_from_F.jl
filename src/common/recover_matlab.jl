push!(LOAD_PATH, "/home/rakshith/PoliMi/Recovering Cameras/cameras_from_F.jl")
push!(LOAD_PATH, "/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/projective_synchronization.jl")
using LinearAlgebra
import cameras_from_F
import projective_synchronization
import SparseArrays, MAT, MATLAB, StaticArrays, Statistics
MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability')"
MATLAB.mat"addpath('/home/rakshith/PoliMi/Recovering Cameras/finite-solvability/Finite_solvability')"
MATLAB.mat"addpath('/home/rakshith/PoliMi/Projective Synchronization/projective-synchronization-julia/GPSFM-code/GPSFM')"

function recover_cameras_from_mat(F_filename::String, init_filename::String, normFile::String, wtsFile::String, Afile::String, output_file::String; method="subspace",varname="FN")
    file = MAT.matopen(F_filename)
    F = read(file, varname)
    close(file)
    F_multiview = cameras_from_F.wrap(F)

    norm_file = MAT.matopen(normFile)
    N = read(norm_file, "normMat")
    close(norm_file)

    wts_file = MAT.matopen(wtsFile)
    inlier_wts = read(wts_file, "wts")
    inlier_wts = (inlier_wts .- minimum(inlier_wts))./(maximum(inlier_wts) - minimum(inlier_wts))
    close(wts_file)

    A_file = MAT.matopen(Afile)
    A = read(A_file, "VG")
    close(wts_file)

    init_file = MAT.matopen(init_filename)
    P_init =read(init_file, "Ps_gpsfm")
    close(init_file)
     
    P_init = cameras_from_F.Cameras{Float64}([cameras_from_F.Camera{Float64}(inv(N[3*i-2:3*i, 3*i-2:3*i])*P_init[i]) for i=1:size(P_init,1)]);
    # P_init = cameras_from_F.Cameras{Float64}([cameras_from_F.Camera{Float64}(P_init[i]) for i=1:size(P_init,1)]);
    
    # n = size(P_init,1);
    # Ẑ = SparseArrays.SparseMatrixCSC{eltype(F_multiview), Integer}(repeat([zero(eltype(F_multiview))], n,n))
    # cameras_from_F.compute_multiviewF_from_cams!(0.0,Ẑ,P_init)
    # wts = cameras_from_F.compute_weights(F_multiview, Ẑ, weight_function=projective_synchronization.cauchy_sq, c=projective_synchronization.c_cauchy)
    # wts = wts .* inlier_wts
    # wts = (wts .- minimum(wts))./(maximum(wts) - minimum(wts))
    # wts[wts.<0.1] .= 0

    #Remove % of F depending on weights, while maintaining solvability
    # cameras_from_F.remove_Fs(F_multiview, wts,remove_frac=0.1)

    # F_multiview = SparseArrays.SparseMatrixCSC{eltype(F_multiview), Integer}(F_multiview .* A)  
    # Ps = cameras_from_F.recover_cameras_iterative(F_multiview; X₀=P_init, method=method, update="order-weights-update-all", min_updates=1, δ=1e-3, set_anchor="fixed", max_iterations=50);
    Ps, Wts = cameras_from_F.outer_irls(cameras_from_F.recover_cameras_iterative, F_multiview, P_init, method, cameras_from_F.compute_error, max_iter_init=50, inner_method_max_it=5, weight_function=projective_synchronization.cauchy , c=projective_synchronization.c_cauchy, max_iterations=25, δ=0.1, δ_irls=0.5, update_init="all", update="order-weights-update-all",  set_anchor="fixed");
    
    Ps_mat = [N[3*i-2:3*i, 3*i-2:3*i]*Matrix(Ps[i]) for i=1:length(Ps) ];
    # Ps_mat = [Matrix(Ps[i]) for i=1:length(Ps) ];
    # Ps_mat = [Ps_mat[i]/LinearAlgebra.norm(Ps_mat[i]) for i=1:length(Ps) ];

    file = MAT.matopen(output_file, "w")
    write(file, "Ps_iterative", Ps_mat)
    close(file) 
end


function mod_huber(r) 
    return 1/max(1e-3, abs(r))
end

function main(args)
    F_file = args[1]
    init_file = args[2]
    normMat_file = args[3]
    wts_file = args[4]
    A_file = args[5]
    output_file = args[6]

    method = "subspace"

    recover_cameras_from_mat(F_file, init_file, normMat_file, wts_file, A_file, output_file; method=method)
end


main(ARGS)


# F_filename = "F.mat"
# init_filename = "Ps_init.mat"
# normFile = "normMat.mat"
# wtsFile = "wts.mat"
# output_file = "Ps_iterative.mat"
# Afile = "AMat.mat"
# method = "subspace"
# method = "skew_symmetric_vectorized"
 
# file = MAT.matopen(F_filename)
# F = read(file, "FN");
# close(file)
# F_multiview = cameras_from_F.wrap(F)
# norm_file = MAT.matopen(normFile)
# N = read(norm_file, "normMat");
# close(norm_file)
# wts_file = MAT.matopen(wtsFile)
# wts1 = read(wts_file, "wts")
# wts1 = (wts1 .- minimum(wts1))./(maximum(wts1) - minimum(wts1))
# 
# close(wts_file)
# init_file = MAT.matopen(init_filename)
# P_init =read(init_file, "Ps_gpsfm")
# close(init_file)

# A_file = MAT.matopen(Afile)
# A = read(A_file, "VG")
# close(A_file)

# P_init = cameras_from_F.Cameras{Float64}([cameras_from_F.Camera{Float64}(inv(N[3*i-2:3*i, 3*i-2:3*i])*P_init[i]) for i=1:size(P_init,1)]);
# P_init = cameras_from_F.Cameras{Float64}([cameras_from_F.Camera{Float64}(P_init[i]) for i=1:size(P_init,1)]);
# 
# n = size(P_init,1);
# Ẑ = SparseArrays.SparseMatrixCSC{eltype(F_multiview), Integer}(repeat([zero(eltype(F_multiview))], n,n))
# cameras_from_F.compute_multiviewF_from_cams!(0.0,Ẑ,P_init)
# wts = cameras_from_F.compute_weights(F_multiview, Ẑ, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy)
# minimum(wts)
# wts[LinearAlgebra.diagind(wts)] .= 0
# wts = wts .* wts1


# wts = SparseArrays.SparseMatrixCSC{Float64, Integer}(wts)
# wts[LinearAlgebra.diagind(wts)] .= 0
# wts = SparseArrays.dropzeros(wts);

#Remove least 10% weights while maintaining solvability
# cameras_from_F.remove_Fs(F_multiview, wts,remove_frac=0.2)

# F_multiview = SparseArrays.SparseMatrixCSC{eltype(F_multiview), Integer}(F_multiview .* A)  
# findmin(wts)
# F_multiview[34,10]

# # Ps = cameras_from_F.recover_cameras_iterative(F_multiview; X₀=P_init, method=method, weights= wts,  update_init="all", update="order-weights-update-all", δ=1e-3, set_anchor="fixed", max_iterations=30);
# Ps, Wts = cameras_from_F.outer_irls(cameras_from_F.recover_cameras_iterative, F_multiview, P_init, method, cameras_from_F.compute_error, max_iter_init=1, inner_method_max_it=5, weight_function=projective_synchronization.cauchy , c=projective_synchronization.c_cauchy  ,max_iterations=5, δ=1e-3, δ_irls=1.0, update_init="all", update="order-weights-update-all",  set_anchor="fixed");
# Statistics.mean(rad2deg.( cameras_from_F.compute_error(cameras_from_F.Cameras{Float64}(P_init), cameras_from_F.Cameras{Float64}(Ps), projective_synchronization.angular_distance)    ))

# for i=1:n 
#     display(Ps[i])
#     display(P_init[i])
#     println("\n")
# end

# rad2deg(Statistics.mean(cameras_from_F.compute_error(Ps, P_init, projective_synchronization.angular_distance)))

# Ps_mat = [N[3*i-2:3*i, 3*i-2:3*i]*Matrix(Ps[i]) for i=1:length(Ps) ];
# Ps_mat = [Matrix(P_init[i]) for i=1:length(Ps) ];
# file = MAT.matopen(output_file, "w")
# write(file, "Ps_iterative", Ps_mat)
# close(file)

