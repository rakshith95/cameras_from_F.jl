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

    init_file = MAT.matopen(init_filename)
    P_init =read(init_file, "Ps_gpsfm")
    close(init_file)
    
    P_init = cameras_from_F.Cameras{Float64}([cameras_from_F.Camera{Float64}(inv(N[3*i-2:3*i, 3*i-2:3*i])*P_init[i]) for i=1:size(P_init,1)]);
    # P_init = cameras_from_F.Cameras{Float64}([cameras_from_F.Camera{Float64}(P_init[i]) for i=1:size(P_init,1)]);

    # F_multiview = SparseArrays.SparseMatrixCSC{eltype(F_multiview), Integer}(F_multiview .* A)  
    # Ps = cameras_from_F.recover_cameras_iterative(F_multiview; X₀=P_init, method=method, update="order-weights-update-all", min_updates=1, δ=1e-3, set_anchor="fixed", max_iterations=50);
    Ps, Wts = cameras_from_F.outer_irls(cameras_from_F.recover_cameras_iterative, F_multiview, P_init, method, cameras_from_F.compute_error, max_iter_init=15, inner_method_max_it=5, weight_function=projective_synchronization.cauchy , c=projective_synchronization.c_cauchy, max_iterations=15, δ=1e-3, δ_irls=1e-1 , update_init="all", update="order-weights-update-all", set_anchor="fixed");
    
    Ps_mat = [N[3*i-2:3*i, 3*i-2:3*i]*Matrix(Ps[i]) for i=1:length(Ps) ];
    # Ps_mat = [Matrix(Ps[i]) for i=1:length(Ps) ];

    file = MAT.matopen(output_file, "w")
    write(file, "Ps_iterative", Ps_mat)
    close(file) 
end

function recover_cameras_from_mat(F_filename::String, ST_filename::String, triplets_filename::String, output_filename::String, method::String)
    file = MAT.matopen(F_filename)
    F = read(file, "FN")
    close(file)
    F_multiview = cameras_from_F.wrap(F)
    
    st_file = MAT.matopen(ST_filename);
    st = Matrix{Integer}(read(st_file, "ST"));
    close(st_file);

    triplets_file = MAT.matopen(triplets_filename);
    trips = read(triplets_file, "triplets");
    close(triplets_file)

    trips = [Integer.(trips[i,:]) for i=1:size(trips,1)];
    mat_data = Dict([("ST",st), ("triplets",trips)]);

    Ps_baseline = cameras_from_F.recover_cameras_baselines(F_multiview, method; matlab_data = mat_data)
    Ps_mat = [Matrix(Ps_baseline[i]) for i=1:length(Ps_baseline) ];

    output_file = MAT.matopen(output_filename, "w")
    write(output_file, "Ps_baseline", Ps_mat)
    close(output_file) 
end

function mod_huber(r) 
    return 1/max(1e-3, abs(r))
end

function main(args)
    method = "baseline colombo"
    F_file = args[1]
    ST_file = args[2]
    triplets_file = args[3]
    output_file = args[4]

    if contains(lowercase(method),"baseline")
        recover_cameras_from_mat(F_file, ST_file, triplets_file, output_file,method)
    else
        init_file = args[2]
        normMat_file = args[3]
        wts_file = args[4]
        A_file = args[5]
        output_file = args[6]    
        recover_cameras_from_mat(F_file, init_file, normMat_file, wts_file, A_file, output_file; method=method)
    end
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

