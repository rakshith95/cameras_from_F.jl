push!(LOAD_PATH, "/home/rakshith/PoliMi/Recovering Cameras/cameras_from_F.jl")
import cameras_from_F
import SparseArrays, LinearAlgebra, MAT

function recover_cameras_from_mat(F_filename::String, init_filename::String, normFile::String, output_file; method="skew_symmetric_vectorized",varname="FN")
    file = MAT.matopen(F_filename)
    F = read(file, varname)
    close(file)
    F_multiview = cameras_from_F.wrap(F)

    norm_file = MAT.matopen(normFile)
    N = read(norm_file, "normMat")
    close(file)

    init_file = MAT.matopen(init_filename)
    P_init =read(init_file, "Ps_gpsfm")
    close(init_file)
    
    P_init = cameras_from_F.Cameras{Float64}([cameras_from_F.Camera{Float64}(inv(N[3*i-2:3*i, 3*i-2:3*i])*P_init[i]) for i=1:size(P_init,1)]);
    # P_init = cameras_from_F.Cameras{Float64}([cameras_from_F.Camera{Float64}(P_init[i]) for i=1:size(P_init,1)]);
    Ps = cameras_from_F.recover_cameras_iterative(F_multiview; P₀=P_init, recovery_method=method, update_init="all", update="start-centrality-update-all", set_anchor="fixed", max_iterations=200);
    Ps_mat = [N[3*i-2:3*i, 3*i-2:3*i]*Matrix(Ps[i]) for i=1:length(Ps) ];
    #Ps_mat = [Matrix(recovered_cams[i]) for i=1:length(recovered_cams) ];

    file = MAT.matopen(output_file, "w")
    write(file, "Ps_iterative", Ps_mat)
    close(file)
end

function main(args)
    F_file = args[1]
    init_file = args[2]
    normMat_file = args[3]
    output_file = args[4]

    method = "skew_symmetric_vectorized"

    recover_cameras_from_mat(F_file, init_file, normMat_file, output_file; method=method)
end


 main(ARGS)
# F_filename = "F.mat"
# normFile = "normMat.mat"
# init_filename = "Ps_init.mat"
# output_file = "Ps_iterative.mat"

# file = MAT.matopen(F_filename)
# F = read(file, "FN")
# close(file)
# F_multiview = cameras_from_F.wrap(F)

# norm_file = MAT.matopen(normFile)
# N = read(norm_file, "normMat")
# close(file)

# init_file = MAT.matopen(init_filename)
# P_init =read(init_file, "Ps")
# close(init_file)

# # P_init = cameras_from_F.Cameras{Float64}([cameras_from_F.Camera{Float64}(inv(N[3*i-2:3*i, 3*i-2:3*i])*P_init[i]) for i=1:size(P_init,1)]);
# P_init = cameras_from_F.Cameras{Float64}([cameras_from_F.Camera{Float64}(P_init[i]) for i=1:size(P_init,1)]);
# Ps = cameras_from_F.recover_cameras_iterative(F_multiview; P₀=P_init, recovery_method="skew_symmetric", update_init="all", update="start-centrality-update-all", set_anchor="fixed", max_iterations=200);
# # Ps_mat = [N[3*i-2:3*i, 3*i-2:3*i]*Matrix(Ps[i]) for i=1:length(Ps) ];
# Ps_mat = [Matrix(Ps[i]) for i=1:length(Ps) ];

# file = MAT.matopen(output_file, "w")
# write(file, "Ps_iterative", Ps_mat)
# close(file)

