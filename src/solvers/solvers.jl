function get_NullSpace_ev(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    D = Symmetric(A'*A)
    (λ, ev) = eigen(D, 1:1)
    return projective_synchronization.unit_normalize(vec(ev))
end

function get_NullSpace_svd(A::AbstractMatrix{T};full=false) where {T<:AbstractFloat}
    A_svd = svd(A, full=full)
    return A_svd.V[:, end] #  Last column of V is solution for null space problem
end

function get_Nullspace_svd_subspace(A::AbstractMatrix{T};full=true, threshold=1e-10 ) where T<:AbstractFloat
    A_svd = missing
    A_svd = svd(A,full=full)
    return A_svd.V[:, end-count(A_svd.S.<threshold)+1:end]
end

function get_Nullspace_l1(A::AbstractMatrix{T}) where T<:AbstractFloat
    # m,n = size(A);
    # f = SVector{2*m+n, T}([zeros(n,1);zeros(m,1);ones(m,1)]);
    # A_ineq = vcat( hcat(zeros(m,n) , SMatrix{m,m,T}(I), -SMatrix{m,m,T}(I) ), hcat(zeros(m,n),-SMatrix{m,m,T}(I),-SMatrix{m,m,T}(I)) );
    # b_ineq = zeros(2*m,1);
    # A_eq = [ [-A ,SMatrix{m,m,T}(I),zeros(m,m)] ; [ones(1,n),zeros(1,m),zeros(1,m)] ];
    # b_eq = [zeros(m,1);1];
    # vX = linprog(f, A_ineq, b_ineq, A_eq, b_eq);
    return MATLAB.mxcall(:SolveNSl1, 1, Matrix{Float64}(A));
end
    


function make_skew_symmetric(x::SVector{3,T}) where T
   return SMatrix{3,3,T}([[0, x[3], -x[2]] [-x[3], 0, x[1]] [x[2], -x[1], 0 ]]) 
end


function get_normalization_mat(X::Pts2D{T}; isotropic_scale=sqrt(2)) where T<:AbstractFloat
    c = mean(X)
    d = mean(norm.(X .- Ref(c)))
    s = isotropic_scale/d
    N = SMatrix{3,3,Float64}([ [1,0,0] [0,1,0] [-c[1], -c[2], 1/s] ])
    return N
end

function get_normalization_mat(X::Pts2D_homo{T}; isotropic_scale=sqrt(2)) where T<:AbstractFloat
    get_normalization_mat(euclideanize.(X); isotropic_scale=isotropic_scale)
end

function relative_projectivity( Ps::Cameras{T}, Qs::Cameras{T} ) where T<:AbstractFloat
    L = zeros(1,16)
    for k=1:length(Ps)
        a = vec(Qs[k])
        L = vcat(L, (a'*a*SMatrix{12,12,T}(I) - a*a') * kron(SMatrix{4,4,T}(I),Ps[k]) )
    end
    L = L[2:end,:]
    println(size(L), rank(L))
    L_svd = svd(L)
    H = SMatrix{4,4,T}(reshape( L_svd.V[:,end], 4 , 4))
    
    return H
end
    
function F_8pt(x::Pts2D_homo{T}, x′::Pts2D_homo{T}) where T
    F_8pt(euclideanize.(x), euclideanize.(x′))
end

function F_8pt(x::Pts2D{T}, x′::Pts2D{T}) where T
    A = ones(8,9)
    for i=1:8
        @views A[i,1:8] = [x′[i][1]*x[i][1], x′[i][1]*x[i][2], x′[i][1], x′[i][2]*x[i][1], x′[i][2]*x[i][2], x′[i][2], x[i][1], x[i][2]]
    end
    A = SMatrix{8,9,T}(A)
    U_Σ_V = svd(A, full=true)
    f = U_Σ_V.V[:,end]
    F = SMatrix{3,3,T}( transpose( reshape(f,(3,3)) ) )
    #rank 2 approximation
    F_svd = svd(F)
    D = diagm([F_svd.S[1:end-1];0])
    F = FundMat{T}(F_svd.U*D*F_svd.Vt)
    return F      
end

function F_8ptNorm(x::Pts2D{T}, x′::Pts2D{T} ) where T
    x_homo = homogenize.(x)
    x′_homo = homogenize.(x′)
    F_8ptNorm(x_homo, x′_homo)
end

function F_8ptNorm(x_homo::Pts2D_homo{T}, x′_homo::Pts2D_homo{T}) where T
    N₁ = get_normalization_mat(x_homo)
    N₂ = get_normalization_mat(x′_homo)
    x₁ = [N₁*x_homo[i] for i=1:length(x_homo)]
    x₂ = [N₂*x′_homo[i] for i=1:length(x′_homo)]

    F_norm = F_8pt(x₁, x₂)
    F = FundMat{T}(N₂'*F_norm*N₁)
    return F
end

function recover_camera_SkewSymm(Ps::Cameras{T}, Fs::FundMats{T}, wts=ones(length(Ps)), P₀=nothing) where T<:AbstractFloat
    # Given Pᵢ, and Fᵢⱼ , find Pⱼ
    # PᵢᵀFᵢⱼPⱼ is skew symmetric
    num_cams = length(Ps)
    D = zeros(10*num_cams,12)
    z = zeros(3)
    for i=1:num_cams
        A = Ps[i]'*Fs[i]'
        k = i-1
        sqrt_wᵢ = √wts[i]
        D[k*10+1, :] = @views sqrt_wᵢ*[A[1,1:3]; z; z; z]
        D[k*10+2, :] = @views sqrt_wᵢ*[z; A[2,1:3]; z; z]
        D[k*10+3, :] = @views sqrt_wᵢ*[z; z; A[3,1:3]; z]
        D[k*10+4, :] = @views sqrt_wᵢ*[z; z; z; A[4,1:3]]
        D[k*10+5, :] = @views sqrt_wᵢ*[A[2,1:3]; A[1,1:3]; z; z]
        D[k*10+6, :] = @views sqrt_wᵢ*[A[3,1:3]; z; A[1,1:3]; z]
        D[k*10+7, :] = @views sqrt_wᵢ*[A[4,1:3]; z; z; A[1,1:3]]
        D[k*10+8, :] = @views sqrt_wᵢ*[z; A[3,1:3]; A[2,1:3]; z]
        D[k*10+9, :] = @views sqrt_wᵢ*[z; A[4,1:3]; z; A[2,1:3]]
        D[k*10+10,:] = @views sqrt_wᵢ*[z; z; A[4,1:3]; A[3,1:3]]
    end

    return Camera{T}(reshape(get_NullSpace_svd(D), 3, 4))

end

function recover_camera_SkewSymm_vectorization(Ps::Cameras{T}, Fs::FundMats{T}, wts=ones(length(Ps)), P₀=nothing; l1=false) where T<:AbstractFloat
    # From section 3.1 in overleaf
    # Given Pᵢ, and Fᵢⱼ , find Pⱼ
    num_cams = length(Ps)
    D = zeros(16*num_cams, 12)
    for i=1:num_cams
        Aᵢ = ( kron( (Ps[i]'*Fs[i]') , I₄)*K₃₄) + (kron( I₄, Ps[i]'*Fs[i]' ))
        D[(i-1)*16+1:(i-1)*16+16, :] = √(wts[i])*Aᵢ
    end
    return Camera{T}(reshape(get_NullSpace_svd(D), 3, 4))
end

function eq_constaint(x::AbstractVector{T}, N) where T<:AbstractFloat
    return 1 - x'*x
end

function subspace_obj_i(x::AbstractVector, N::AbstractMatrix)
    return x - N*N'*x
end

function jacobian_fi( x::AbstractVector{T}, Nᵢ::AbstractMatrix{T}) where T<:AbstractFloat
    return SMatrix{length(x),length(x),T}(I) - Nᵢ*Nᵢ'
end

function jacobian_constraints(x::AbstractVector{T}, N) where T<:AbstractFloat
    return -2*x
end

function recover_camera_subspace_svd(Ps::Cameras{T}, Fs::FundMats{T}, wts=ones(length(Ps)), P₀=nothing) where T<:AbstractFloat
    n = length(Ps)
    D = zeros(12*n,12)
    for i=1:length(Ps)
        N = nullspace( kron( (Ps[i]'*Fs[i]') , I₄)*K₃₄ + kron( I₄, Ps[i]'*Fs[i]' ) )
        D[(i-1)*12+1:i*12,:] = √wts[i]*(SMatrix{12,12,T}(I) - N*N')
    end
    return get_NullSpace_svd(D)
end

function recover_camera_subspace_angular(Ps::Cameras{T}, Fs::FundMats{T}, wts=ones(length(Ps)), P₀=recover_camera_SkewSymm_vectorization(Ps,Fs,wts);) where T<:AbstractFloat
    # return P₀/norm(P₀)
    return subspace_angular_distance([nullspace( kron( (Ps[i]'*Fs[i]') , I₄)*K₃₄ + kron( I₄, Ps[i]'*Fs[i]' ) ) for i=1:length(Ps)], Vector{T}(vec(P₀)); wts=wts)
end

function subspace_angular_distance(N::AbstractVector, c₀::AbstractVector{T}; wts=ones(length(N)), max_iterations=1e2, δ=1e-3, σ=1e-4) where T<:AbstractFloat
    c = c₀
    projective_synchronization.unit_normalize!(c)
    it=0

    while it < max_iterations
        c_prev = c
        c = SVector{length(c_prev)}(zero(c_prev))
        for i in collect(1:length(N))
            B = N[i]*N[i]';
            Bc = B*c_prev;
            if ((c_prev'*Bc)/norm(Bc)) < 1
                # var = (2*Bc*norm(Bc) - ((c_prev'*Bc)*((B'*Bc)/norm(Bc))) )/(norm(Bc)^2)
                # c = c + wts[i]* ((1/√(1 - ((c_prev'*Bc)/norm(Bc))^2))*var)
                c = c + wts[i]* ( ( 2*norm(Bc)^2*(Bc)  - (c_prev'*Bc)*(B'*Bc)) / ( norm(Bc)^4*√(norm(Bc)^2 - (c_prev'*Bc)^2 ) ) )
            end
        end           
        if iszero(c)
            return c_prev
        end
        c = projective_synchronization.unit_normalize(c)
        it += 1
        # if norm(c-c_prev) < δ 
        if projective_synchronization.angular_distance(c,c_prev) < δ 
            break
        end
    end
    return c
end

function recover_camera_l2_unitSum(Ps::Cameras{T}, Fs::FundMats{T}, wts=ones(length(Ps)), P₀=nothing) where T<:AbstractFloat
    num_cams = length(Ps)
    D = zeros(16*num_cams, 12)
    for i=1:num_cams
        Aᵢ = ( kron( (Ps[i]'*Fs[i]') , I₄)*K₃₄) + (kron( I₄, Ps[i]'*Fs[i]' ))
        # D[(i-1)*16+1:(i-1)*16+16, :] = (√wts[i])*Aᵢ
        D[(i-1)*16+1:(i-1)*16+16, :] = (wts[i])*Aᵢ #Works empirically better without sqrt. Why?
    end

    DataMat = [ [D'*D  ones(length(Ps[1]),1)];[ones(1,length(Ps[1])) 0] ]
    obs_vec = [ones(length(Ps[1]),1);1]
    res = DataMat \ obs_vec

    return Camera{T}(reshape(res[1:end-1],3,4))
end


function recover_camera_l1_UnitSum(Ps::Cameras{T}, Fs::FundMats{T}, wts=ones(length(Ps)), P₀=nothing;) where T<:AbstractFloat
    # Given Pᵢ, and Fᵢⱼ , find Pⱼ
    num_cams = length(Ps)
    D = zeros(16*num_cams, 12)
    for i=1:num_cams
        Aᵢ = ( kron( (Ps[i]'*Fs[i]') , I₄)*K₃₄) + (kron( I₄, Ps[i]'*Fs[i]' ))
        D[(i-1)*16+1:(i-1)*16+16, :] = (wts[i])*Aᵢ
    end

    ns_opt =MATLAB.mxcall(:SolveNSl1, 1, Matrix{Float64}(D));
    projective_synchronization.unit_normalize!(ns_opt);
    return Camera{T}(reshape(ns_opt, 3, 4))
end
    
function recover_camera_l1_unitNorm_CvxCcv(Ps::Cameras{T}, Fs::FundMats{T}, wts=ones(length(Ps)), P₀=recover_camera_SkewSymm_vectorization(Ps,Fs,wts);δ=1e-3, max_iterations=5) where T<:AbstractFloat
    num_cams = length(Ps)
    D = zeros(16*num_cams, 12)
    for i=1:num_cams
        Aᵢ = ( kron( (Ps[i]'*Fs[i]') , I₄)*K₃₄) + (kron( I₄, Ps[i]'*Fs[i]' ))
        D[(i-1)*16+1:(i-1)*16+16, :] = (wts[i])*Aᵢ
    end
    x₀ = vec(P₀)
    # x_curr = missing
    x_prev = projective_synchronization.unit_normalize(x₀)
    it=0
    # while it <= max_iterations 
        # x = Convex.Variable(length(vec(Ps[1])))
        # problem = Convex.minimize(Convex.norm(D*x,1), [ Convex.norm(x) - 1 <= 0, (1 - (x_prev/norm(x_prev))'*x) <= 0 ])
        # Convex.solve!(problem, SCS.Optimizer; silent=true)
        # println(problem.status)
        # x_curr = MATLAB.mxcall(:SolveConeProg_CvxCcv, 1, Matrix{Float64}(D), Vector{Float64}(x_prev));
        # x_curr = reshape(x_curr,12)
        # if ( norm(x_prev - (x_curr/norm(x_curr)) ) <= δ )
            # break
        # end
        # x_prev = projective_synchronization.unit_normalize(x_curr)
        # it += 1
    # end
    # x_opt = x_curr;

    x_opt = MATLAB.mxcall(:SolveConeProg_CvxCcv2, 1, Matrix{Float64}(D), Vector{Float64}(x_prev), max_iterations, δ);

    return Camera{T}(reshape(x_opt, 3, 4))

end

function split(Adj::AbstractSparseMatrix; num_partitions=round(size(Adj,1)/25))
    clusters = MATLAB.mat"spectralcluster($Adj, $num_partitions)";
    partitions = Vector{SparseMatrixCSC}(undef, Int(num_partitions));
    for (i,el) in enumerate(unique(clusters))
        g = findall(clusters .== el)
        G = induced_subgraph(Graph(Adj), g);
        A = adjacency_matrix(G[1]);
        # tG, t = get_triplet_cover(A, max_size=5000)
        # covered_nodes = unique(reduce(hcat,t[tG[2][1:nv(tG[1])]]))
        # nonTriplet_cams = setdiff( collect(1:size(A,1)), covered_nodes)
        C = rand(4,size(A,1))*100;
        println("Checking solvability")
        solvable = MATLAB.mxcall(:is_finite_solvable, 1, Matrix{Float64}(A), C, "eigs")
        if solvable
            partitions[i] = A
        else
            partitions[i] = spzeros(1,1)
        end
    end
    return partitions
end


function split_and_solve(Adj::AbstractSparseMatrix, F_multiview::AbstractSparseMatrix, Matches::AbstractSparseMatrix, Ps_gt::Cameras{Float64}; partitions=round(size(Adj,1)/25))
    clusters = MATLAB.mat"spectralcluster($Adj, $partitions)";
    println("Got clusters")
    nodes = 1:size(Adj,1);
    errs_gpsfm = Vector{Float64}(undef, Int(partitions))
    errs_ours = Vector{Float64}(undef, Int(partitions))
    for el in unique(clusters)
        g = findall(clusters .== el)
        G = induced_subgraph(Graph(Adj), g);
        A = adjacency_matrix(G[1]);
        # tG, t = get_triplet_cover(A, max_size=5000)
        # covered_nodes = unique(reduce(hcat,t[tG[2][1:nv(tG[1])]]))
        # nonTriplet_cams = setdiff( collect(1:size(A,1)), covered_nodes)
        println(size(A,1))
        if length(nonTriplet_cams) > 0
            C = rand(4,size(A,1))*100;
            println("Checking solvability")
            solvable = MATLAB.mxcall(:is_finite_solvable, 1, Matrix{Float64}(A), C, "eigs")
            if !solvable
                return false
            end
        end
            
        F = F_multiview[nodes .∈ Ref(g), nodes .∈ Ref(g)]
        M = Matches[nodes .∈ Ref(g), nodes .∈ Ref(g)]

        F_gpsfm = F[1:end .∉ Ref(nonTriplet_cams), 1:end .∉ Ref(nonTriplet_cams)]
        M_gpsfm = M[1:end .∉ Ref(nonTriplet_cams), 1:end .∉ Ref(nonTriplet_cams)]
        F_unwrap = unwrap(F_gpsfm);
        println("Running GPSFM")
        P_gpsfm = Cameras{Float64}(MATLAB.mxcall(:runProjective_direct, 1, F_unwrap, "gpsfm", M_gpsfm));
        println("GPSFM done")
        errs_gpsfm[Int(el)] = mean(rad2deg.(compute_error(Ps_gt[g], P_gpsfm, projective_synchronization.angular_distance)))
        
        P_init = Vector{Camera{Float64}}(repeat([Camera_canonical], size(A,1)))
        P_init[intersect(collect(1:size(A,1)), unique(reduce(hcat,t[tG[2][1:nv(tG[1])]])) )] = P_gpsfm            

        P_ours = recover_cameras_iterative(F; X₀=P_init, method="subspace_angular",  update="order-centrality-update-all");
        # P_ours, Wts = outer_irls(recover_cameras_iterative, F, P_init, "subspace_angular", compute_error, max_iter_init=15, inner_method_max_it=5, weight_function=projective_synchronization.cauchy , c=projective_synchronization.c_cauchy, max_iterations=15, δ=1e-3, δ_irls=1e-1 , update_init="all", update="order-weights-update-all", set_anchor="fixed");
        errs_ours[Int(el)] = mean(rad2deg.(compute_error(Ps_gt[g], P_ours, projective_synchronization.angular_distance)))
    end
    return errs_gpsfm, errs_ours
end