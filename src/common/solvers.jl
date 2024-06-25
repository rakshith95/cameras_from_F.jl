function get_NullSpace_ev(A::SMatrix{M,N,T}) where {M, N, T<:AbstractFloat}
    D = Symmetric(A'*A)
    (λ, ev) = eigen(D, 1:1)
    return projective_synchronization.unit_normalize(vec(ev))
end

function make_skew_symmetric(x::SVector{3,T}) where T
   return SMatrix{3,3,T}([[0, x[3], -x[2]] [-x[3], 0, x[1]] [x[2], -x[1], 0 ]]) 
end

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
        a =  vec(Qs[k])
        L = vcat(L, (a'*a*SMatrix{12,12,T}(I) - a*a') * kron(SMatrix{4,4,T}(I),Ps[k]) )
    end
    L = L[2:end,:]
    L_svd = svd(L)
    H = SMatrix{4,4,T}(reshape( L_svd.V[:,end], 4 , 4))
    
    return H
end

function relative_projectivity_sphere(Ps::Cameras{T}, Qs::Cameras{T}) where T<:AbstractFloat
    num_cams = length(Ps)
    L = zeros(16, num_cams)

    for i=1:num_cams
        L[:,i] = kron(SMatrix{4,4,T}(I), Ps[i]')*vec(Qs[i])
    end
    for j=2:size(L,2)
        if dot(view(L,:,1),view(L,:,j)) < 0
            L[:,j] = -L[:,j]
        end
    end

    c = projective_synchronization.spherical_mean(L)
    return SMatrix{4,4,T}(reshape(c, 4, 4))
end
    
function relative_projectivity_analytic(Ps::Cameras{T}, Qs::Cameras{T}) where T<:AbstractFloat
    num_cams = length(Ps)
    D = zeros(12*num_cams,16+num_cams)
    z = zeros(4)
    for i=1:num_cams
        k=i-1   

        D[k*12+1, 1:16] = [ Ps[i][1,:]; z; z; z]
        D[k*12+1, 16+i] = -Qs[i][1,1]

        D[k*12+2, 1:16] = [ z; Ps[i][1,:]; z; z]
        D[k*12+2, 16+i] = -Qs[i][1,2]

        D[k*12+3, 1:16] = [ z; z; Ps[i][1,:]; z]
        D[k*12+3, 16+i] = -Qs[i][1,3]

        D[k*12+4, 1:16] = [ z; z; z; Ps[i][1,:]]
        D[k*12+4, 16+i] = -Qs[i][1,4]

        D[k*12+5, 1:16] = [ Ps[i][2,:]; z; z; z]
        D[k*12+5, 16+i] = -Qs[i][2,1]

        D[k*12+6, 1:16] = [ z; Ps[i][2,:]; z; z]
        D[k*12+6, 16+i] = -Qs[i][2,2]
        
        D[k*12+7, 1:16] = [ z; z; Ps[i][2,:]; z]
        D[k*12+7, 16+i] = -Qs[i][2,3]

        D[k*12+8, 1:16] = [ z; z; z; Ps[i][2,:]]
        D[k*12+8, 16+i] = -Qs[i][2,4]

        D[k*12+9, 1:16] = [Ps[i][3,:]; z; z; z]
        D[k*12+9, 16+i] = -Qs[i][3,1]

        D[k*12+10, 1:16] = [ z; Ps[i][3,:]; z; z]
        D[k*12+10, 16+i] = -Qs[i][3,2]

        D[k*12+11, 1:16] = [ z; z; Ps[i][3,:]; z]
        D[k*12+11, 16+i] = -Qs[i][3,3]

        D[k*12+12, 1:16] = [ z; z; z; Ps[i][3,:]]
        D[k*12+12, 16+i] = -Qs[i][3,4]
    end
    D_svd = svd(D, full=true)
    H = SMatrix{4,4,T}(reshape(D_svd.V[1:16, end], 4, 4)) #  Last column of V is solution for null space problem
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

function recover_camera_SkewSymm(Ps::Cameras{T}, Fs::FundMats{T};) where T<:AbstractFloat
    # Given Pᵢ, and Fᵢⱼ , find Pⱼ
    # PᵢᵀFᵢⱼPⱼ is skew symmetric
    num_cams = length(Ps)
    D = zeros(10*num_cams,12)
    z = zeros(3)
    for i=1:num_cams
        A = Ps[i]'*Fs[i]'
        k = i-1
        D[k*10+1, :] = [A[1,1:3]; z; z; z]
        D[k*10+2, :] = [z; A[2,1:3]; z; z]
        D[k*10+3, :] = [z; z; A[3,1:3]; z]
        D[k*10+4, :] = [z; z; z; A[4,1:3]]
        D[k*10+5, :] = [A[2,1:3]; A[1,1:3]; z; z]
        D[k*10+6, :] = [A[3,1:3]; z; A[1,1:3]; z]
        D[k*10+7, :] = [A[4,1:3]; z; z; A[1,1:3]]
        D[k*10+8, :] = [z; A[3,1:3]; A[2,1:3]; z]
        D[k*10+9, :] = [z; A[4,1:3]; z; A[2,1:3]]
        D[k*10+10,:] = [z; z; A[4,1:3]; A[3,1:3]]
    end

    D_svd = svd(D, full=true)
    # display(D*D_svd.V[:, end])
    Pⱼ = Camera{T}(reshape(D_svd.V[:, end], 3, 4)) #  Last column of V is solution for null space problem
    # nullV = get_NullSpace_ev(SMatrix{size(D)...,T}(D))
    # Pⱼ = Camera{T}(reshape(nullV, 3,4))
    return Pⱼ
end

function recover_camera_SkewSymm_vectorization(Ps::Cameras{T}, Fs::FundMats{T}) where T<:AbstractFloat
    # From section 3.1 in overleaf
    # Given Pᵢ, and Fᵢⱼ , find Pⱼ
 
    num_cams = length(Ps)
    I₄ = SMatrix{4,4,T}(I)
    D = zeros(16*num_cams, 12)
    K₃₄ = get_commutation_matrix(3,4)
    for i=1:num_cams
        D[(i-1)*16+1:(i-1)*16+16, :] = ( kron( (Ps[i]'*Fs[i]') , I₄)*K₃₄) + (kron( I₄, Ps[i]'*Fs[i]' ))
    end
    D_svd = svd(D)
    Pⱼ = Camera{T}(reshape(D_svd.V[:, end], 3, 4)) #  Last column of V is solution for null space problem
    return Pⱼ
end

function recover_camera_PseudoInverse_withScales(Ps::Cameras{T}, Fs::FundMats{T}) where T<:AbstractFloat
    # Needs at least 3 neighbors
    num_cams = length(Ps)
    D = zeros(9*num_cams,12+num_cams)
    for i=1:num_cams
        k=i-1   
        F_svd = svd(Fs[i])
        P_svd = svd(Ps[i], full=true)

        e = F_svd.U[:,end] #left null space of F         
        Q = P_svd.V * vcat(diagm(1 ./P_svd.S), zeros(3)') *P_svd.U' # pinv(P)

        D[k*9+1, 1:12] = [0, -e[3]*Q[1,1], e[2]*Q[1,1], 0, -e[3]*Q[2,1], e[2]*Q[2,1], 0, -e[3]*Q[3,1], e[2]*Q[3,1], 0, -e[3]*Q[4,1], e[2]*Q[4,1]]
        D[k*9+1, 12+i] = -Fs[i][1,1]

        D[k*9+2, 1:12] = [0, -e[3]*Q[1,2], e[2]*Q[1,2], 0, -e[3]*Q[2,2], e[2]*Q[2,2], 0, -e[3]*Q[3,2], e[2]*Q[3,2], 0, -e[3]*Q[4,2], e[2]*Q[4,2]]
        D[k*9+2, 12+i] = -Fs[i][1,2]

        D[k*9+3, 1:12] = [0, -e[3]*Q[1,3], e[2]*Q[1,3], 0, -e[3]*Q[2,3], e[2]*Q[2,3], 0, -e[3]*Q[3,3], e[2]*Q[3,3], 0, -e[3]*Q[4,3], e[2]*Q[4,3]]
        D[k*9+3, 12+i] = -Fs[i][1,3]

        D[k*9+4, 1:12] = [e[3]*Q[1,1], 0, -e[1]*Q[1,1], e[3]*Q[2,1], 0, -e[1]*Q[2,1], e[3]*Q[3,1], 0, -e[1]*Q[3,1], e[3]*Q[4,1], 0, -e[1]*Q[4,1]]
        D[k*9+4, 12+i] = -Fs[i][2,1]

        D[k*9+5, 1:12] = [e[3]*Q[1,2], 0, -e[1]*Q[1,2], e[3]*Q[2,2], 0, -e[1]*Q[2,2], e[3]*Q[3,2], 0, -e[1]*Q[3,2], e[3]*Q[4,2], 0, -e[1]*Q[4,2]]
        D[k*9+5, 12+i] = -Fs[i][2,2]

        D[k*9+6, 1:12] = [e[3]*Q[1,3], 0, -e[1]*Q[1,3], e[3]*Q[2,3], 0, -e[1]*Q[2,3], e[3]*Q[3,3], 0, -e[1]*Q[3,3], e[3]*Q[4,3], 0, -e[1]*Q[4,3]]
        D[k*9+6, 12+i] = -Fs[i][2,3]
        
        D[k*9+7, 1:12] = [-e[2]*Q[1,1], e[1]*Q[1,1], 0, -e[2]*Q[2,1], e[1]*Q[2,1], 0, -e[2]*Q[3,1], e[1]*Q[3,1], 0, -e[2]*Q[4,1], e[1]*Q[4,1], 0]
        D[k*9+7, 12+i] = -Fs[i][3,1]

        D[k*9+8, 1:12] = [-e[2]*Q[1,2], e[1]*Q[1,2], 0, -e[2]*Q[2,2], e[1]*Q[2,2], 0, -e[2]*Q[3,2], e[1]*Q[3,2], 0, -e[2]*Q[4,2], e[1]*Q[4,2], 0]
        D[k*9+8, 12+i] = -Fs[i][3,2]

        D[k*9+9, 1:12] = [-e[2]*Q[1,3], e[1]*Q[1,3], 0, -e[2]*Q[2,3], e[1]*Q[2,3], 0, -e[2]*Q[3,3], e[1]*Q[3,3], 0, -e[2]*Q[4,3], e[1]*Q[4,3], 0]
        D[k*9+9, 12+i] = -Fs[i][3,3]
    end
    D_svd = svd(D)
    sol = D_svd.V[:, end]
    Pⱼ = Camera{T}(reshape(sol[1:12], 3, 4)) #  Last column of V is solution for null space problem
    F_svd = svd(Fs[1])
    e = F_svd.U[:,end] #left null space of F         
    return Pⱼ
end

function recover_camera_PseudoInverse(Ps::Cameras{T}, Fs::FundMats{T}) where T<:AbstractFloat
    # Find Pⱼ, given Pᵢ, Fⱼᵢ
    num_cams = length(Ps)
    I₉ = SMatrix{9,9,T}(I)
    D = zeros(9*num_cams, 12)

    for i=1:num_cams
        f = vec(Fs[i])
        F_svd = svd(Fs[i])
        P_svd = svd(Ps[i], full=true)
        P_inv = P_svd.V * vcat(diagm(1 ./P_svd.S), zeros(3)') *P_svd.U' # pinv(P)
        e = F_svd.U[:,end] #left null space of F 
        D[(i-1)*9+1:(i-1)*9+9 , :] = (I₉ - (f*f') / (f'*f))*( kron(P_inv', make_skew_symmetric(e) ) ) 
        
    end
    D_svd = svd(D)
    Pⱼ = Camera{T}(reshape(D_svd.V[:, end], 3, 4)) #  Last column of V is solution for null space problem
    return Pⱼ
end

function recover_camera_sphere(Ps::Cameras{T}, Fs::FundMats{T}) where T<:AbstractFloat 
    # 3.2.3 in overleaf
    # initial_guess = 2*Vector{T}(vec(recover_camera_SkewSymm(Ps, Fs)))
    num_cams = length(Ps)
    L = zeros(12, num_cams)
    for i=1:num_cams
        F_svd = svd(Fs[i])
        P_svd = svd(Ps[i], full=true)
        P_inv = P_svd.V * vcat(diagm(1 ./P_svd.S), zeros(3)') *P_svd.U' # pinv(P)
        e = F_svd.U[:,end] #left null space of F 
        L[:,i] = (kron(P_inv, -1*make_skew_symmetric(e)))*vec(Fs[i])
    end
    for j=2:size(L,2)
        if dot(view(L,:,1),view(L,:,j)) < 0
            L[:,j] = -L[:,j]
        end
    end

    c = projective_synchronization.spherical_mean(L)
    return Camera{T}(reshape(c, 3,4))
end

function recover_camera_averaging(Ps::Cameras{T}, Fs::FundMats{T}; recover_camera=recover_camera_SkewSymm, average_fn=projective_synchronization.spherical_mean) where T<:AbstractFloat
    nc2 = collect(Combinatorics.combinations(1:length(Ps), 2))
    M = zeros(12, length(nc2))
    for (i,pair) in enumerate(nc2)
        M[:,i] = vec(recover_camera(Ps[pair], Fs[pair]) )
    end
    for j=2:size(M,2)
        if dot(view(M,:,1),view(M,:,j)) < 0
            M[:,j] = -M[:,j]
        end
    end
    c = average_fn(M)
    return Camera{T}(reshape(c,3,4))
end


# function get_cam(H, P, Q)
    # D = kron(H', SMatrix{3,3,Float64}(I))    