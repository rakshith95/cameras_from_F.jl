function get_NullSpace_ev(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    D = Symmetric(A'*A)
    (λ, ev) = eigen(D, 1:1)
    return projective_synchronization.unit_normalize(vec(ev))
end

function get_NullSpace_svd(A::AbstractMatrix{T};full=false) where {T<:AbstractFloat}
    A_svd = svd(A, full=full)
    return A_svd.V[:, end] #  Last column of V is solution for null space problem
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

function recover_camera_SkewSymm(Ps::Cameras{T}, Fs::FundMats{T}, wts=ones(length(Ps)), P₀=nothing; l1=false) where T<:AbstractFloat
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

    if !l1
        return Camera{T}(reshape(get_NullSpace_svd(D), 3, 4))
    else
        return Camera{T}(reshape(l1_nullspace_irls(D, 10, δ=1e-3), 3, 4))
    end

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

function recover_cameras_gradient_descent(Ps::Cameras{T}, Fs::FundMats{T}, wts = nothing, P₀=recover_camera_SkewSymm_vectorization(Ps, Fs); λ=10, max_iterations=100, δ=1e-4) where T
    num_cams = length(Ps)
    x = vec(P₀)

    it = 0
    As = Vector{SMatrix{16,12,Float64}}(undef,num_cams)
    for i=1:num_cams
        As[i] = ( kron( (Ps[i]'*Fs[i]') , I₄)*K₃₄) + (kron( I₄, Ps[i]'*Fs[i]' ))
    end
    
    while it < max_iterations
        x_prev = x
        ∇ = zero(x_prev)
        for i=1:num_cams
            Aᵢ =  @views As[i]
            if norm(Aᵢ*x_prev) > 1e-5
                ∇ = ∇ + ((Aᵢ'*Aᵢ)*x_prev)/norm(Aᵢ*x_prev)
            end
        end
        ct=0
        while ct < 100
            x = x_prev - λ*∇
            # x = projective_synchronization.unit_normalize(x)
            prev = 0
            curr = 0
            for i=1:num_cams
                Aᵢ = @views As[i]
                curr = curr + norm(Aᵢ*x)
                prev = prev + norm(Aᵢ*x_prev)
            end
            if curr > prev
                λ = λ/2
            else
                λ = λ*2
                # println(prev,"\t",curr, "\t", l2_prev)
                break
            end
            ct+=1
        end
        it +=1
        if norm(x-x_prev)/norm(x_prev) < δ  
            break
        end
    end
    # println(it)
    return x
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

function linearized_lagrangian_optimizer(fᵢ, c, x₀::AbstractVector{T}, data::AbstractVector; wts=ones(length(data)), max_iterations=50, δ_tol=1e-6, δ_break=1e-6) where T<:AbstractFloat
    it = 0
    x_prev = copy(x₀)
    x̂ = missing
    n = length(x₀)    
    while it < max_iterations
        A = -2*x_prev # change this from specific to common
        B = zeros(n,n)
        D = zeros(n)
        for i=1:length(data)
            obj_i = fᵢ(x_prev, data[i])
            if norm(obj_i) < δ_tol
                continue
            end
            # Jᵢ = ForwardDiff.jacobian(x̂ -> fᵢ(x̂, data[i]), x_prev)
            Jᵢ = jacobian_fi(x_prev, data[i])
            B = B + wts[i]*Jᵢ'*Jᵢ
            D = D + wts[i]*Jᵢ'*obj_i
        end
        if all(B .< δ_tol)
            x̂ = x_prev
            break
        end

        # B_svd = svd(B)
        # d_inv = Matrix{Float64}(diagm(1 ./B_svd.S))
        # d_inv[d_inv .> 1e12] .= 0
        # B_inv = B_svd.V*d_inv*B_svd.U'
        B_inv = pinv(B)
        
        C = c(x_prev, data)
        λ = (C - A'*B_inv*D)/(A'*B_inv*A)
        dX = -B_inv*(D + λ*A)
        x̂ = x_prev + dX
        it += 1
        if norm(x̂ - x_prev) < δ_break
            break
        end
        x_prev = copy(x̂)
    end
    # println(it)
    return x̂
end

function recover_camera_subspace(Ps::Cameras{T}, Fs::FundMats{T}, wts=ones(length(Ps)), P₀=recover_camera_SkewSymm_vectorization(Ps,Fs, wts); max_iterations=100, δ=1e-6) where T<:AbstractFloat    
    return linearized_lagrangian_optimizer(subspace_obj_i, eq_constaint, vec(P₀), [nullspace( kron( (Ps[i]'*Fs[i]') , I₄)*K₃₄ + kron( I₄, Ps[i]'*Fs[i]' ) ) for i=1:length(Ps)]; wts=wts, max_iterations=max_iterations, δ_break=δ)
end

function recover_camera_subspace_angular(Ps::Cameras{T}, Fs::FundMats{T}, wts=ones(length(Ps)), P₀=recover_camera_SkewSymm_vectorization(Ps,Fs,wts);) where T<:AbstractFloat
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
            var1 = N[i]*N[i]'*c_prev 
            var2 = (c_prev'*var1 )
            if var2 < 1
                c = c + wts[i]*((var1)/ sqrt( 1 - var2^2))
            end
        end
        c = projective_synchronization.unit_normalize(c)
        it += 1
        if norm(c-c_prev) < δ 
            break
        end
    end
    return c
end

# P1 = Camera{Float64}(rand(3,4));
# P2 = Camera{Float64}(rand(3,4));
# P3 = Camera{Float64}(rand(3,4));
# P4 = Camera{Float64}(rand(3,4));
# P5 = Camera{Float64}(rand(3,4));
# 
# F_12_noised = noise_F_angular(0.2, P2, P1 );
# F_13_noised = noise_F_angular(0.02, P3, P1);
# F_14_noised = noise_F_angular(0.02, P4, P1);
# F_15_noised = noise_F_angular(0.02, P5, P1);
#  
# Ps = Cameras{Float64}([P2, P3, P4, P5]);
# Fs = FundMats{Float64}([F_12_noised, F_13_noised, F_14_noised, F_15_noised]);
# num_cams = 4
# for i=1:num_cams
#     Ps[i] = Ps[i]/norm(Ps[i]);
#     Fs[i] = Fs[i]/norm(Fs[i]);
# end
# D = zeros(16*num_cams, 12)
# for i=1:num_cams
    # A = ( kron( (Ps[i]'*Fs[i]') , I₄)*K₃₄) + (kron( I₄, Ps[i]'*Fs[i]' ))
    # D[(i-1)*16+1:(i-1)*16+16, :] = A
# end

# @time P1_rec_vec = recover_camera_SkewSymm_vectorization(Ps, Fs, [1, 1, 1, 1.0, 1.0]);
# @time P1_rec_subspace_l2 = Camera{Float64}(reshape(recover_camera_subspace(Ps, Fs, [1.0, 1.0, 1.0, 1.0]),3,4));
# @time P = Camera{Float64}(reshape(recover_camera_subspace_angular(Ps, Fs, [1,1,1,1,1.0]), 3,4 ));

# x = vec(P1_rec_vec);
# norm(D*x)
# for i=1:num_cams
    # print(norm(D[(i-1)*16+1:(i-1)*16+16, :]*x ),"\t")
    # println( norm(obj_i(x, D[(i-1)*16+1:(i-1)*16+16, :])) )
# end

# rad2deg(projective_synchronization.angular_distance(P1_rec_vec, P1))
# rad2deg(projective_synchronization.angular_distance(P1_rec_subspace_l2, P1))
# rad2deg(projective_synchronization.angular_distance(P, P1))
