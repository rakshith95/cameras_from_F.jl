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

function relative_projectivity( Ps::Cameras{T}, Qs::Cameras{T}; affine=false ) where T<:AbstractFloat
    if !affine
        L = zeros(1,16)
        for k=1:length(Ps)
            a = vec(Qs[k])
            L = vcat(L, (a'*a*SMatrix{12,12,T}(I) - a*a') * kron(SMatrix{4,4,T}(I),Ps[k]) )
        end
        L = L[2:end,:]
        L_svd = svd(L)
        H = SMatrix{4,4,T}(reshape( L_svd.V[:,end], 4 , 4))
    else
        H = relative_affinity(Ps, Qs)
    end
    return H
end
    
function relative_affinity(Ps::Cameras{T}, Qs::Cameras{T}) where T<:AbstractFloat
    ncams = length(Ps)
    D = zeros(8*ncams, 12)
    I₃ = SMatrix{3,3,T}(I)
    res = zeros(8*ncams)
    z₆₃ = zeros(6,3)
    z₂₉ = zeros(2,9)

    for i=1:ncams
        M1 = @views(Ps[i][1:2,1:3])
        t1 = @views(Ps[i][1:2,end])
        M2 = @views(Qs[i][1:2,1:3])
        t2 = @views(Qs[i][1:2,end])

        D[(i-1)*8 + 1: 8*i,: ] = [ [kron(I₃,M1) z₆₃]; [z₂₉ M1]]
        res[(i-1)*8 + 1: 8*i] = [vec(M2);(t2 - t1)]
    end
    h = D\res
    H = SMatrix{4,4,T}( [ [reshape(h[1:9],3,3) reshape(h[10:12],3,1)]; [zeros(1,3) 1] ] )
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

function CamsFromF_gpsfm(F_mv::AbstractSparseMatrix) 
    Eig_Fmv = eigen(Symmetric(unwrap(F_mv)); sortby = x -> -abs(x));
    n = size(F_mv,1)
    eig_vals = Eig_Fmv.values[1:6]
    eig_vecs = Eig_Fmv.vectors[:,1:6]
    ord = sortperm(eig_vals, rev=true)

    # if (count(eig_vals .> 0) > 3)
        # println(Eig_Fmv.values, "\t", rank(unwrap(F_mv); atol=1e-12))
    # end

    # Σ₁ = SMatrix{3,3,Float64}(diagm(eig_vals[ord[1:3]]))
    Σ₁ = SMatrix{3,3,Float64}(diagm(abs.(eig_vals[ord[1:3]])))
    # Σ₂ = SMatrix{3,3,Float64}(diagm(-1*eig_vals[ord[4:6]]))
    Σ₂ = SMatrix{3,3,Float64}(diagm(abs.(eig_vals[ord[4:6]])))

    X̃ = eig_vecs[:,ord[1:3]]
    Ỹ = eig_vecs[:,ord[4:6]]

    # F = [X̃ Ỹ]*[[Σ₁ zeros(3,3)];[zeros(3,3) -Σ₂]]*([X̃ Ỹ]')
    X = X̃*sqrt.(Σ₁)
    Y = Ỹ*sqrt.(Σ₂)
    # F = ((X*X') - (Y*Y'))

    V = (X-Y)/(√2)
    if (rank(V[1:3,1:3]; atol=1e-10) == 3)
        U = (X+Y)/(√2)
    else
        U = (X-Y)/(√2)
        V = (X+Y)/(√2)
    end
    

    Ps = Cameras{Float64}(repeat([Camera_canonical],n))
    for i=1:n
        Vᵢ = @views V[(i-1)*3+1:3*i,:]
        Uᵢ = @views U[(i-1)*3+1:3*i,:]
        if rank(Uᵢ;atol=1e-10) > 2
            Uᵢ_svd = svd(Uᵢ)
            Uᵢ = Uᵢ_svd.U*diagm( [Uᵢ_svd.S[1:2];0] )*Uᵢ_svd.Vt;
        end

        try
            Tᵢ = inv(Vᵢ)*Uᵢ
        catch
            println(rank(Vᵢ), "\t", rank(Uᵢ))
        end
        Tᵢ = (1/2)*( Tᵢ - Tᵢ' )
        tᵢ = SVector{3,Float64}([-Tᵢ[2,end], Tᵢ[1,end], -Tᵢ[1,2]])

        Ps[i] = Camera{Float64}([inv(Vᵢ)' -inv(Vᵢ)'*tᵢ]) 
        # Ps[i] = Ps[i]/norm(Ps[i])
    end
    return Ps
end

function get_scales(F_mv::AbstractSparseMatrix, Ps_est::Cameras{T}) where T<:AbstractFloat
    n = length(Ps_est)
    F_est = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],n,n)) 
    compute_multiviewF_from_cams!(0.0, F_est, Ps_est; F_estimation=F_from_cams_gpsfm, noise_type="angular", normalize=false)
    scales = spzeros(n,n)

    for i=1:n-1
        for j=i+1:n
            scales[i,j] = norm(F_mv[i,j])/norm(F_est[i,j])
            F_est[i,j] = scales[i,j]*F_est[i,j]
            F_est[j,i] = F_est[i,j]'
        end
    end
    return scales, F_est
end

function SVP(A::AbstractMatrix{T}, desired_rank) where T<:AbstractFloat
    n = size(A,1)
    A_svd = svd(A)
    A_proj =  A_svd.U*diagm([A_svd.S[1:desired_rank];zeros(n-desired_rank)])*A_svd.Vt;
    return A_proj
end

function iterative_scale_estimation(F_mv::AbstractSparseMatrix; error=projective_synchronization.angular_distance, δ=1e-2, max_iterations=100)
    it=1
    n = size(F_mv,1)

    Ps_prev = CamsFromF_gpsfm(F_mv)
    scales, Fs_est = get_scales(F_mv, Ps_prev)
    # Fs_est = wrap(SVP(unwrap(F_mv), 6))
    Ps_est = missing
    # display(Fs_est)
    while (it<=max_iterations)
        Ps_est = CamsFromF_gpsfm(Fs_est)
        scales, Fs_est = get_scales(Fs_est, Ps_est)
        # Fs_est = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],n,n)) 
        # compute_multiviewF_from_cams!(0.0, Fs_est, Ps_est; F_estimation=F_from_cams_gpsfm, noise_type="angular", normalize=false)

        if rad2deg( mean(compute_error(Ps_prev, Ps_est, error)) ) <= δ
            break
        end
        Ps_prev = Ps_est
        it += 1
    end
    println(it)
    return Ps_est
end