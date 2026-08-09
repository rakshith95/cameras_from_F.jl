


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

function F_pts_aff(x::Pts2D{T}, x′::Pts2D{T}) where T<:AbstractFloat
    num_pts = length(x)
    D = zeros(num_pts, 5)
    for i=1:num_pts
        D[i,:] = [x′[i]' x[i]' 1]
    end
    f = get_NullSpace_svd(D)
    return SMatrix{3,3,Float64}( [ zeros(2,2)  f[1:2]; f[3:end]'] )
end

function F_8pt(x::Pts2D_homo{T}, x′::Pts2D_homo{T}; normalize=true, affine=false) where T
    if affine
        F_pts_aff(euclideanize.(x), euclideanize.(x′))
    else
        F_8pt(euclideanize.(x), euclideanize.(x′); normalize=normalize)
    end
end

function F_8pt(x::Pts2D{T}, x′::Pts2D{T}; normalize=true) where T
    num_pts = length(x);
    A = ones(num_pts,9)
    for i=1:num_pts
        @views A[i,1:8] = [x′[i][1]*x[i][1], x′[i][1]*x[i][2], x′[i][1], x′[i][2]*x[i][1], x′[i][2]*x[i][2], x′[i][2], x[i][1], x[i][2]]
    end
    A = SMatrix{num_pts,9,T}(A)
    U_Σ_V = svd(A, full=true)
    f = U_Σ_V.V[:,end]
    F = SMatrix{3,3,T}( transpose( reshape(f,(3,3)) ) )
    #rank 2 approximation
    F_svd = svd(F)
    D = diagm([F_svd.S[1:end-1];0])
    F = FundMat{T}(F_svd.U*D*F_svd.Vt)
    return normalize ? F/norm(F) : F       
end

function F_8ptNorm(x::Pts2D{T}, x′::Pts2D{T} ; scale_normalize=true) where T
    x_homo = homogenize.(x)
    x′_homo = homogenize.(x′)
    F_8ptNorm(x_homo, x′_homo; scale_normalize=scale_normalize)
end

function F_8ptNorm(x_homo::Pts2D_homo{T}, x′_homo::Pts2D_homo{T}; scale_normalize=true) where T
    N₁ = get_normalization_mat(x_homo)
    N₂ = get_normalization_mat(x′_homo)
    F_8ptNorm(x_homo, x′_homo, N₁, N₂;scale_normalize=scale_normalize)
end

function F_8ptNorm(x_homo::Pts2D_homo{T}, x′_homo::Pts2D_homo{T}, N₁::SMatrix{3,3,T}, N₂::SMatrix{3,3,T};scale_normalize=true) where T<:AbstractFloat
    x₁ = [N₁*x for x in x_homo]
    x₂ = [N₂*x′ for x′ in x′_homo]
    F_norm = F_8pt(x₁, x₂)
    F = FundMat{T}(N₂'*F_norm*N₁)
    return scale_normalize ? F/norm(F) : F
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
    c_prev = copy(c₀)
    c = copy(c₀)
    projective_synchronization.unit_normalize!(c)
    it=0

    while it < max_iterations
        copyto!(c_prev, c)
        fill!(c,0.0)
        for i in collect(1:length(N))
            B = N[i]*N[i]';
            Bc = B*c_prev;
            if ((dot(c_prev,Bc))/norm(Bc)) < 1
                # var = (2*Bc*norm(Bc) - ((c_prev'*Bc)*((B'*Bc)/norm(Bc))) )/(norm(Bc)^2)
                # c = c + wts[i]* ((1/√(1 - ((c_prev'*Bc)/norm(Bc))^2))*var)
                tmp_vec = ( ( 2*norm(Bc)^2*(Bc)  - (c_prev'*Bc)*(B'*Bc)) / ( norm(Bc)^4*√(norm(Bc)^2 - (c_prev'*Bc)^2 ) ) )
                # c = c + wts[i]* tmp_vec
                axpy!(wts[i],tmp_vec,c)
            end
        end           
        if iszero(c)
            return c_prev
        end
        # c = projective_synchronization.unit_normalize(c)
        projective_synchronization.unit_normalize!(c)
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
        Tᵢ = inv(Vᵢ)*Uᵢ
        Tᵢ = (1/2)*( Tᵢ - Tᵢ' )
        tᵢ = SVector{3,Float64}([-Tᵢ[2,end], Tᵢ[1,end], -Tᵢ[1,2]])

        Ps[i] = Camera{Float64}([inv(Vᵢ)' -inv(Vᵢ)'*tᵢ]) 
        # Ps[i] = Ps[i]/norm(Ps[i])
    end
    return Ps
end

function SVP(A::AbstractMatrix{T}, desired_rank) where T<:AbstractFloat
    n = size(A,1)
    A_svd = svd(A)
    A_proj =  A_svd.U*diagm([A_svd.S[1:desired_rank];zeros(n-desired_rank)])*A_svd.Vt;
    return A_proj
end

function triplet_coincidence_cost(X::AbstractVector{T}, Fs::FundMats{T2}) where {T, T2<:AbstractFloat}
    # X: 3*7 + 2*3n vector of points: [x11, x21,...,xn1, x12, x22,....,xn2, x13,x23,...xn3] 
    # Assume that every point in an image has a correspondence in the other 2 images 
    # Fs: {F21, F31, F32}
    
    # Output: 2*3n vector of error residuals: [ err(x11, F21,x12), err(x11, F31,x13), err(x21, F21,x23),...,err(xn1, F21,xn2),err(xn1, F31,xn3)]
    
    # First 3*7 params are for F21,F31, and F32 respectively. 
    # parameterization from Sweeney et al. (2015)

    nPts = div(length(X) - 7*length(Fs),6)

    Fji = Fs[1];
    Fij = Fji';
    
    Fki = Fs[2];
    Fik = Fki';
    
    Fkj = Fs[3];
    Fjk = Fkj';

    # d = Vector{T}(undef, 3)
    E = Vector{T}(undef, 2*3*n)
    Y = @view X[7*length(Fs)+1:end]

    for i=1:nPts
        xᵢ = view(Y, (i-1)*2+1:2*i )
        xᵢ_hom = homogenize(xᵢ)

        xⱼ = view(Y,  2*n .+ ((i-1)*2+1:2*i) )
        xⱼ_hom = homogenize(xⱼ)

        xₖ = view(Y,  4*n .+ ((i-1)*2+1:2*i) )
        xₖ_hom = homogenize(xₖ)

        E[(i-1)*2+1] = norm( (Fji'*xⱼ_hom*xⱼ_hom'*Fji)/(xⱼ_hom'*Fji*Fji'*xⱼ_hom)*xᵢ_hom );
        E[2*i] = norm( (Fki'*xₖ_hom*xₖ_hom'*Fki)/(xₖ_hom'*Fki*Fki'*xₖ_hom)*xᵢ_hom );

        E[2*n + (i-1)*2+1] = norm( (Fij'*xᵢ_hom*xᵢ_hom'*Fij)/(xᵢ_hom'*Fij*Fij'*xᵢ_hom)*xⱼ_hom );
        E[2*n + 2*i] = norm( (Fkj'*xₖ_hom*xₖ_hom'*Fkj)/(xₖ_hom'*Fkj*Fkj'*xₖ_hom)*xⱼ_hom );

        E[4*n + (i-1)*2+1] = norm( (Fik'*xᵢ_hom*xᵢ_hom'*Fik)/(xᵢ_hom'*Fik*Fik'*xᵢ_hom)*xₖ_hom );
        E[4*n + 2*i] = norm( (Fjk'*xⱼ_hom*xⱼ_hom'*Fjk)/(xⱼ_hom'*Fjk*Fjk'*xⱼ_hom)*xₖ_hom );
    end
    return E
end


function triplet_coincidence_angle(X::AbstractVector{T}, Fs::FundMats{T2}) where {T, T2<:AbstractFloat}
    # X: 2*3n vector of points: [x11, x21,...,xn1, x12, x22,....,xn2, x13,x23,...xn3] 
    # Assume that every point in an image has a correspondence in the other 2 images 
    # Fs: {F21, F31, F32}

    # Output: 2*3n vector of error residuals: [ err(x11, F21,x12), err(x11, F31,x13), err(x21, F21,x23),...,err(xn1, F21,xn2),err(xn1, F31,xn3)]
    
    n = div(length(X),6)
    Fji = Fs[1];
    Fij = Fji';
    
    Fki = Fs[2];
    Fik = Fki';
    
    Fkj = Fs[3];
    Fjk = Fkj';

    # d = Vector{T}(undef, 3)
    E = Vector{T}(undef, 2*3*n) 

    for i=1:n
        xᵢ = view(X, (i-1)*2+1:2*i )
        xᵢ_hom = homogenize(xᵢ)
        
        xⱼ = view(X,  2*n .+ ((i-1)*2+1:2*i) )
        xⱼ_hom = homogenize(xⱼ)
    
        xₖ = view(X,  4*n .+ ((i-1)*2+1:2*i) )
        xₖ_hom = homogenize(xₖ)

        E[(i-1)*2+1] = acos(clamp(dot( xᵢ_hom , (I₃ - (Fji'*xⱼ_hom*xⱼ_hom'*Fji)/(dot(Fji'*xⱼ_hom, Fji'*xⱼ_hom)) )*xᵢ_hom ) / (norm(xᵢ_hom)*norm((I₃ - (Fji'*xⱼ_hom*xⱼ_hom'*Fji)/(dot(Fji'*xⱼ_hom, Fji'*xⱼ_hom)) )*xᵢ_hom) ),-1,1) );
        E[2*i] = acos( clamp(dot( xᵢ_hom , (I₃ - (Fki'*xₖ_hom*xₖ_hom'*Fki)/(dot(Fki'*xₖ_hom, Fki'*xₖ_hom)) )*xᵢ_hom ) / (norm(xᵢ_hom)*norm((I₃ - (Fki'*xₖ_hom*xₖ_hom'*Fki)/(dot(Fki'*xₖ_hom, Fki'*xₖ_hom)) )*xᵢ_hom)),-1,1) );

        E[2*n + (i-1)*2+1] = acos(clamp( dot( xⱼ_hom , (I₃ - (Fji*xᵢ_hom*xᵢ_hom'*Fji')/(dot(Fji*xᵢ_hom, Fji*xᵢ_hom)))*xⱼ_hom ) / (norm(xⱼ_hom)*norm((I₃ - (Fji*xᵢ_hom*xᵢ_hom'*Fji')/(dot(Fji*xᵢ_hom, Fji*xᵢ_hom)))*xⱼ_hom) ),-1,1) ) ;
        E[2*n + 2*i] = acos(clamp( dot( xⱼ_hom , (I₃ - (Fkj'*xₖ_hom*xₖ_hom'*Fkj)/(dot(Fkj'*xₖ_hom, Fkj'*xₖ_hom)) )*xⱼ_hom ) / (norm(xⱼ_hom)*norm((I₃ - (Fkj'*xₖ_hom*xₖ_hom'*Fkj)/(dot(Fkj'*xₖ_hom, Fkj'*xₖ_hom)) )*xⱼ_hom)), -1,1) );

        E[4*n + (i-1)*2+1] = acos(clamp( dot( xₖ_hom , (I₃ - (Fki*xᵢ_hom*xᵢ_hom'*Fki')/(dot(Fki*xᵢ_hom, Fki*xᵢ_hom)) )*xₖ_hom ) / (norm(xₖ_hom)*norm((I₃ - (Fki*xᵢ_hom*xᵢ_hom'*Fki')/(dot(Fki*xᵢ_hom, Fki*xᵢ_hom)) )*xₖ_hom)), -1,1) ); 
        E[4*n + 2*i] = acos(clamp( dot( xₖ_hom , (I₃ - (Fkj*xⱼ_hom*xⱼ_hom'*Fkj')/(dot(Fkj*xⱼ_hom, Fkj*xⱼ_hom)) )*xₖ_hom ) / (norm(xₖ_hom)*norm((I₃ - (Fkj*xⱼ_hom*xⱼ_hom'*Fkj')/(dot(Fkj*xⱼ_hom, Fkj*xⱼ_hom)) )*xₖ_hom)), -1,1) );
    end
    return E
end

function point_dist_cost(X::AbstractVector{T}, X₀::Vector{Tf}) where {T,Tf<:AbstractFloat}
    nPts = div(length(X),6)
    E = Vector{T}(undef, 6*nPts)

    for i=1:nPts
        xᵢ = @view X[   (i-1)*2+1   :   i*2]
        x₀i = @view X₀[ (i-1)*2+1   :   i*2  ]
        
        xⱼ = @view X[2*nPts     .+ ((i-1)*2+1:i*2)]
        x₀j = @view X₀[2*nPts   .+ ((i-1)*2+1: i*2)  ]

        xₖ = @view  X[4*nPts   .+ ((i-1)*2+1 : i*2)]
        x₀k = @view X₀[4*nPts   .+ ((i-1)*2+1 : i*2)]

        E[(i-1)*2+1 : i*2] = xᵢ - x₀i
        E[2*nPts   .+ ((i-1)*2+1: i*2)] = xⱼ - x₀j        
        E[4*nPts   .+ ((i-1)*2+1 : i*2)] = xₖ - x₀k
    end
    # E = X-X₀
    return E
end


# Homography averaging sort of, for mosaics 