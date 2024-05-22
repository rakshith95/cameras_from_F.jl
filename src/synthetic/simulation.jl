function noise_F_from_points(σ, P₁::Camera{T}, P₂::Camera{T}, r=20) where T<:AbstractFloat
    # Think about how best  to do this
    X = Pts3D{Float64}([rand(3),rand(3),rand(3),rand(3),rand(3),rand(3),rand(3),rand(3)])
    X_homo = homogenize.(X)
    x₁_homo = Pts2D_homo{Float64}([P₁*X_homo[i] + rand(Distributions.Normal(0, σ), 3) for i=1:length(X)])
    x₂_homo = Pts2D_homo{Float64}([P₂*X_homo[i] + rand(Distributions.Normal(0, σ), 3) for i=1:length(X)])
    
    F = F_8ptNorm(x₁_homo, x₂_homo)
    return F
end

function create_cameras!(cameras::Cameras, normalize)
    num_cams = size(cameras,1)
    for i=1:num_cams
        cameras[i] = Camera{Float64}(rand(3,4))
        if normalize
            cameras[i] = cameras[i]/norm(cameras[i],2)
        end
    end
end 

function F_from_cams(P₁::Camera{T}, P₂::Camera{T}) where T
    # This works only if 1st 3x3 block of cameras is non-singular
    Q₁ = @views P₁[1:3,1:3]
    Q₂ = @views P₂[1:3,1:3]
    P₂_svd = svd(P₂, full=true)
    C₂ = P₂_svd.V[:,end]
    e₁ = Pt2D_homo{Float64}(P₁*C₂)
    e₁ₓ = make_skew_symmetric(e₁)
    F = FundMat{T}(inv(Q₂)'*Q₁'*e₁ₓ)
    
    return F
end

function compute_multiviewF_from_cams!(σ, F_multiview::AbstractSparseMatrix, cams::Cameras{T}; noise_type="angular") where T<:AbstractFloat
    n = length(cams)
    for i=1:n
        for j=1:i 
            if i==j
                continue
            end
            if occursin("angular", noise_type)
                F = F_from_cams(cams[i], cams[j])
                θ = abs(rand(Distributions.Normal(0,σ)))
                F_multiview[i,j] = FundMat{T}(reshape(projective_synchronization.angular_noise(vec(F), θ), 3, 3))
            elseif occursin("points", noise_type)
                F_multiview[i,j] = noise_F_from_points(σ, cams[i], cams[j])
                # F = F_from_cams(cams[i], cams[j])
                # display(F/norm(F))
                # display(F_multiview[i,j]/norm(F_multiview[i,j]))
                # println("\n\n")
            end
            F_multiview[j,i] = F_multiview[i,j]'
        end
    end
end


function create_synthetic_environment(σ; noise_type="angular", error=projective_synchronization.angular_distance, kwargs...)
    normalize_cameras = get(kwargs, :normalize, true)
    n = get(kwargs, :num_cams, 25)
    ρ = get(kwargs, :holes_density, 0.4)
    Ρ = get(kwargs, :outliers_density, 0.0)

    cameras = Cameras{Float64}(repeat([Camera(zeros(3,4))], n))
    create_cameras!(cameras, normalize_cameras)
    F_multiview = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],n,n)) # Relative projectivities
    compute_multiviewF_from_cams!(σ, F_multiview, cameras, noise_type=noise_type)
    
    A = missing
    while true
        A = sprand(n,n, ρ)
        A[A.!=0] .= 1.0
        A = sparse(ones(n,n)) - A
        A = triu(A,1) + triu(A,1)'
        G = Graph(A)
        if Graphs.is_connected(G)
            break
        end
    end
    F_multiview = SparseMatrixCSC{FundMat{Float64}, Int64}(F_multiview .* A)
    
    return F_multiview
end
# create_synthetic_environment(0.0; holes_density=0.5, num_cams=25, noise_type="points")