const Camera{T <: AbstractFloat}       = SMatrix{3,4, T}

const Camera_canonical = Camera{Float64}( [ [1,0,0 ] [0,1,0] [0,0,1] [0,0,0] ] );;

const AffineCamera_canonical = Camera{Float64}( [ [1,0,0] [0,1,0] [0,0,0] [0,0,1] ] );

const Cameras{T <: AbstractFloat}      = Vector{Camera{T}}

const Pt2D{T <: AbstractFloat}         =  SVector{2,T}

const Pts2D{T <: AbstractFloat}        = Vector{Pt2D{T}}

const Pt2D_homo{T <: AbstractFloat}    = SVector{3,T}

const Pts2D_homo{T <: AbstractFloat}   = Vector{Pt2D_homo{T}}

const Pt3D{T <: AbstractFloat}         =  SVector{3,T}

const Pts3D{T <: AbstractFloat}        = Vector{Pt3D{T}}

const Pt3D_homo{T <: AbstractFloat}    = SVector{4,T}

const Pts3D_homo{T <: AbstractFloat}   = Vector{Pt3D_homo{T}}

const FundMat{T <: AbstractFloat}      = SMatrix{3,3,T}

const FundMats{T <: AbstractFloat}     = Vector{FundMat{T}}

const I₄ = SMatrix{4,4,Float64}(I)

const K₃₄ = get_commutation_matrix(3,4)

struct CameraParams{T<:AbstractFloat}
    f₁::T
    f₂::T
    pp::Pt2D{T}
    α::T
    K::SMatrix{3,3,T};
end 
CameraParams(f::T, pp::Pt2D{T}, α=0.0) where T<:AbstractFloat = CameraParams{T}(f,f,pp,α, SMatrix{3,3,T}([[f,0,0];[α,f,0];[pp[1],pp[2],1]]));
CameraParams(f₁::T, f₂::T, pp::Pt2D{T}, α=0.0) where T<:AbstractFloat = CameraParams{T}(f₁,f₂,pp,α, SMatrix{3,3,T}([[f₁,0,0];[α,f₂,0];[pp[1],pp[2],1]]));

function homogenize(Pt::Pt2D{T})::Pt2D_homo{T} where T
    return Pt2D_homo{T}([Pt; 1])
end

function homogenize(Pt::Pt3D{T})::Pt3D_homo{T} where T
    return Pt3D_homo{T}([Pt;1])
end

function euclideanize(Pt_homo::Pt2D_homo{T})::Pt2D{T} where T
    return Pt2D{T}( (Pt_homo/Pt_homo[end])[1:end-1]  )
end

function euclideanize(Pt_homo::Pt3D_homo{T})::Pt3D{T} where T
    return Pt3D{T}( (Pt_homo/Pt_homo[end])[1:end-1]  )
end

function wrap!(F::SparseMatrixCSC{FundMat{T}, S}, F_unwrapped::AbstractMatrix{T}) where {T<:AbstractFloat, S<:Integer}
    n = size(F,1)
    for i=1:n
        for j=i+1:n
            # display(F_unwrapped[(i-1)*3+1:i*3, (j-1)*3+1:j*3])
            if iszero(view(F_unwrapped, (i-1)*3+1:i*3, (j-1)*3+1:j*3))
                continue
            end
            F[i,j] = FundMat{T}(@views F_unwrapped[(i-1)*3+1:i*3, (j-1)*3+1:j*3])
            F[j,i] = FundMat{T}(@views F[i,j]')
        end
    end
end

function wrap(F_unwrapped::AbstractMatrix{T}) where T<:AbstractFloat
    n = div(size(F_unwrapped,1),3)
    F = SparseMatrixCSC{FundMat{T}, Int64}(spzeros(FundMat{T},n,n))
    wrap!(F, F_unwrapped)
    return F
end

function unwrap!(F_unwrapped::AbstractMatrix{T}, F_multiview::SparseMatrixCSC{FundMat{T}, S}) where {T<:AbstractFloat, S<:Integer}
    n = size(F_multiview,1)
    for i=1:3:(n*3)-3+1
        for j=i+3:3:(n*3)-3+1
            if !iszero(F_multiview[div(i,3)+1,div(j,3)+1])
                F_unwrapped[i:i+3-1, j:j+3-1] = F_multiview[div(i,3)+1,div(j,3)+1]
                F_unwrapped[j:j+3-1, i:i+3-1] = F_unwrapped[i:i+3-1, j:j+3-1]'
            end
        end
    end
end

function unwrap(F_multiview::SparseMatrixCSC{FundMat{T}, S}) where {T<:AbstractFloat, S<:Integer}
    F_unwrapped = zeros(T, size(F_multiview,1)*3, size(F_multiview,1)*3)
    unwrap!(F_unwrapped, F_multiview)
    return F_unwrapped
end