const Camera{T <: AbstractFloat}       = SMatrix{3,4, T}

const Cameras{T <: AbstractFloat}      = Vector{Camera{T}}

const Pt2D{T <: AbstractFloat}         =  SVector{2,T}

const Pts2D{T <: AbstractFloat}        = Vector{Pt2D{T}}

const Pt2D_homo{T <: AbstractFloat}    = SVector{3,T}

Base.zero(::Type{Pts2D{T}}) where T<:AbstractFloat = Pts2D{T}[]
Base.zero(::Pts2D{T}) where T<:AbstractFloat = zero(Pts2D{T})

const Pts2D_homo{T <: AbstractFloat}   = Vector{Pt2D_homo{T}}

const Pt3D{T <: AbstractFloat}         =  SVector{3,T}

const Pts3D{T <: AbstractFloat}        = Vector{Pt3D{T}}

const Pt3D_homo{T <: AbstractFloat}    = SVector{4,T}

const Pts3D_homo{T <: AbstractFloat}   = Vector{Pt3D_homo{T}}

const Point{T<:AbstractFloat} = Union{Pt2D{T}, Pt2D_homo{T}, Pt3D{T}, Pt3D_homo{T}}

const FundMat{T <: AbstractFloat}      = SMatrix{3,3,T}

const FundMats{T <: AbstractFloat}     = Vector{FundMat{T}}

const I₃ = SMatrix{3,3,Float64}(I)
const I₄ = SMatrix{4,4,Float64}(I)

const K₃₄ = get_commutation_matrix(3,4)

function AffineCamera(params::SVector{8,T}) where T<:AbstractFloat
    return Camera{T}([params[1:4]';params[5:end]';[0 0 0 1]])
end

function AffineCamera(A::AbstractMatrix{T}, t::AbstractVector{T}) where T<:AbstractFloat
    return Camera{T}( [ [ A t]; [zeros(1,3) 1]]  )
end

function vec_aff(P::Camera{T}) where T<:AbstractFloat
    return SVector{8,T}([vec(P[1:2,1:3]);vec(P[1:2,end])])
end

function homogenize(Pt::Pt2D{T})::Pt2D_homo{T} where T
    return Pt2D_homo{T}([Pt; 1])
end

function homogenize(Pt::Pt3D{T})::Pt3D_homo{T} where T
    return Pt3D_homo{T}([Pt;1])
end

function homogenize(Pt::AbstractVector{T}) where T
    return vcat(Pt,one(T))
end

function euclideanize(Pt_homo::Pt2D_homo{T})::Pt2D{T} where T
    return Pt2D{T}( (Pt_homo/Pt_homo[end])[1:end-1]  )
end

function euclideanize(Pt_homo::Pt3D_homo{T})::Pt3D{T} where T
    return Pt3D{T}( (Pt_homo/Pt_homo[end])[1:end-1]  )
end

function euclideanize(Pt_homo::AbstractVector{T}) where T
    return (Pt_homo/Pt_homo[end])[1:end-1]
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

#Maybe add keypoint_id to this struct for identity 
struct point_id{Pt_type<:Point}
    point::Pt_type
    image_id::Int
    keypoint::Int
end
point_id(pt::Pt, img::Int, kp::Int) where Pt = point_id{Pt}(pt,img, kp); 

struct keypoint_id{N<:Integer}
    keypoint::N
    image_id::N
end

struct correspondence{Pt}
    point1::Pt
    point2::Pt
end
# correspondence(pt1::Pt, pt2::Pt) where Pt = correspondence{Pt}(pt1, pt2);

const point_id2D{T<:AbstractFloat} = point_id{ Pt2D{T} }
const track2D{point_id2D} = StructArrays.StructArray{point_id2D}
const keypoints{keypoint_id} = StructArrays.StructArray{keypoint_id}

const correspondence2D{T} = correspondence{T}
const correspondences2D{T} = StructArrays.StructArray{correspondence2D{T}}

Base.zero(::Type{correspondences2D{T}}) where T = correspondences2D{T}[]
Base.zero(::correspondences2D{T}) where T = zero(correspondences2D{T})
Base.iszero(c::correspondences2D{T}) where T = return (length(c)>0 ? false : true);
