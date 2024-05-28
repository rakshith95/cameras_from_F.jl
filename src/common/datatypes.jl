const Camera{T <: AbstractFloat}       = SMatrix{3,4, T}

const Camera_canonical = Camera{Float64}( [ [1,0,0 ] [0,1,0] [0,0,1] [0,0,0] ] )

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
