@enum CcdUnits begin
    mm
    pixels
end

struct DigitalCamera{T <: AbstractFloat}
    fx::T
    fy::T
    pp::Pt2D{T}
    sensor_size::Pt2D{T}
    pixel_size::Pt2D{T}
    units::CcdUnits    
end

image_size(camera) = camera.sensor_size./camera.pixel_size
fov(camera) = 2*atan(sqrt(sum(camera.sensor_size.^2))/2,camera.fx)
hfov(camera) = 2*atan(camera.sensor_size[1]/2,camera.fx)
vfov(camera) = 2*atan(camera.sensor_size[2]/2,camera.fy)
aspect_ratio(camera) = camera.sensor_size[1]/camera.sensor_size[2]

pixel_intrinsics_cam(fx::T,fy::T,pp::AbstractVector{T},resolution::AbstractVector{T}) where T <: AbstractFloat = DigitalCamera(fx, fy,Pt2D(pp),Pt2D(resolution),Pt2D([fx/fy, 1.]),pixels)
pixel_intrinsics_cam(K::Diagonal{T},resolution::AbstractVector{T}) where T <: AbstractFloat = pixel_intrinsics_cam(K[1,1],K[2,2],K[1:2,3],resolution)


full_frame_cam(resolution::AbstractVector{T}) where T <: AbstractFloat = DigitalCamera(convert(T,35.), convert(T,35.), SVector{2,T}(18.,12.),SVector{2,T}(36.,24.), SVector{2,T}([36.,24.]./resolution),mm)
full_frame_cam(::Type{T} = Float64) where T <: AbstractFloat = full_frame_cam(SVector{2,T}(1200.,800.))

const IntrinsicsMatrix{T <: AbstractFloat} = UpperTriangular{T,SMatrix{3,3,T,9}}
IntrinsicsMatrix(K::AbstractMatrix{T}) where T <: AbstractFloat = IntrinsicsMatrix{T}(K)
IntrinsicsMatrix(f::AbstractFloat) = IntrinsicsMatrix(Diagonal([f,f,one(f)]))
IntrinsicsMatrix(f,pp) = IntrinsicsMatrix([f       zero(f)   pp[1];
                                           zero(f)      f    pp[2];
                                           zero(f) zero(f) one(f)])
IntrinsicsMatrix(fx,fy,pp) = IntrinsicsMatrix([fx       zero(fx)    pp[1];
                                               zero(fx)      fy     pp[2];
                                               zero(fx) zero(fx) one(fx)])

function IntrinsicsMatrix(camera::DigitalCamera{T}) where T<:AbstractFloat 
    resolution = image_size(camera)
    f = @. resolution*SVector{2,T}([camera.fx,camera.fy])/camera.sensor_size
    pp = @. resolution*camera.pp/camera.sensor_size
    
    return IntrinsicsMatrix(f[1],f[2],pp)
end

const Camera_canonical = Camera{Float64}( [ [1,0,0 ] [0,1,0] [0,0,1] [0,0,0] ] );;

const AffineCamera_canonical = Camera{Float64}( [ [1,0,0] [0,1,0] [0,0,0] [0,0,1] ] );

