import LeastSquaresOptim

function vec_to_R(x::SVector{3,T}) where T
    x_skew = make_skew_symmetric(x)
    R = SMatrix{3,3,T}(exp(x_skew))
    return R
end

function apply_update(f::AbstractVector{T}, F_svd::StaticArrays.SVD{S}) where {S,T}
    Rᵤ = vec_to_R(SVector{3,T}(f[1:3]))
    Rᵥ = vec_to_R(SVector{3,T}(f[4:6]))
    σ = f[7]
    s_new = F_svd.S[2] + σ
    return (Rᵤ*F_svd.U)*StaticArrays.Diagonal( SVector{3,T}([1.0,s_new,0]) )*(Rᵥ*F_svd.V)' 
end

function two_view_sampson_err(f::AbstractVector{T}, F₀_svd::StaticArrays.SVD{S}, keypoints::Vector{Pts2D{TF}}, Corr::correspondences2D{keypoint_id}) where {S,T,TF}
    # Make sure F₀ has svdvals of 1,σ
    Fji = apply_update(f, F₀_svd) 
    Fij = transpose(Fji)
    δ = Vector{T}();
    # kp_ids = Corr.keypoint
    # img_ids = Corr.image_id
    xᵢ = Vector{TF}([0,0,1]);
    xⱼ = Vector{TF}([0,0,1]);
    J = zeros(T,4)

    for (point1,point2) in zip(Corr.point1,Corr.point2)
        xᵢ[1:2] = keypoints[point1.image_id][point1.keypoint]
        xⱼ[1:2] = keypoints[point2.image_id][point2.keypoint]
        J[1] = dot(xⱼ, Fji[:,1])
        J[2] = dot(xⱼ, Fji[:,2])
    
        J[3] = dot(xᵢ, Fij[:,1])
        J[4] = dot(xᵢ, Fij[:,2])
        C = xⱼ'*Fji*xᵢ
        # Eₛ += (C^2)/dot(J,J)
        append!(δ, -(C/dot(J,J))*J )
    end
    return δ
end

function two_view_sampson_err(F::SMatrix{3,3,T}, pts1::Pts2D{T2}, pts2::Pts2D{T2}) where {T, T2<:AbstractFloat} 
    e = Vector{T}()
    xᵢ = Vector{T2}([0,0,1]);
    xⱼ = Vector{T2}([0,0,1]);
    J = zeros(T,4)

    @assert length(pts1) == length(pts2)
    for i in eachindex(pts1)
        xᵢ[1:2] = pts1[i];
        xⱼ[1:2]  = pts2[i]
        J[1] = dot(xⱼ, F[:,1])
        J[2] = dot(xⱼ, F[:,2])
        J[3] = dot(xᵢ, transpose(F)[:,1])
        J[4] = dot(xᵢ, transpose(F)[:,2])
        C = xⱼ'*F*xᵢ
        append!(e, -(C/dot(J,J))*J )
    end
    return e
end

function refineF_pairwise!(F_mult::AbstractSparseMatrix{FundMat{T}}, keypoints::Vector{Pts2D{T}}, CorresMat::AbstractSparseMatrix{ correspondences2D{keypoint_id}}; max_its=1000 ) where {T<:AbstractFloat}
    nCams = size(F_mult,1);
    fUpdate_init = zeros(T, 7);
    for i=1:nCams-1
        for j=i+1:nCams
            if iszero(F_mult[i,j])
                continue
            end
            F_mult[j,i] = F_mult[j,i]/svdvals(F_mult[j,i])[1] # Set 1st singular val to 1
            F_svd = svd(F_mult[j,i])
            Corr = CorresMat[i,j]
            opt = LeastSquaresOptim.optimize( x->two_view_sampson_err(x, F_svd, keypoints, Corr), fUpdate_init, LeastSquaresOptim.LevenbergMarquardt(), autodiff=:forward, iterations=max_its)
            # println(opt.converged," ", opt.iterations)
            F_mult[j,i] = apply_update(opt.minimizer, F_svd)
            F_mult[i,j] = transpose(F_mult[j,i])
        end
    end
end