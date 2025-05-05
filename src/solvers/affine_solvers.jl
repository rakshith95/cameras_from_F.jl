function AffineCams_from_F(F_multiview::AbstractSparseMatrix; params=missing, lad=false)
    # Implement with Kronecker product
    ncams = size(F_multiview,1);
    # Ps = Vector{Camera{Float64}}(repeat([AffineCamera_canonical], ncams))

    nrows = binomial(ncams,2);
    D = zeros(4*nrows + 12, 8*ncams);
    res = zeros(4*nrows + 12);
    if ismissing(params)
        rand_params = rand(4);
    else
        rand_params = params;
    end
    ct = 1;
    for i=1:ncams-1 
        for j=i+1:ncams
            Fji = @views F_multiview[j,i];
            a,b = @views Fji[1:2,end]
            c,d = @views Fji[end,1:2]
            e   = @views Fji[end,end]
            D[ct*4-3, (i-1)*8 + 1 : (i-1)*8 + 2 ] = [c d]; 
            D[ct*4-3, (j-1)*8 + 1 : (j-1)*8 + 2 ] = [a b]; 
            
            D[ct*4-2, (i-1)*8 + 3 : (i-1)*8 + 4 ] = [c d]; 
            D[ct*4-2, (j-1)*8 + 3 : (j-1)*8 + 4 ] = [a b]; 
    
            D[ct*4-1, (i-1)*8 + 5 : (i-1)*8 + 6 ] = [c d]; 
            D[ct*4-1, (j-1)*8 + 5 : (j-1)*8 + 6 ] = [a b]; 
    
            D[ct*4-0, (i-1)*8 + 7 : (i-1)*8 + 8 ] = [c d]; 
            D[ct*4-0, (j-1)*8 + 7 : (j-1)*8 + 8 ] = [a b]; 
            res[ct*4] = -e;
            ct += 1;
        end
    end
    for i=1:8
        D[4*nrows+i,i] = 1;
    end
    D[4*nrows+9 , 9]  = 1; 
    D[4*nrows+10, 11] = 1;
    D[4*nrows+11, 13] = 1;
    D[4*nrows+12, 15] = 1;

    res[4*nrows+1:4*nrows+8] = [1;0;0;1;0;0;0;0];
    res[4*nrows+9:end] = rand_params;
    if lad
        return SolveLAD_LP(D, res, ncams)
    end
    cams_vec = D \ res;
    i=1;
    # display(AffineCamera(reshape(cams_vec[(i-1)*8+1:(i-1)*8+6],2,3), cams_vec[i*8 - 1:i*8]))
    return Cameras{Float64}([AffineCamera(reshape(cams_vec[(i-1)*8+1:(i-1)*8+6],2,3), cams_vec[i*8 - 1:i*8]) for i=1:ncams] );
end

function solvability_affine(A::AbstractSparseMatrix)
    solvable = false;
    ncams = size(A,1)
    rand_cams = Cameras{Float64}(repeat([Camera(zeros(3,4))], ncams))
    create_cameras!(rand_cams; affine=true)
    F_mv = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],ncams,ncams))
    compute_multiviewF_from_cams!(0.0, F_mv, rand_cams)
    F_mv = SparseMatrixCSC{FundMat{Float64}, Int64}(F_mv .* A)
    
    nrows = binomial(ncams,2);
    D = zeros(4*nrows + 12, 8*ncams);
    ct = 1;
    for i=1:ncams-1 
        for j=i+1:ncams
            Fji = @views F_mv[j,i];
            a,b = @views Fji[1:2,end]
            c,d = @views Fji[end,1:2]

            D[ct*4-3, (i-1)*8 + 1 : (i-1)*8 + 2 ] = [c d]; 
            D[ct*4-3, (j-1)*8 + 1 : (j-1)*8 + 2 ] = [a b]; 
            
            D[ct*4-2, (i-1)*8 + 3 : (i-1)*8 + 4 ] = [c d]; 
            D[ct*4-2, (j-1)*8 + 3 : (j-1)*8 + 4 ] = [a b]; 
    
            D[ct*4-1, (i-1)*8 + 5 : (i-1)*8 + 6 ] = [c d]; 
            D[ct*4-1, (j-1)*8 + 5 : (j-1)*8 + 6 ] = [a b]; 
    
            D[ct*4-0, (i-1)*8 + 7 : (i-1)*8 + 8 ] = [c d]; 
            D[ct*4-0, (j-1)*8 + 7 : (j-1)*8 + 8 ] = [a b]; 
            ct += 1;
        end
    end
    for i=1:8
        D[4*nrows+i,i] = 1;
    end
    D[4*nrows+9 , 9]  = 1; 
    D[4*nrows+10, 11] = 1;
    D[4*nrows+11, 13] = 1;
    D[4*nrows+12, 15] = 1;

    if rank(D; atol=1e-12) == size(D,2)
        solvable = true
    end
    return solvable
end
import HiGHS
function SolveLAD_LP(A::Matrix{T}, b::AbstractVector{T}, ncams::Integer) where T<:AbstractFloat
    # solve min: || Ax - b ||₁  as a linear program
    nArows, nAcols = size(A) 
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.@variable(model, x[1:nAcols])
    JuMP.@variable(model, t[1:nArows])
    JuMP.@objective(model, Min, ones(nArows)'*t)
    JuMP.@constraint(model, (A*x - b) - t  .<= zeros(nArows))
    JuMP.@constraint(model, -(A*x - b) - t .<= zeros(nArows))
    JuMP.optimize!(model)
    x = JuMP.value.(x)
    return Cameras{Float64}([AffineCamera(reshape(x[(i-1)*8+1:(i-1)*8+6],2,3), x[i*8 - 1:i*8]) for i=1:ncams] )
end