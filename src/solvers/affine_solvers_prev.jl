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

function AffineCams_from_F(F_multiview::AbstractSparseMatrix, wts=ones(nnz(triu(F_multiview))); params=missing, lad=false) 
    # TODO:Implement with Kronecker product
    ncams = size(F_multiview,1);
    if ncams < 5
        Ps = SizedVector{ncams, Camera{Float64}}(repeat([AffineCamera_canonical], ncams))
    else
        Ps = Vector{Camera{Float64}}(repeat([AffineCamera_canonical], ncams))
    end

    # nrows = binomial(ncams,2);
    nrows = nnz(triu(F_multiview));
    D = zeros(4*nrows, 8*ncams - 12);
    res = zeros(4*nrows);
    if ismissing(params)
        rand_params = rand(4);
    else
        rand_params = params;
    end
    ct = 1;
    for i=1:2
        for j=i+1:ncams
            
            w_sqrt = √wts[ct]

            Fji = @views F_multiview[j,i];
            if iszero(Fji)
                continue
            end
            a,b = @views Fji[1:2,end]
            c,d = @views Fji[end,1:2]
            e   = @views Fji[end,end]
            if i==1
                if j==2
                    D[(ct*4)-3, 1] = w_sqrt*b
                    res[(ct-1)*4 + 1] = w_sqrt*(-c - rand_params[1]*a)

                    D[(ct*4)-2, 2] = w_sqrt*b
                    res[(ct-1)*4 + 2] = w_sqrt*(-d - rand_params[2]*a)

                    D[(ct*4) - 1, 3] = w_sqrt*b
                    res[ (ct-1)*4 + 3 ] = w_sqrt*(-rand_params[3]*a)

                    D[ct*4 , 4] = w_sqrt*b
                    res[ct*4] = w_sqrt*(-rand_params[4]*a - e)
                else
                    D[(ct*4)-3, (j-1)*8 + 1 - 12: (j-1)*8 + 2 - 12] = w_sqrt*[a b]
                    res[(ct-1)*4 + 1] = w_sqrt*-c

                    D[(ct*4)-2, (j-1)*8 + 3 - 12: (j-1)*8 + 4 - 12] = w_sqrt*[a b]
                    res[(ct-1)*4 + 2] = w_sqrt*-d 

                    D[(ct*4) - 1, (j-1)*8 + 5 - 12: (j-1)*8 + 6 - 12] = w_sqrt*[a,b]
                    # res[(ct-1)*4 + 3] = 0
                
                    D[ct*4, (j-1)*8 + 7 - 12: (j-1)*8 + 8 - 12] = w_sqrt*[a b];
                    res[ct*4] = w_sqrt*-e
                end
            end
            if (i==2)
                D[ct*4-3, 1] =  w_sqrt*d;
                D[ct*4-3, (j-1)*8 + 1 - 12: (j-1)*8 + 2 - 12] = w_sqrt*[a b];
                res[ (ct-1)*4 + 1] = w_sqrt*(-rand_params[1]*c) 
        
                D[ct*4-2, 2] =  w_sqrt*d;
                D[ct*4-2, (j-1)*8 + 3 - 12: (j-1)*8 + 4 - 12] = w_sqrt*[a b];
                res[ (ct-1)*4 + 2] = w_sqrt*(-rand_params[2]*c)
                
                D[ct*4-1, 3] =  w_sqrt*d;
                D[ct*4-1, (j-1)*8 + 5 - 12: (j-1)*8 + 6 - 12] = w_sqrt*[a b];
                res[ (ct-1)*4 + 3] = w_sqrt*(-rand_params[3]*c)
        
                D[ct*4, 4] =  w_sqrt*d;
                D[ct*4, (j-1)*8 + 7 - 12: (j-1)*8 + 8 - 12] = w_sqrt*[a b];
                res[ (ct-1)*4 + 4] = w_sqrt*(-rand_params[4]*c - e)
            end
            ct += 1
        end
    end

    for i=3:ncams-1 
        for j=i+1:ncams
            Fji = @views F_multiview[j,i];
            if iszero(Fji)
                continue
            end
            w_sqrt = √wts[ct]
            
            a,b = @views Fji[1:2,end]
            c,d = @views Fji[end,1:2]
            e   = @views Fji[end,end]
            D[ct*4-3, (i-1)*8 + 1 - 12: (i-1)*8 + 2 - 12] = w_sqrt*[c d]; 
            D[ct*4-3, (j-1)*8 + 1 - 12: (j-1)*8 + 2 - 12] = w_sqrt*[a b]; 
            
            D[ct*4-2, (i-1)*8 + 3 - 12: (i-1)*8 + 4 - 12] = w_sqrt*[c d]; 
            D[ct*4-2, (j-1)*8 + 3 - 12: (j-1)*8 + 4 - 12] = w_sqrt*[a b]; 
    
            D[ct*4-1, (i-1)*8 + 5 - 12: (i-1)*8 + 6 - 12] = w_sqrt*[c d]; 
            D[ct*4-1, (j-1)*8 + 5 - 12: (j-1)*8 + 6 - 12] = w_sqrt*[a b]; 
    
            D[ct*4-0, (i-1)*8 + 7 - 12: (i-1)*8 + 8 - 12] = w_sqrt*[c d]; 
            D[ct*4-0, (j-1)*8 + 7 - 12: (j-1)*8 + 8 - 12] = w_sqrt*[a b]; 
            res[ct*4] = w_sqrt*-e;
            ct += 1;
        end
    end
    if lad
        return SolveLAD_LP(D, res, ncams; params_vec=rand_params)
    end
    # @time cams_vec = D\res;
    cams_vec = inv(D'*D)*D'*res; #Why is this faster than above?

    Ps[2] = AffineCamera(SVector{8,Float64}( [rand_params; cams_vec[1:4]] ))
    for i=3:ncams
        Ps[i] = AffineCamera(reshape(cams_vec[(i-1)*8+1 - 12:(i-1)*8+6-12],2,3) , cams_vec[i*8 - 1 - 12:i*8 - 12]  )
    end
    return Ps;
end

function solvability_affine(A::AbstractSparseMatrix)
    solvable = false;
    ncams = size(A,1)
    rand_cams = Cameras{Float64}(repeat([Camera(zeros(3,4))], ncams))
    create_cameras!(rand_cams; affine=true)
    F_mv = SparseMatrixCSC{FundMat{Float64}, Int64}(repeat([FundMat(zeros(3,3))],ncams,ncams))
    compute_multiviewF_from_cams!(0.0, F_mv, rand_cams)
    F_mv = SparseMatrixCSC{FundMat{Float64}, Int64}(F_mv .* A)
    # nrows = binomial(ncams,2);
    nrows = nnz(triu(A));
    D = zeros(4*nrows + 12, 8*ncams);
    ct = 1;
    for i=1:ncams-1 
        for j=i+1:ncams
            Fji = @views F_mv[j,i];
            if iszero(Fji)
                continue
            end
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

    if rank(D'*D; atol=1e-12) == size(D,2)
        solvable = true
    end
    return solvable
end
import HiGHS
function SolveLAD_LP(A::Matrix{T}, b::AbstractVector{T}, ncams::Integer; params_vec=rand(4)) where T<:AbstractFloat
    # solve min: || Ax - b ||₁  as a linear program
    Ps = Vector{Camera{Float64}}(repeat([AffineCamera_canonical], ncams))
    nArows, nAcols = size(A) 

    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.@variable(model, x[1:nAcols])
    JuMP.@variable(model, t[1:nArows])
    JuMP.@objective(model, Min, ones(nArows)'*t)
    JuMP.@constraint(model, (A*x - b) - t  .<= zeros(nArows))
    JuMP.@constraint(model, -(A*x - b) - t .<= zeros(nArows))
    JuMP.optimize!(model)
    x = JuMP.value.(x)
    Ps[2] = AffineCamera(SVector{8,Float64}( [params_vec; x[1:4]] ))
    for i=3:ncams
        Ps[i] = AffineCamera(reshape(x[(i-1)*8+1 - 12:(i-1)*8+6-12],2,3) , x[i*8 - 1 - 12:i*8 - 12]  )
    end
    return Ps
end