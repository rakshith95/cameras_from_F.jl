function relative_affinity(Ps::Cameras{T}, Qs::Cameras{T}) where T<:AbstractFloat
    ncams = length(Ps)
    D = zeros(8*ncams, 12)
    I₃ = SMatrix{3,3,T}(I)
    res = zeros(8*ncams)
    z₆₃ = zeros(6,3)
    z₂₉ = zeros(2,9)

    for i=1:ncams
        M1 = Ps[i][1:2,1:3]
        t1 = Ps[i][1:2,end]
        M2 = Qs[i][1:2,1:3]
        t2 = Qs[i][1:2,end]

        D[(i-1)*8 + 1: 8*i,: ] = [ [kron(I₃,M1) z₆₃]; [z₂₉ M1]]
        res[(i-1)*8 + 1: 8*i] = [vec(M2);(t2 - t1)]
    end
    h = D\res
    H = SMatrix{4,4,T}( [ [reshape(h[1:9],3,3) reshape(h[10:12],3,1)]; [zeros(1,3) 1] ] )
    return H
end

function AffineCams_from_F_vectorized_alternate(F_multiview::AbstractSparseMatrix, wts=ones(nnz(triu(F_multiview))); irls=false, ambiguity_params=missing, lad=false) 
    ncams = size(F_multiview,1);
    if ncams < 5
        Ps = SizedVector{ncams, Camera{Float64}}(repeat([AffineCamera_canonical], ncams))
    else
        Ps = Vector{Camera{Float64}}(repeat([AffineCamera_canonical], ncams))
    end

    if ismissing(ambiguity_params)
        params = rand(4)
    else
        params = ambiguity_params;
    end

    nrows = nnz(triu(F_multiview));
    D = zeros(4*nrows, 8*ncams-12);
    res = zeros(4*nrows);

    ct = 1;
    I₃ = SMatrix{3,3,Float64}(I)
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
                    res[(ct-1)*4 + 1] = w_sqrt*(-c - params[1]*a)

                    D[(ct*4)-2, 2] = w_sqrt*b
                    res[(ct-1)*4 + 2] = w_sqrt*(-d - params[2]*a)

                    D[(ct*4) - 1, 3] = w_sqrt*b
                    res[ (ct-1)*4 + 3 ] = w_sqrt*(-params[3]*a)

                    D[ct*4 , 4] = w_sqrt*b
                    res[ct*4] = w_sqrt*(-params[4]*a - e)
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
                res[ (ct-1)*4 + 1] = w_sqrt*(-params[1]*c) 
        
                D[ct*4-2, 2] =  w_sqrt*d;
                D[ct*4-2, (j-1)*8 + 3 - 12: (j-1)*8 + 4 - 12] = w_sqrt*[a b];
                res[ (ct-1)*4 + 2] = w_sqrt*(-params[2]*c)
                
                D[ct*4-1, 3] =  w_sqrt*d;
                D[ct*4-1, (j-1)*8 + 5 - 12: (j-1)*8 + 6 - 12] = w_sqrt*[a b];
                res[ (ct-1)*4 + 3] = w_sqrt*(-params[3]*c)
        
                D[ct*4, 4] =  w_sqrt*d;
                D[ct*4, (j-1)*8 + 7 - 12: (j-1)*8 + 8 - 12] = w_sqrt*[a b];
                res[ (ct-1)*4 + 4] = w_sqrt*(-params[4]*c - e)
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
            v₁ = @views Fji[1:2,end]
            v₂ = @views Fji[end,1:2]
            e   = @views Fji[end,end]
            w_sqrt = √(wts[ct])


            D[ct*4-3:ct*4-1, (i-1)*8 + 1 - 12: (i-1)*8 + 6 - 12] = w_sqrt*kron(I₃, v₂') 
            D[ct*4-3:ct*4-1, (j-1)*8 + 1 - 12: (j-1)*8 + 6 - 12] = w_sqrt*kron(I₃, v₁')
            
            D[ct*4, (i-1)*8 + 7 - 12: (i-1)*8 + 8 -12] = w_sqrt*v₂'
            D[ct*4, (j-1)*8 + 7 - 12: (j-1)*8 + 8 -12] = w_sqrt*v₁' 
    
            res[ct*4] = w_sqrt*-e;
            
            ct += 1;
        end
    end

    if lad
        return SolveLAD_LP(D, res, ncams; params_vec=params, alternate=true)
    end

    if irls
        cams_vec = lsq_irls(D, res;max_it=30, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, window=4, set_last=0, extend_wts=true, regularization=false)
    else
        # cams_vec = D\res;
        
        cams_vec = inv(Symmetric(D'*D))*D'*res; #Why is this faster than above?
        
        # x = get_NullSpace_svd([D -res])
        # cams_vec = x/x[end,end]
    end

    Ps[2] = AffineCamera(SVector{8,Float64}( [params; cams_vec[1:4]] ))
    for i=3:ncams
        Ps[i] = AffineCamera(reshape(cams_vec[(i-1)*8+1 - 12:(i-1)*8+6-12],2,3) , cams_vec[i*8 - 1 - 12:i*8 - 12]  )
    end
    return Ps;
end

function AffineCams_from_F_vectorized(F_multiview::AbstractSparseMatrix, wts=ones(nnz(triu(F_multiview))); irls=false, ambiguity_params=missing, lad=false, wts_window=4, wts_set_last=12, extend_wts=true, regularize=false) 
    ncams = size(F_multiview,1);
    if ncams < 5
        Ps = SizedVector{ncams, Camera{Float64}}(repeat([AffineCamera_canonical], ncams))
    else
        Ps = Vector{Camera{Float64}}(repeat([AffineCamera_canonical], ncams))
    end

    nrows = nnz(triu(F_multiview));
    D = zeros(4*nrows+12, 8*ncams);
    
    res = zeros(4*nrows+12);

    if ismissing(ambiguity_params)
        params = [[1;0;0;1;0;0;0;0];rand(4)]
    else
        params = ambiguity_params;
    end

    ct = 1;
    I₃ = SMatrix{3,3,Float64}(I)

    for i=1:ncams-1 
        for j=i+1:ncams
            Fji = @views F_multiview[j,i];
            if iszero(Fji)
                continue
            end
            v₁ = @views Fji[1:2,end]
            v₂ = @views Fji[end,1:2]
            e   = @views Fji[end,end]
            w_sqrt = √(wts[ct])


            D[ct*4-3:ct*4-1, (i-1)*8 + 1 : (i-1)*8 + 6] = w_sqrt*kron(I₃, v₂') 
            D[ct*4-3:ct*4-1, (j-1)*8 + 1 : (j-1)*8 + 6] = w_sqrt*kron(I₃, v₁')
            
            D[ct*4, (i-1)*8 + 7 : (i-1)*8 + 8] = w_sqrt*v₂'
            D[ct*4, (j-1)*8 + 7 : (j-1)*8 + 8] = w_sqrt*v₁' 
    
            res[ct*4] = w_sqrt*-e;
            
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
    res[4*nrows+1:end] = params;
    

    # R = rand(length(res),length(res))
    # D = R*D
    # res = R*res
    
    if lad
        return SolveLAD_LP(D, res, ncams; params_vec=params)
    end
    
    if irls
        cams_vec = lsq_irls(D, res;max_it=30, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, window=wts_window, set_last=wts_set_last, extend_wts=extend_wts, regularization=regularize)
    else
        # cams_vec = D\res;
        
        cams_vec = inv(Symmetric(D'*D))*D'*res; #Why is this faster than above?
        cams_vec = cams_vec
        # println(D*(D\res))
        
        # Q,R = qr(D)
        # cams_vec = inv(R) *(Matrix{Float64}(Q)'*res)
        
        # x = get_NullSpace_svd([D -res])
        # cams_vec = x/x[end,end]
    end

    for i=1:ncams
        Ps[i] = AffineCamera(reshape(cams_vec[(i-1)*8+1:(i-1)*8+6],2,3), cams_vec[i*8 - 1:i*8] )
    end
    return Ps;
    # return Ps, D, res;

end


function AffineCams_from_F_separate(F_multiview::AbstractSparseMatrix, wts=ones(nnz(triu(F_multiview))); irls=false, params=missing)
    ncams = size(F_multiview,1);
    if ncams < 5
        Ps = SizedVector{ncams, Camera{Float64}}(repeat([AffineCamera_canonical], ncams))
    else
        Ps = Vector{Camera{Float64}}(repeat([AffineCamera_canonical], ncams))
    end
    nrows = nnz(triu(F_multiview));

    D_M = zeros(3*nrows+9, 6*ncams);
    res_M = zeros(3*nrows+9);
    # D_M = zeros(3*nrows, 6*ncams);
    # res_M = zeros(3*nrows);

    D_t = zeros(nrows+3, 2*ncams);
    res_t = zeros(nrows+3);
    # D_t = zeros(nrows, 2*ncams);
    # res_t = zeros(nrows);
    
    if ismissing(params)
        rand_params = rand(4)*100;
    else
        rand_params = params;
    end

    ct = 1;
    I₃ = SMatrix{3,3,Float64}(I)
    
    for i=1:ncams-1 
        for j=i+1:ncams
            Fji = @views F_multiview[j,i];
            if iszero(Fji)
                continue
            end
            v₁ = @views Fji[1:2,end]
            v₂ = @views Fji[end,1:2]
            e   = @views Fji[end,end]
            w_sqrt = √(wts[ct])

            D_M[(ct-1)*3+1 : ct*3, (i-1)*6+1 : i*6] = w_sqrt*kron(I₃, v₂') 
            D_M[(ct-1)*3+1 : ct*3, (j-1)*6+1 : j*6] = w_sqrt*kron(I₃, v₁')
               
            D_t[ct, (i-1)*2+1 : i*2] = w_sqrt*v₂'
            D_t[ct, (j-1)*2+1 : j*2] = w_sqrt*v₁' 
            res_t[ct] = -w_sqrt*e;
            ct += 1;
        end
    end
    for i=1:6
            D_M[3*nrows + i, i] = 1;
            if (i<=3)
                D_t[nrows + i, i] = 1;
            end
    end
    D_M[3*nrows + 7, 7] = 1;
    D_M[3*nrows + 8, 9] = 1;
    D_M[3*nrows + 9, 11] = 1;

    res_M[3*nrows+1 : 3*nrows+9] = [1;0;0;1;0;0; rand_params[1:3]]
    res_t[nrows+1 : nrows+3] = [zeros(2);rand_params[4]]

    if irls
        Ms = lsq_irls(D_M, res_M; max_it=20, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, window=3, set_last=9)
        ts = lsq_irls(D_t, res_t; max_it=20, weight_function=projective_synchronization.cauchy, c=projective_synchronization.c_cauchy, window=1, set_last=3)
    else
        # cams_vec = D\res;
        # Ms = D_M \ res_M;
        # ts = D_t \ res_t;

        Ms = inv(Symmetric(D_M'*D_M))*D_M'*res_M;
        ts = inv(Symmetric(D_t'*D_t))*D_t'*res_t;
        
        # x = get_NullSpace_svd([D -res])
        # cams_vec = x/x[end,end]
    end
    for i=1:ncams
        Ps[i] = AffineCamera(reshape(Ms[(i-1)*6+1:i*6],2,3), ts[(i-1)*2+1:i*2] )
    end
    return Ps;
end

function AffineCams_from_F(F_multiview::AbstractSparseMatrix, wts=ones(nnz(triu(F_multiview))); irls=false, params=missing, lad=false) 
    # TODO:Implement with Kronecker product
    ncams = size(F_multiview,1);
    if ncams < 5
        Ps = SizedVector{ncams, Camera{Float64}}(repeat([AffineCamera_canonical], ncams))
    else
        Ps = Vector{Camera{Float64}}(repeat([AffineCamera_canonical], ncams))
    end

    # nrows = binomial(ncams,2);
    nrows = nnz(triu(F_multiview));
    # D = zeros(4*nrows, 8*ncams - 12);
    D = zeros(4*nrows+12, 8*ncams);
    
    # res = zeros(4*nrows);
    res = zeros(4*nrows+12);

    if ismissing(params)
        rand_params = rand(4)*100;
        rand_params = rand(4);
    else
        rand_params = params;
    end
    ct = 1;
    for i=1:ncams-1 
        for j=i+1:ncams
            Fji = @views F_multiview[j,i];
            if iszero(Fji)
                continue
            end
            a,b = @views Fji[1:2,end]
            c,d = @views Fji[end,1:2]
            e   = @views Fji[end,end]
            w_sqrt = √(wts[ct])

            D[ct*4-3, (i-1)*8 + 1 : (i-1)*8 + 2 ] = w_sqrt*[c d]; 
            D[ct*4-3, (j-1)*8 + 1 : (j-1)*8 + 2 ] = w_sqrt*[a b]; 
            
            D[ct*4-2, (i-1)*8 + 3 : (i-1)*8 + 4 ] = w_sqrt*[c d]; 
            D[ct*4-2, (j-1)*8 + 3 : (j-1)*8 + 4 ] = w_sqrt*[a b]; 
    
            D[ct*4-1, (i-1)*8 + 5 : (i-1)*8 + 6 ] = w_sqrt*[c d]; 
            D[ct*4-1, (j-1)*8 + 5 : (j-1)*8 + 6 ] = w_sqrt*[a b]; 
    
            D[ct*4-0, (i-1)*8 + 7 : (i-1)*8 + 8 ] = w_sqrt*[c d]; 
            D[ct*4-0, (j-1)*8 + 7 : (j-1)*8 + 8 ] = w_sqrt*[a b]; 
            res[ct*4] = -w_sqrt*e;
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

    # res[4*nrows+1:4*nrows+8] = [1;0;0;1;0;0;0;0];
    res[4*nrows+1:4*nrows+8] = rand(8);
    res[4*nrows+9:end] = rand_params*100;

    if lad
        return SolveLAD_LP(D, res, ncams; params_vec=rand_params)
    end
    
    # Condition D
    # S = diagm(1 ./  maximum.(eachrow(abs.(D))) )
    # S = diagm(1 ./  [minimum( v[v .!= 0] ) for v in eachcol(abs.(D))] )
    # println([minimum( v[v .!= 0] ) for v in eachcol(abs.(D))])
    # D′ = S*D;
    # b′ = S*res;
    # println(maximum(D′))
    # cams_vec′ = inv(Symmetric(D′'*D′))*D′'*b′; #Why is this faster than above?
    # cams_vec = cams_vec′    
    # println(cams_vec)

    if irls
        cams_vec = lsq_irls(D, res;max_it=20)
    else
        # cams_vec = D\res;
        # println(cond(D), "\t", cond(D′))
        cams_vec = inv(Symmetric(D'*D))*D'*res; #Why is this faster than above?
    
        # x = get_NullSpace_svd([D -res])
        # cams_vec = x/x[end,end]
    end     

    # Ps[2] = AffineCamera(SVector{8,Float64}( [rand_params; cams_vec[1:4]] ))
    # for i=3:ncams
    for i=1:ncams
        # Ps[i] = AffineCamera(reshape(cams_vec[(i-1)*8+1 - 12:(i-1)*8+6-12],2,3) , cams_vec[i*8 - 1 - 12:i*8 - 12]  )
        Ps[i] = AffineCamera(reshape(cams_vec[(i-1)*8+1:(i-1)*8+6],2,3), cams_vec[i*8 - 1:i*8] )
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

# import HiGHS
# function SolveLAD_LP(A::Matrix{T}, b::AbstractVector{T}, ncams::Integer; params_vec=rand(4), alternate=false) where T<:AbstractFloat
#     # solve min: || Ax - b ||₁  as a linear program
#     Ps = Vector{Camera{Float64}}(repeat([AffineCamera_canonical], ncams))
#     nArows, nAcols = size(A) 
#     model = JuMP.Model(HiGHS.Optimizer)
#     JuMP.@variable(model, x[1:nAcols])
#     JuMP.@variable(model, t[1:nArows])
#     JuMP.@objective(model, Min, ones(nArows)'*t)
#     JuMP.@constraint(model, (A*x - b) - t  .<= zeros(nArows))
#     JuMP.@constraint(model, -(A*x - b) - t .<= zeros(nArows))
#     JuMP.optimize!(model)
#     x = JuMP.value.(x)
#     if alternate
#         Ps[2] = AffineCamera(SVector{8,Float64}( [params_vec; x[1:4]] ))
#         for i=3:ncams
#             Ps[i] = AffineCamera(reshape(x[(i-1)*8+1 - 12:(i-1)*8+6-12],2,3) , x[i*8 - 1 - 12:i*8 - 12]  )
#         end
#     else
#         for i=1:ncams
#             Ps[i] = AffineCamera(reshape(x[(i-1)*8+1:(i-1)*8+6],2,3), x[i*8 - 1:i*8] )
#         end
#     end
#     return Ps
# end