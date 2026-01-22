function create_random_fundmat()
    F = FundMat{Float64}(rand(3,3));
    f_svd = svd(F);
    return FundMat{Float64}(f_svd.U*diagm([f_svd.S[1:2];0.0])*f_svd.Vt)
end

function F_trips(F_multiview::AbstractSparseMatrix, triplet_indices::SVector{3,T}) where T<:Integer
    # Triplets of fundamental matrices Fji, Fki, and Fkj
    return FundMats{Float64}([F_multiview[triplet_indices[2],triplet_indices[1]], F_multiview[triplet_indices[3],triplet_indices[1]], F_multiview[triplet_indices[3],triplet_indices[2]]])
end

function get_cams_from_triplet_sinha(F_triplet::FundMats{T}) where T<:AbstractFloat
    # Triplets of fundamental matrices Fji, Fki, and Fkj
    Fji = F_triplet[1];
    Fki = F_triplet[2];
    Fkj = F_triplet[3];

    eji = SVector{3,T}(get_NullSpace_svd(Fji'));
    eki = SVector{3,T}(get_NullSpace_svd(Fki'));
    ekj = SVector{3,T}(get_NullSpace_svd(Fkj'));

    Pᵢ = Camera{T}([SMatrix{3,3,T}(I) zeros(3)]);
    Pⱼ = Camera{T}([ make_skew_symmetric(eji)*Fji eji]);
    Cⱼ = SVector{4,T}(nullspace(Pⱼ));

    Mₖ = SMatrix{3,4,T}([make_skew_symmetric(eki)*Fki zeros(3)]);
    D = [kron(transpose(pinv(Pⱼ)), (make_skew_symmetric(ekj)*eki))  -vec(Fkj) zeros(9); kron(Cⱼ',eki) zeros(3) -ekj];
    sol = D \ [vec(-make_skew_symmetric(ekj)*Mₖ*pinv(Pⱼ));-Mₖ*Cⱼ];
    v = sol[1:4];
    Pₖ = Mₖ + eki*transpose(v);
    Pₖ = Pₖ/norm(Pₖ)
    
    F̄kj = make_skew_symmetric(ekj)*Pₖ*pinv(Pⱼ);

    return Cameras{T}([Pᵢ, Pⱼ, Pₖ]), FundMats{T}([Fji, Fki, F̄kj])
end

function get_cams_from_triplet_sinha(F_triplet::FundMats{T}, Ps::Cameras{T}) where T<:AbstractFloat
    Fji = F_triplet[1];
    Fki = F_triplet[2];
    Fkj = F_triplet[3];
    Pᵢ = Ps[1];
    Aᵢ = Pᵢ[1:3,1:3]
    aᵢ = Pᵢ[:,end]
    
    Pⱼ = Ps[2];

    eki = SVector{3,T}(get_NullSpace_svd(Fki'));
    ekj = SVector{3,T}(get_NullSpace_svd(Fkj'));

    Cⱼ = SVector{4,T}(nullspace(Pⱼ)[:,1]);

    Mₖ = SMatrix{3,4,T}([make_skew_symmetric(eki)*Fki*Aᵢ make_skew_symmetric(eki)*Fki*aᵢ]);
    D = [kron(transpose(pinv(Pⱼ)), (make_skew_symmetric(ekj)*eki))  -vec(Fkj) zeros(9); kron(Cⱼ',eki) zeros(3) -ekj];
    sol = D \ [vec(-make_skew_symmetric(ekj)*Mₖ*pinv(Pⱼ));-Mₖ*Cⱼ];
    v = sol[1:4];
    # println(norm(D*sol - [vec(-make_skew_symmetric(ekj)*Mₖ*pinv(Pⱼ));-Mₖ*Cⱼ]))
    Pₖ = Mₖ + eki*transpose(v);
    Pₖ = Pₖ/norm(Pₖ)
    F̄kj = make_skew_symmetric(ekj)*Pₖ*pinv(Pⱼ);

    return Cameras{T}([Ps[1], Pⱼ, Pₖ]), FundMats{T}([Fji, Fki, F̄kj])
end


function get_3rd_camera_colombo(F_triplet::FundMats{T}, Ps::Cameras{T}, param_vector::SVector{4,T}) where T<:AbstractFloat
    Fji = @views F_triplet[1];
    Fki = @views F_triplet[2];
    Fkj = @views F_triplet[3];

    Pᵢ = @views Ps[1];
    ρⱼ = @views param_vector[1:3];
    σⱼ = @views param_vector[end];
    Pⱼ = @views Ps[2];

    eij = SVector{3,T}(get_NullSpace_svd(Fji));
    eji = SVector{3,T}(get_NullSpace_svd(Fji'));

    eik = SVector{3,T}(get_NullSpace_svd(Fki));
    eki = SVector{3,T}(get_NullSpace_svd(Fki'));
    
    ejk = SVector{3,T}(get_NullSpace_svd(Fkj));
    ekj = SVector{3,T}(get_NullSpace_svd(Fkj'));
    
    Q = -Fki'*make_skew_symmetric(eki)*Fkj*make_skew_symmetric(eji)*Fji

    lᵢ = make_skew_symmetric(eij)*eik
    lᵢ = lᵢ/norm(lᵢ)

    lⱼ = make_skew_symmetric(eji)*ejk
    lⱼ = lⱼ/norm(lⱼ)

    lₖ = make_skew_symmetric(eki)*ekj
    lₖ = lₖ/norm(lₖ)

    𝑋 = -lᵢ'*Fki'*make_skew_symmetric(eki)*Fkj*eji
    ϵ = -lᵢ'*Fji'*make_skew_symmetric(eji)*Fkj'*eki
    h = -lᵢ'*Fji'*make_skew_symmetric(eji)*Fkj'*make_skew_symmetric(eki)*Fki*eij
    k = -lᵢ'*Fki'*make_skew_symmetric(eki)*Fkj*make_skew_symmetric(eji)*Fji*eik

    λ₁ = tr(Q)
    λ₂ = k/norm(make_skew_symmetric(eij)*eik)
    λ₃ = h/norm(make_skew_symmetric(eij)*eik)

    B = λ₁*SMatrix{3,3,T}(I) + λ₂*make_skew_symmetric(eij) - λ₃*make_skew_symmetric(eik)
    sol = -(1/ϵ)*(lᵢ'*B*Pᵢ + 𝑋*SVector{4,Float64}([ρⱼ;σⱼ])')
    ρₖ = sol[1:3]
    σₖ = sol[end]
    
    Pₖ = Camera{T}([make_skew_symmetric(eki)*Fki*Pᵢ[1:3,1:3] + eki*ρₖ'  make_skew_symmetric(eki)*Fki*Pᵢ[:,end] + σₖ*eki])

    return Cameras{T}([Pᵢ, Pⱼ, Pₖ]), FundMats{T}([Fji, Fki, Fkj])
end

function get_cams_from_triplet_colombo(F_triplet::FundMats{T}) where T<:AbstractFloat
    # Triplets of fundamental matrices Fji, Fki, and Fkj
    Fji = F_triplet[1];
    eji = SVector{3,T}(get_NullSpace_svd(Fji'));

    # Pᵢ = Camera_canonical;
    Pᵢ = Camera{T}(rand(3,4));
    ρⱼ = rand(3);
    σⱼ = rand();
    # Pⱼ = Camera{T}([ make_skew_symmetric(eji)*Fji + eji*ρⱼ' σⱼ*eji]);
    Pⱼ = Camera{T}([ make_skew_symmetric(eji)*Fji eji]*SMatrix{4,4,T}([Pᵢ[1:3,1:3] Pᵢ[1:3,end]; ρⱼ' σⱼ])  );

    return get_3rd_camera_colombo(F_triplet, Cameras{Float64}([Pᵢ, Pⱼ]), SVector{4,T}([ρⱼ;σⱼ])) 
end

function get_cams_from_triplet_colombo(F_triplet::FundMats{T}, Ps::Cameras{T}) where T<:AbstractFloat
    # Given triplets of F, and  Pᵣ, Pₛ computed wrt a P₁; obtain Pₜ in the same reference frame
    Fsr = F_triplet[1];
    Pᵣ = Ps[1];
    Pₛ = Ps[2];

    esr = SVector{3,T}(get_NullSpace_svd(Fsr'));
    ζ = (1/(Pₛ[:,end]'*make_skew_symmetric(esr)*Fsr*Pᵣ[:,end])) * norm(make_skew_symmetric(esr)*Pₛ[:,end])^2

    param_vector_s = (1/ζ)*esr'*Pₛ

    return get_3rd_camera_colombo(F_triplet, Ps, SVector{4,T}(param_vector_s))
end

function recover_cameras_baselines(bigF::AbstractSparseMatrix, method::String; triplet_cover=nothing, matlab_data=nothing)
    F_multiview = copy(bigF)
    num_cams = size(F_multiview,1)
    Ps = Vector{Camera{Float64}}(repeat([Camera_canonical], num_cams))

    if !isnothing(matlab_data)
        triplets = matlab_data["triplets"]
        ST = matlab_data["ST"]
        root_node = ST[1,1]
        triplet_root = triplets[root_node]
        if contains(method, "colombo")
            Ps[triplet_root] = get_cams_from_triplet_colombo(F_trips(F_multiview, SVector{3,Int}(sort(triplet_root))))[1]
        elseif contains(method, "sinha")
            Ps_root, Fs_root  = get_cams_from_triplet_sinha( F_trips( F_multiview, SVector{3,Int}(sort(triplet_root)) ) )
            Ps[triplet_root] = Ps_root;
            F_multiview[triplet_root[3], triplet_root[2]]  = Fs_root[end];
        end

        covered_nodes = zeros(Bool, num_cams)
        covered_nodes[triplet_root] .= true

        for i=1:size(ST,1)
            dest = ST[i,2]
            new_cam = setdiff(triplets[dest], triplets[ST[i,1]])[1]
            covered_nodes[new_cam] = true

            if contains(method, "colombo")
                Ps[new_cam] = get_cams_from_triplet_colombo( F_trips(F_multiview, SVector{3,Int}([intersect(triplets[dest], triplets[ST[i,1]]);new_cam]) ), Ps[ sort(intersect(triplets[dest], triplets[ST[i,1]]))] )[1][end]
            elseif contains(method,"sinha")
                t = SVector{3,Int}([intersect(triplets[dest], triplets[ST[i,1]]);new_cam]);
                Ps_new, Fs_new = get_cams_from_triplet_sinha( F_trips(F_multiview, t), Ps[intersect(triplets[dest], triplets[ST[i,1]])] )
                F_multiview[t[3],t[2]] = Fs_new[end];
                Ps[new_cam] = Ps_new[end];
            end
        end
    else
        if isnothing(triplet_cover)
            Adj = spzeros(num_cams,num_cams);
            for i=1:num_cams
                for j=i+1:num_cams
                    if !iszero(F_multiview[i,j])
                        Adj[i,j] = 1
                        Adj[j,i] = 1
                    end
                end
            end
            tG, triplets = get_triplet_cover(Adj)
        else
            tG, triplets = triplet_cover
        end

        vmap = tG[2];
        tG = tG[1];
        ST = Graphs.prim_mst(tG)
        ST_dict = Dict{Int,Vector{Int}}()
        for E in ST
            push!(get!(ST_dict, E.src,Int[]), E.dst)
        end

        root_node = setdiff(collect(1:num_cams), [v.dst for v in ST])[1]
        triplet_root = triplets[vmap[root_node]]
                
        if contains(method, "colombo")
            Ps[triplet_root] = get_cams_from_triplet_colombo(F_trips(F_multiview, SVector{3,Int}(sort(triplet_root))))[1]
        elseif contains(method, "sinha")
            Ps_root, Fs_root  = get_cams_from_triplet_sinha(F_trips(F_multiview, SVector{3,Int}(sort(triplet_root))))
            Ps[triplet_root] = Ps_root;
            F_multiview[triplet_root[3], triplet_root[2]]  = Fs_root[end];
        end
        
        covered_nodes = zeros(Bool, num_cams)
        covered_nodes[triplet_root] .= true
        nodes_list = Vector{Int}([root_node])
        
        triplet_nodes = zeros(Bool, length(triplets))
        for (k,v) in ST_dict
            if all(covered_nodes)
                break
            end
            while length(nodes_list) > 0
                if all(covered_nodes)
                    break
                end
                prev_node = nodes_list[1]
                nodes_list = nodes_list[2:end]
                if !(prev_node in keys(ST_dict))
                    continue 
                end        
                dests = ST_dict[prev_node]
                nodes_list = [nodes_list; dests]
                for dest in dests
                    triplet_nodes[vmap[dest]] = true
                    new_cam = setdiff(triplets[vmap[dest]], triplets[vmap[prev_node]])[1]
                    if covered_nodes[new_cam]
                        continue
                    end
                    if contains(method, "colombo")
                        Ps[new_cam] = get_cams_from_triplet_colombo( F_trips(F_multiview, SVector{3,Int}([intersect(triplets[vmap[dest]], triplets[vmap[prev_node]]);new_cam])), Ps[intersect(triplets[vmap[dest]], triplets[vmap[prev_node]])] )[1][end]
                    elseif contains(method, "sinha")
                        t = SVector{3,Int}([intersect(triplets[vmap[dest]], triplets[vmap[prev_node]]);new_cam]);
                        Ps_new, Fs_new = get_cams_from_triplet_sinha( F_trips(F_multiview, t), Ps[intersect(triplets[vmap[dest]], triplets[vmap[prev_node]])] )
                        F_multiview[t[3],t[2]] = Fs_new[end];
                        Ps[new_cam] = Ps_new[end];
                    end
                    covered_nodes[new_cam] = true
                end
            end
            nodes_list = [nodes_list; k]
        end
    end
    return Ps
end

function recover_cameras_baselines_general(bigF::AbstractSparseMatrix, method::String, virtual=true) 
    F_multiview = copy(bigF)
    num_cams = size(F_multiview,1)
    Ps = Vector{Camera{Float64}}(repeat([Camera_canonical], num_cams))
    covered_nodes = zeros(Bool, num_cams)
    
    Adj = spzeros(num_cams,num_cams);
    for i=1:num_cams
        for j=i+1:num_cams
            if !iszero(F_multiview[i,j])
                Adj[i,j] = 1
                Adj[j,i] = 1
            end
        end
    end
    triplets = get_triplets(Adj)
    if length(triplets) == 0
        println("No triplet found")
        return Ps, covered_nodes
    end
    
    can_extend = false
    extend_node = [] 
    iters = 1
    triplet_root = triplets[iters]

    if contains(method, "colombo")
        Ps[triplet_root] = get_cams_from_triplet_colombo(F_trips(F_multiview, SVector{3,Int}(sort(triplet_root))))[1]
    elseif contains(method, "sinha")
        Ps_root, Fs_root  = get_cams_from_triplet_sinha(F_trips(F_multiview, SVector{3,Int}(sort(triplet_root))))
        Ps[triplet_root] = Ps_root;
        F_multiview[triplet_root[3], triplet_root[2]]  = Fs_root[end];
    end

    while !can_extend
        if iters >= length(triplets)
            break
        end
        for i=1:num_cams
            if i in triplet_root
                continue
            end
            if length(intersect(findall(x->x>0, view(Adj,i,1:num_cams)), triplet_root)) >= 2
                can_extend = true
                extend_node = [extend_node;i]
                # break
            end
        end
        if !can_extend
            iters += 1
            triplet_root = triplets[iters]
        end
    end
    if !can_extend
        println("CANT EXTEND")
        return Ps, covered_nodes

    end
    covered_nodes[triplet_root] .= true

    next_nodes = Vector{Int}(extend_node)
    iters = 0

    while !all(covered_nodes)
        if iters>1e3
            break
        end
        if length(next_nodes) == 0
            next_nodes = findall(x->x==false, covered_nodes)
        end

        new_cam = next_nodes[1]
        next_nodes = next_nodes[2:end]
        covered_cams = findall(x->x==true, covered_nodes)
        if new_cam in covered_cams
            continue
        end
        new_cam_neighbors = findall(x->x>0, view(Adj,new_cam,1:num_cams))
        # println(new_cam, "\t", new_cam_neighbors, "\t", covered_cams)
        common = intersect(new_cam_neighbors, covered_cams)

        iters += 1
        if length(common) < 2
            continue
        end
        next_nodes = [next_nodes; new_cam_neighbors]

        possiblePairs = Combinatorics.combinations(common, 2)
        wts = map(x -> Float64(Adj[x...]), possiblePairs)
        if all(wts .== 0)
            wts = ones(size(wts))
        end
        common_pair = StatsBase.sample(collect(possiblePairs), StatsBase.Weights(wts))
        # common_pair = common[1:2]
        if Adj[common_pair...] == 0
            F_multiview[common_pair...] = F_from_cams(Ps[common_pair[2]], Ps[common_pair[1]])
            F_multiview[reverse(common_pair)...] = F_multiview[common_pair...]'
            # Adj[common_pair...] = 1 
            # Adj[reverse(common_pair)...] = 1 
        end

        if contains(method, "colombo")
            Ps[new_cam] = get_cams_from_triplet_colombo( F_trips(F_multiview, SVector{3,Int}([common_pair;new_cam])), Ps[common_pair] )[1][end]
        elseif contains(method, "sinha")
            t = SVector{3,Int}([common_pair;new_cam]);
            Ps_new, Fs_new = get_cams_from_triplet_sinha( F_trips(F_multiview, t), Ps[common_pair] )
            F_multiview[t[3],t[2]] = Fs_new[end];
            Ps[new_cam] = Ps_new[end];
        end
        covered_nodes[new_cam] = true
    end
    return Ps, covered_nodes
end

# test_mthds = ["baseline_sinha"]
# test_mthds = [];
# Ps_gt, F, Err = create_synthetic_environment(0.01, test_mthds; holes_density=0.6,  initialize=false,  num_cams=10, noise_type="angular");
# Ps = recover_cameras_baselines_general(F, "sinha");
# mean(rad2deg.(compute_error(Ps_gt, Ps, projective_synchronization.angular_distance)))
# Ps = recover_cameras_baselines(F, "sinha");
# mean(rad2deg.(compute_error(Ps_gt, Ps, projective_synchronization.angular_distance)))
