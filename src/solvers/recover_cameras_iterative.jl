function recover_cameras_iterative(F_multiview::AbstractSparseMatrix; P₀=nothing, weights=nothing, kwargs...) 
    method = get(kwargs, :recovery_method, "skew_symmetric")
    max_it = get(kwargs, :max_iterations, 1000)
    max_updates = get(kwargs, :max_updates, max_it)
    min_updates = get(kwargs, :min_updates, 10)
    δ = get(kwargs, :δ, 1e-10)
    initial_updated = get(kwargs, :update_init, "only-anchor")
    update_method = get(kwargs, :update, "all-random")
    set_anchor = get(kwargs, :anchor, "fixed")

    if occursin("skew_symm", lowercase(method))
        avg = recover_camera_SkewSymm
    elseif occursin("vectorized", lowercase(method))
        avg = recover_camera_SkewSymm_vectorization
    elseif occursin("combinations", lowercase(method))
        avg = recover_camera_averaging
    end

    num_cams = size(F_multiview, 1)
    if isnothing(weights)
        weights = ones(num_cams, num_cams)
    end

    Adj = sparse(zeros(num_cams,num_cams))
    Adj[findall(F_multiview .!= 0)] .= 1
    
    G = Graph(Adj)
    steady = zeros(Bool, num_cams)
    updated = zeros(Int, num_cams)


    if isnothing(P₀)
        if num_cams < 40
            Ps = SizedVector{num_cams, Camera{Float64}}(repeat([Camera_canonical], num_cams))
            # Ps = Vector{Camera{Float64}}(repeat([rand(3,4)], num_cams))
            # for i=1:num_cams
                # Ps[i] = Camera{Float64}(rand(3,4))
            # end
        else
            Ps = Vector{Camera{Float64}}(repeat([Camera_canonical], num_cams))
        end
    else
        Ps = copy(P₀)
    end

    if occursin("all", lowercase(initial_updated))
        updated .= 1
        # if !isnothineg(P₀)
            # for i=1:enum_cams
                # if Pes[i] == Camera_canonical
                    # eupdated[i] = 0 
                # ende
            # ende
        # ende
    end
    C = nothing
    try
        C = eigenvector_centrality(G)
    catch
        C = degree_centrality(G)
    end
    if median(C) < 1e-5
        C = degree_centrality(G, normalize=true)
    end

    if occursin("nothing", set_anchor) || occursin("none", set_anchor)
        anchor = rand(1:n)
        updated[anchor] = 1
    else
        if occursin("centrality", set_anchor)
            # Set anchor node according to centrality
            if median(C) < 1e-6
                _,anchor = findmax(C)
            else
                N = neighbors(G,findmax(C)[2])
                N_degs = C[N]        
                anchor = N[findmin(N_degs)[2]]
            end
        elseif occursin("fixed", set_anchor)
            anchor = 1
        elseif occursin("rand", set_anchor)
            anchor = rand(1:num_cams)
        end
        updated[anchor] = max_updates
        steady[anchor] = true
    end
    
    nodes = collect(1:num_cams)
    
    if occursin("start", lowercase(update_method))
        if occursin("centrality", lowercase(update_method))
            nodes = sortperm(C, rev=true)
            # Try reverse
        else
            Random.shuffle!(nodes)
        end
    end

    iter = 0

    exit_loop = false
    while !exit_loop
        if iter >= max_it
            exit_loop = true
            continue
        end
        if occursin("all", lowercase(update_method))
            prev_steady = copy(steady)
            if occursin("random", lowercase(update_method))
                Random.shuffle!(nodes)
            end
            for j in nodes
                if prev_steady[j] && steady[j] 
                    continue
                end
                N = neighbors(G, j)
                if iszero(updated[N])
                    continue
                end
                oldP = Ps[j]
                updated_N = N[updated[N].!=0]
                F_inds = [ CartesianIndex(j,i) for i in updated_N ]
                Ps[j] = avg(Ps[updated_N], FundMats{Float64}(F_multiview[F_inds])) 
                updated[j] += 1
                steady[j] = updated[j] >= min_updates && projective_synchronization.angular_distance(vec(oldP), vec(Ps[j])) <= δ
                if all(steady) || all(updated .>= max_updates)
                    exit_loop = true
                    break
                end
            end
        else
            wts =  (max_updates .- updated)./max_updates .* .!steady
            if iszero(wts)
                exit_loop = true
                break
            end
            j = StatsBase.wsample(projective_synchronization.unit_normalize!(wts))
            N = neighbors(G, j)
            if iszero(updated[N])
                continue
            end
            oldP = Ps[j]
            updated_N = N[updated[N].!=0]
            F_inds = [ CartesianIndex(j,i) for i in updated_N ]
            Ps[j] = avg(Ps[updated_N], FundMats{Float64}(F_multiview[F_inds])) 
            updated[j] += 1
            steady[j] = updated[j] >= min_updates && projective_synchronization.angular_distance(vec(oldP), vec(Ps[j])) <= δ
            if all(steady) || all(updated .>= max_updates)
                exit_loop = true
                break
            end
        end
        iter += 1
    end
    # println(iter)
    return Cameras{Float64}(Ps)
end