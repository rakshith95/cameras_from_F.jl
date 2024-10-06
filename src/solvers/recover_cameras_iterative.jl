function recover_cameras_iterative(F_multiview::AbstractSparseMatrix; X₀=nothing, weights=ones(size(F_multiview)...), kwargs...) 
    method = get(kwargs, :method, "subspace")
    max_it = get(kwargs, :max_iterations, 100)
    max_updates = get(kwargs, :max_updates, max_it)
    min_updates = get(kwargs, :min_updates, 10)
    δ = get(kwargs, :δ, 1e-6)
    initial_updated = get(kwargs, :update_init, "all")
    update_method = get(kwargs, :update, "order-weights-update-all")
    set_anchor = get(kwargs, :anchor, "fixed")
    # gt = get(kwargs, :gt, nothing)
    num_cams = size(F_multiview, 1)

    if occursin("skew_symm", lowercase(method))
        if occursin("vectorized", lowercase(method))
            avg = recover_camera_SkewSymm_vectorization
        elseif occursin("l1", lowercase(method))
            avg(Ps,Fs,wts) = recover_camera_SkewSymm(Ps,Fs,wts; l1=true)
        else
            avg = recover_camera_SkewSymm
        end
    elseif occursin("grad", lowercase(method))
        avg = recover_cameras_gradient_descent
    elseif occursin("subspace", lowercase(method))
        if occursin("angular", method)
            avg = recover_camera_subspace_angular
        else
            avg = recover_camera_subspace
        end
    end

    steady = zeros(Bool, num_cams)
    updated = zeros(Int, num_cams)


    if isnothing(X₀)
        if num_cams < 40
            Ps = SizedVector{num_cams, Camera{Float64}}(repeat([Camera_canonical], num_cams))
        else
            Ps = Vector{Camera{Float64}}(repeat([Camera_canonical], num_cams))
        end
    else
        Ps = copy(X₀)
    end

    if occursin("all", lowercase(initial_updated))
        updated .= 1
        if !isnothing(X₀)
            for i=1:num_cams
                if Ps[i] == Camera_canonical
                    updated[i] = 0 
                end
            end
        end
    end

    # Normalize
    Adj = sparse(zeros(num_cams,num_cams))
    for i=1:num_cams
        Ps[i] = Ps[i]/norm(Ps[i])
        for j=i+1:num_cams
            if iszero(F_multiview[i,j])
                continue
            else
                Adj[i,j] = 1
                Adj[j,i] = 1
                F_multiview[i,j] = F_multiview[i,j]/norm(F_multiview[i,j])
                F_multiview[j,i] = F_multiview[i,j]'
            end
        end
    end
    G = Graph(Adj)
    # C = eigenvector_centrality(G)
    C = closeness_centrality(G)

    if occursin("nothing", set_anchor) || occursin("none", set_anchor)
        anchor = rand(1:n)
        updated[anchor] = 1
    else
        if occursin("centrality", set_anchor)
            # Set anchor node according to centrality
                _,anchor = findmax(C)
        elseif occursin("fixed", set_anchor)
            anchor = 1
        elseif occursin("rand", set_anchor)
            anchor = rand(1:num_cams)
        end
        updated[anchor] = max_updates
        steady[anchor] = true
    end

    nodes = collect(1:num_cams)
    Ne = [neighbors(G, j) for j=1:num_cams]

    if occursin("order", lowercase(update_method))
        if occursin("centrality", lowercase(update_method))
            nodes = sortperm(C, rev=true)
            # Try reverse
        elseif occursin("rand", lowercase(update_method))
            Random.shuffle!(nodes)
        elseif occursin("weights", lowercase(update_method))
            nodes = sortperm( [ prod(weights[j,Ne[j]])*C[j] for j=1:num_cams], rev=true)
            # println([ prod(weights[j,Ne[j]])*C[j] for j=1:num_cams][nodes])
            # println(nodes)
            # nodes = collect(1:num_cams)
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
                N = Ne[j]
                if iszero(updated[N])
                    continue
                end
                oldP = Ps[j]
                updated_N = N[updated[N].!=0]
                F_inds = [ CartesianIndex(j,i) for i in updated_N ]
                Ps[j] = avg(Ps[updated_N], FundMats{Float64}(F_multiview[F_inds]), weights[F_inds], oldP)
                # Ps[j] = avg(Ps[updated_N], FundMats{Float64}(F_multiview[F_inds]), weights[F_inds])

                updated[j] += 1
                steady[j] = updated[j] >= min_updates && projective_synchronization.angular_distance(vec(oldP), vec(Ps[j])) <= δ
                if all(steady) || all(updated .> max_updates)
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
            N = Ne[j]
            if iszero(updated[N])
                continue
            end
            oldP = Ps[j]
            updated_N = N[updated[N].!=0]
            F_inds = [ CartesianIndex(j,i) for i in updated_N ]
            Ps[j] = avg(Ps[updated_N], FundMats{Float64}(F_multiview[F_inds]), weights[F_inds], oldP) 
            updated[j] += 1
            steady[j] = updated[j] >= min_updates && projective_synchronization.angular_distance(vec(oldP), vec(Ps[j])) <= δ
            if all(steady) || all(updated .>= max_updates)
                exit_loop = true
                break
            end
        end
        iter += 1
    end
    # println("\n\n")
    println(iter)
    return Cameras{Float64}(Ps)
end