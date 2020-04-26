"""
----------------------------------
Value Iteration Algorithms
----------------------------------
"""

function vanilla_VI(mdp; tol=0.001, max_iter=2000, print_every=1, init_Q=[])
	nS, nA = mdp.nS, mdp.nA
	γ = mdp.γ
	println("building matrices...")
	T, R = build_matrices(mdp)
	println("Done!")
	Q = length(init_Q) > 0 ? init_Q : zeros(nS, nA)
	Q_rc = Dict()
	U = zeros(nS)

	for i = 1:max_iter
		Q_prev = copy(Q)
		for a in 1:nA
			Q_rc[a] = remotecall(bellman_update, mod(a, nprocs()-1) + 2, mdp, R[:,a], T[a], U)
		end
		for a in 1:nA
			Q[:,a] = fetch(Q_rc[a])
		end

		resid = maximum(abs.(Q .- Q_prev))
		println("iter: $i, resid: $resid")
		if resid < tol
			break
		end
		U = maximum(Q, dims=2)
	end

	return Q
end

function mfVI(vmdp, hmdp, s_τv, s_τh; tol=0.001, max_iter=2000, print_every=1, init_Qv=[], init_Qh=[])
	nSv, nAv = vmdp.nS, vmdp.nA
	γv = vmdp.γ
	println("building matrices...")
	Tv, Rv = build_matrices(vmdp)
	println("Done!")
	Qv = length(init_Qv) > 0 ? init_Qv : zeros(nSv, nAv)
	Qv_rc = Dict()
	Uv = maximum(Qv, dims=2)

	nSh, nAh = hmdp.nS, hmdp.nA
	γh = hmdp.γ
	println("building matrices...")
	Th, Rh = build_matrices(hmdp)
	println("Done!")
	Qh = length(init_Qh) > 0 ? init_Qh : zeros(nSh, nAh)
	Qh_rc = Dict()
	Uh = maximum(Qh, dims=2)

	for i = 1:max_iter
		Qv_prev = copy(Qv)
		Qh_prev = copy(Qh)
		for a in 1:nAv
			Qv_rc[a] = remotecall(bellman_update, mod(a, nprocs()-1) + 2, vmdp, Rv[:,a], Tv[a], Uv)
		end
		for a in 1:nAv
			Qv[:,a] = fetch(Qv_rc[a])
		end
		for a in 1:nAh
			Qh_rc[a] = remotecall(bellman_update, mod(a, nprocs()-1) + 2, hmdp, Rh[:,a], Th[a], Uh)
		end
		for a in 1:nAh
			Qh[:,a] = fetch(Qh_rc[a])
		end

		resid = max(maximum(abs.(Qv .- Qv_prev)), maximum(abs.(Qh .- Qh_prev)))
		println("iter: $i, resid: $resid")
		if resid < tol
			break
		end
		Uv = maximum(Qv, dims=2)
		Uh = maximum(Qh, dims=2)

		if mod(i, 2) == 0 && i > 50
			println("Updating vertical...")
			update_T_τ!(vmdp, hmdp, Th, s_τh, Qh)
			GC.gc()
			println("Updating horizontal...")
			update_T_τ!(hmdp, vmdp, Tv, s_τv, Qv)
			GC.gc()
		end
	end

	return Qv, Qh
end

"""
----------------------------------
Helper functions
----------------------------------
"""

function build_matrices(mdp)
	nS, nA = mdp.nS, mdp.nA
	T = Dict()
	R = zeros(nS, nA)
	rc = Dict()

	for a in 1:nA
		rc[a] = remotecall(T_R_action, mod(a, nprocs()-1) + 2, mdp, a)
	end
	for a in 1:nA
		(T[a], R[:,a]) = fetch(rc[a])
	end
	return T, R
end

function T_R_action(mdp, a)
	nS = mdp.nS
	nτ = size(mdp.T_τ, 1)
	nS_sub = convert(Int64, nS/nτ)

	R = zeros(nS)
	for i = 1:nτ
		for j = 1:nS_sub
			R[(i-1)*nS_sub + j] = reward(mdp, j, i, a)
		end
	end

	rval = zeros(Int32, mdp.nS*100)
    cval = zeros(Int32, mdp.nS*100)
    zval = zeros(Float32, mdp.nS*100)
    index = 1

	for i = 1:nS_sub
		sps, probs = transition(mdp, i, a)
		for (spi, probi) in zip(sps,probs)
            rval[index] = i
            cval[index] = spi
            zval[index] = probi
            index += 1
        end
    end

    T = sparse(rval[1:index-1], cval[1:index-1], zval[1:index-1], nS_sub, nS_sub)
    return (T, R)
end

function bellman_update(mdp, R, T, U)
	nS = mdp.nS
	γ = mdp.γ
	T_τ = mdp.T_τ
	nτ = size(mdp.T_τ, 1)
	nS_sub = convert(Int64, nS/nτ)

	Qa = R
	for i = 1:nτ
		r_newτ = γ*T*U[(i-1)*nS_sub+1:i*nS_sub]
		for j = 1:nτ
			Qa[(j-1)*nS_sub+1:j*nS_sub] += T_τ[j,i]*r_newτ
		end
	end
	return Qa
end

"""
----------------------------------
Mean Field Approximation
----------------------------------
"""

function update_T_τ!(mdp_to_update, τmdp, T, s_τ, Q)
	nS = size(Q, 1)
	nτ_states = size(τmdp.T_τ, 1)
	nτ = size(mdp_to_update.T_τ, 1)
	nS_sub = convert(Int64, nS/nτ_states)
	τs = collect(range(0, step=1, length=nτ))

	s_τ_all = repeat(s_τ, nτ_states)
	T_τ = zeros(nτ, nτ)

	for i = 1:nτ_states
		# Get the policy transition matrix
		actions = argmax(Q[(i-1)*nS_sub+1:i*nS_sub, :], dims=2)
		mod(i,15) == 0 ? println("$i: getting T_pol") : nothing
		T_pol = get_T_policy(T, Q)
		# Get the partial sums over s1 for all of them
		partial_over_s1 = T_pol*s_τ # nS_sub x nτ
		partial_over_s0 = s_τ'*partial_over_s1 # nτ x nτ
		T_τ .+= partial_over_s0
	end

	# Normalize
	for i = 1:nτ
		T_τ[i,:] = T_τ[i,:]/sum(T_τ[i,:])
	end

    mdp_to_update.T_τ = T_τ
	return T_τ
end

# function get_T_policy(T, Q)
# 	n = size(T[1], 1)
# 	T_pol = spzeros(n, n)
# 	actions = argmax(Q, dims=2)
# 	for i = 1:n
# 		T_pol[i,:] = T[actions[i][2]][i,:]
# 	end
# 	return T_pol
# end

function get_T_policy(T, Q)
	n = size(T[1], 1)
	actions = argmax(Q, dims=2)
	colvals = [T[actions[i][2]][i,:].nzind for i in 1:n]
	rowvals = [ones(length(T[actions[i][2]][i,:].nzind))*i for i in 1:n]
	zvals = [T[actions[i][2]][i,:].nzval for i in 1:n]
	T_pol = sparse(vcat(rowvals...), vcat(colvals...), vcat(zvals...))
	return T_pol
end

# function update_T_τ!(mdp_to_update, τmdp, T, s_τ, Q)
# 	nS = size(Q, 1)
# 	nτ_states = size(τmdp.T_τ, 1)
# 	nτ = size(mdp_to_update.T_τ, 1)
# 	nS_sub = convert(Int64, nS/nτ_states)
# 	τs = collect(range(0, step=1, length=nτ))

# 	s_τ_all = repeat(s_τ, nτ_states)
# 	T_τ = zeros(nτ, nτ)

# 	println("Getting policy transition matrix...")
# 	T_pol = get_T_policy(τmdp, Q)
# 	println("Done!")

# 	for τ⁰ in τs
# 		τ⁰ind = τ⁰ + 1
# 		den = sum(s_τ_all[:, τ⁰ind])
# 		for τ¹ in τs
# 			τ¹ind = τ¹ + 1

# 			# Actual computation
# 			inner_sum = T_pol*s_τ_all[:, τ¹ind]
# 			outer_sum = sum(s_τ_all[:, τ⁰ind].*inner_sum)

# 			T_τ[τ⁰ind, τ¹ind] = outer_sum == 0 ? 0 : outer_sum/den
# 		end
# 	end
# 	T_pol = 0
#     mdp_to_update.T_τ = T_τ
# 	return T_τ
# end

# function update_T_τ!(mdp_to_update, τmdp, T, s_τ, Q)
# 	nS = size(Q, 1)
# 	nτ_states = size(τmdp.T_τ, 1)
# 	nτ = size(mdp_to_update.T_τ, 1)
# 	nS_sub = convert(Int64, nS/nτ_states)
# 	τs = collect(range(0, step=1, length=nτ))

# 	s_τ_all = repeat(s_τ, nτ_states)
# 	T_τ = zeros(nτ, nτ)

# 	for τ⁰ in τs
# 		println(τ⁰)
#         τ⁰ind = τ⁰ + 1
# 		den = sum(s_τ_all[:, τ⁰ind])
# 		for τ¹ in τs
# 			τ¹ind = τ¹ + 1

# 			# Actual computation
#             outer_sum = 0.0
#             for s0 in 1:5000 #nS
#                 if s_τ_all[s0, τ⁰ind] != 0.0
#                     a = argmax(Q[s0,:])
#                     Ta = T[a]
#                     inner_sum = 0.0
#                     for s1 in 1:5000 #nS
#                         if s_τ_all[s1, τ¹ind] != 0.0
#                             @inbounds inner_sum += Ta[mod(s0, nS_sub), mod(s1, nS_sub)]*s_τ_all[s1, τ¹ind]
#                         end
#                     end
#                     outer_sum += s_τ_all[s0, τ⁰ind]*inner_sum
#                 end
#             end

# 			T_τ[τ⁰ind, τ¹ind] = outer_sum == 0.0 ? 0.0 : outer_sum/den
# 		end
# 	end
    
#     mdp_to_update.T_τ = T_τ
# 	return T_τ
# end

# function update_T_τ!(mdp_to_update, τmdp, T, s_τ, Q)
# 	nS = size(Q, 1)
# 	nτ_states = size(τmdp.T_τ, 1)
# 	nτ = size(mdp_to_update.T_τ, 1)
# 	nS_sub = convert(Int64, nS/nτ_states)
# 	τs = collect(range(0, step=1, length=nτ))

# 	s_τ_all = repeat(s_τ, nτ_states)
# 	T_τ = zeros(nτ, nτ)

# 	actions = argmax(Q, dims=2)

# 	# Go through the τs BUT THESE ARE PART OF STATES
# 	for i = 1:nτ_states
# 		inner_sum = zeros(nS)
# 		for j = 1:nτ_states
# 			T_pol = [T[actions[k][2]][k,:] for k in (i-1)*nS_sub+1:(i-1)*nS_sub]
# 			# I don't knowwwwwww
# 		end
# 	end




# 	for τ⁰ in τs
# 		println(τ⁰)
#         τ⁰ind = τ⁰ + 1
# 		den = sum(s_τ_all[:, τ⁰ind])
# 		for τ¹ in τs
# 			τ¹ind = τ¹ + 1

# 			# Actual computation
#             outer_sum = 0.0
#             for s0 in 1:5000 #nS
#                 if s_τ_all[s0, τ⁰ind] != 0.0
#                     a = argmax(Q[s0,:])
#                     inner_sum = 0.0
#                     for s1 in 1:5000 #nS
#                         if s_τ_all[s1, τ¹ind] != 0.0
#                             inner_sum += T[a][mod(s0, nS_sub), mod(s1, nS_sub)]*s_τ_all[s1, τ¹ind]
#                         end
#                     end
#                     outer_sum += s_τ_all[s0, τ⁰ind]*inner_sum
#                 end
#             end

# 			T_τ[τ⁰ind, τ¹ind] = outer_sum == 0.0 ? 0.0 : outer_sum/den
# 		end
# 	end
    
#     mdp_to_update.T_τ = T_τ
# 	return T_τ
# end

# function get_T_policy(mdp, Q)
# 	nS, nA = mdp.nS, mdp.nA
# 	T_τ = mdp.T_τ
# 	nτ = size(T_τ, 1)
# 	nS_sub = convert(Int64, nS/nτ)

# 	rval = zeros(Int32, mdp.nS*72*10) #Vector{Int32}() #zeros(Int32, mdp.nS*800)
#     cval = zeros(Int32, mdp.nS*72*10) #Vector{Int32}() #zeros(Int32, mdp.nS*800)
#     zval = zeros(Float32, mdp.nS*72*10) #Vector{Float32}() #zeros(Float32, mdp.nS*800)
#     index = 1

#     for i = 1:nS_sub
# 		for j = 1:nτ
#             # Keep dictionary of all the indices (this is about to get messy)
#             index_dict = Dict()
# 			sps, probs = transition(mdp, i, argmax(Q[(j-1)*nS_sub + i, :]))
# 			inds = findall(T_τ[j,:] .!= 0)
#             if length(sps) > 72
#                 println("there is an issue!")
#             end
# 			for ind in inds
# 				for (spi, probi) in zip(sps,probs)
#                     c = (ind-1)*nS_sub + spi
#                     if haskey(index_dict, c)
#                         zval[index_dict[c]] += probi*T_τ[j, ind]
#                     else
#                         rval[index] = (j-1)*nS_sub + i #push!(rval, (j-1)*nS_sub + i)
#                         cval[index] = c #push!(cval, (ind-1)*nS_sub + spi)
#                         index_dict[c] = index
#                         zval[index] = probi*T_τ[j, ind] #push!(zval, probi*T_τ[j, ind])
#                         index += 1
#                     end
# 		        end
# 		    end
# 	    end
#     end

#     return sparse(rval[1:index-1], cval[1:index-1], zval[1:index-1], nS, nS)
# end

"""
----------------------------------
Offline Mean Field Approximation Calculations
----------------------------------
"""

function compute_s_τ(mdp::vMDP, nτv)
	nS = mdp.nS
	nτ = size(mdp.T_τ, 1)
	nS_sub = convert(Int64, nS/nτ)
	s_τ = spzeros(nS_sub, nτv)
	for i = 1:nS_sub
		τ = vertical_τ(mdp, i)
		# These next few lines just do 1D interpolation
		if (τ < 0) || (τ ≥ nτv - 1)
			s_τ[i, nτv] = 1.0
		elseif τ == 0
			s_τ[i, 1] = 1.0
		else
			floorτ = floor(τ)
			lower_ind = convert(Int64, floorτ) + 1
			s_τ[i, lower_ind] = τ - floorτ
			s_τ[i, lower_ind+1] = 1 - (τ - floorτ)
		end
	end
	return s_τ
end

function vertical_τ(mdp::vMDP, s_ind)
	s_grid = ind2x(mdp.grid, s_ind)
	h, ḣ₀, ḣ₁ = s_grid[1], s_grid[2], s_grid[3]
	τ = 0
	if ḣ₀ - ḣ₁ == 0
		τ = h == 0 ? 0 : -1
	else
		τ = h/(ḣ₀ - ḣ₁)
	end
	return τ
end

function compute_s_τ(mdp::hMDP, nτv)
	nS = mdp.nS
	nτ = size(mdp.T_τ, 1)
	nS_sub = convert(Int64, nS/nτ)
	s_τ = spzeros(nS_sub, nτv)
	for i = 1:nS_sub
		τ = horizontal_τ(mdp, i)
		# These next few lines just do 1D interpolation
		if (τ < 0) || (τ ≥ nτv - 1)
			s_τ[i, nτv] = 1.0
		elseif τ == 0
			s_τ[i, 1] = 1.0
		else
			floorτ = floor(τ)
			lower_ind = convert(Int64, floorτ) + 1
			s_τ[i, lower_ind] = τ - floorτ
			s_τ[i, lower_ind+1] = 1 - (τ - floorτ)
		end
	end
	return s_τ
end

function horizontal_τ(mdp::hMDP, s_ind)
	s_grid = ind2x(mdp.grid, s_ind)
	r, θ, ψ = s_grid[1], s_grid[2], s_grid[3]
	v = mdp.vh

	x₀ = 0.0; y₀ = 0.0; x₁ = r*cos(θ); y₁ = r*sin(θ)
	ẋ₀ = v; ẏ₀ = 0.0; ẋ₁ = v*cos(ψ); ẏ₁ = v*sin(ψ)
	x₀ += ẋ₀; y₀ += ẏ₀; x₁ += ẋ₁; y₁ += ẏ₁

	x₁next = x₁ - x₀; y₁next = y₁ - y₀
	rnext = √(x₁next^2 + y₁next^2)

	ṙ = r - rnext
	τ = ṙ == 0 ? -1.0 : r/ṙ
	if r == 0
		τ = 0.0
	end

	return τ
end
