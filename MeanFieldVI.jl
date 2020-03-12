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
# # This could possibly be vectorized better
# function update_T_τ!(hmdp::hMDP, vmdp::vMDP, T, s_τ, Q)
# 	nS = size(Q, 1)
# 	nτh = size(vmdp.T_τ, 1)
# 	nτv = size(hmdp.T_τ, 1)
# 	nS_sub = convert(Int64, nS/nτh)
# 	τs = collect(range(0, step=1, length=nτv))

# 	T_τ = zeros(nτv, nτv)

# 	# Printing
# 	fracBase = 0.1
#     frac = fracBase

# 	for τ⁰ in τs
# 		τ⁰ind = τ⁰ + 1
# 		# Compute denominator
# 		den = sum(s_τ[:, τ⁰ind])
# 		# Get T for specific policy
# 	    println("Getting T_pol...")
# 	    T_pol = get_T_policy(T, Q[τ⁰*nS_sub+1:(τ⁰+1)*nS_sub,:])
# 	    println("Done!")
# 		for τ¹ in τs
# 			τ¹ind = τ¹ + 1
# 			# outer_sum = 0
# 			# for s⁰ in 1:nS_sub
# 			# 	s_ind = nS_sub*τ⁰ + s⁰
# 			# 	action = argmax(Q[s_ind, :])
# 			# 	inner_sum = 0
# 			# 	for s¹ in 1:nS_sub
# 			# 		inner_sum += s_τ[s¹, τ¹ind]*T[action][s⁰, s¹]
# 			# 	end
# 			# 	outer_sum += s_τ[s⁰, τ⁰ind]*inner_sum
# 			# end
# 			# T_τ[τ⁰ind, τ¹ind] = outer_sum/den
# 			inner_sum = T_pol*s_τ[:, τ¹ind]
# 			outer_sum = sum(s_τ[:, τ⁰ind].*inner_sum)
# 			T_τ[τ⁰ind, τ¹ind] = outer_sum/den
# 		end

# 		if τ⁰ind/nτv > frac
# 			print(round(frac*100))
#             println("% Complete")
#             frac += fracBase
#         end
# 	end
# 	return T_τ
# end

function update_T_τ!(hmdp::hMDP, vmdp::vMDP, T, s_τ, Q)
	nS = size(Q, 1)
	nτh = size(vmdp.T_τ, 1)
	nτv = size(hmdp.T_τ, 1)
	nS_sub = convert(Int64, nS/nτh)
	τs = collect(range(0, step=1, length=nτv))

	s_τ_all = repeat(s_τ, nτh) #./nτh
	T_τ = zeros(nτv, nτv)

	println("Getting policy transition matrix...")
	T_pol = get_T_policy(vmdp, Q)
	println("Done!")

	for τ⁰ in τs
		τ⁰ind = τ⁰ + 1
		den = sum(s_τ_all[:, τ⁰ind])
		for τ¹ in τs
			τ¹ind = τ¹ + 1

			# Actual computation
			inner_sum = T_pol*s_τ_all[:, τ¹ind]
			outer_sum = sum(s_τ_all[:, τ⁰ind].*inner_sum)

			T_τ[τ⁰ind, τ¹ind] = outer_sum/den
		end
	end
	return T_τ
end

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
	h, ḣ₀, ḣ₁ = s_grid[1], s_grid[2], s_grid[3]
	τ = 0
	if ḣ₀ - ḣ₁ == 0
		τ = h == 0 ? 0 : -1
	else
		τ = h/(ḣ₀ - ḣ₁)
	end
	return τ
end

function get_T_policy(mdp, Q)
	nS, nA = mdp.nS, mdp.nA
	T_τ = mdp.T_τ
	nτ = size(T_τ, 1)
	nS_sub = convert(Int64, nS/nτ)

	rval = zeros(Int32, mdp.nS*100)
    cval = zeros(Int32, mdp.nS*100)
    zval = zeros(Float32, mdp.nS*100)
    index = 1

    for i = 1:nS_sub
		for j = 1:nτ
			sps, probs = transition(mdp, i, argmax(Q[(j-1)*nS_sub + i, :]))
			inds = findall(T_τ[j,:] .!= 0)
			for ind in inds
				for (spi, probi) in zip(sps,probs)
		            rval[index] = (j-1)*nS_sub + i
		            cval[index] = (ind-1)*nS_sub + spi
		            zval[index] = probi*T_τ[j, ind]
		            index += 1
		        end
		    end
	    end
    end

    return sparse(rval[1:index-1], cval[1:index-1], zval[1:index-1], nS, nS)
end



# function get_T_policy(T, Q)
# 	nS_sub = size(T[1], 1)
# 	nS = size(Q, 1)
# 	nτ = convert(Int64, nS/nS_sub)
# 	nA = size(Q,2)
# 	T_tot = Dict()
# 	for i = 1:nA
# 		T_tot[i] = repeat(T[i], nτ)
# 	end
# 	actions = argmax(Q, dims=2)
# 	println("Computing T_pol...")
# 	colvals = [T_tot[actions[i][2]][i,:].nzind for i in 1:nS]
# 	println("Progress: 33%")
# 	rowvals = [ones(length(T_tot[actions[i][2]][i,:].nzind))*i for i in 1:nS]
# 	println("Progress: 67%")
# 	zvals = [T_tot[actions[i][2]][i,:].nzval for i in 1:nS]
# 	println("Progress: 100%")
# 	T_pol = sparse(vcat(rowvals...), vcat(colvals...), vcat(zvals...))
# 	return T_pol
# end

# function get_T_policy(T, Q)
# 	n = size(T[1], 1)
# 	actions = argmax(Q, dims=2)
# 	colvals = [T[actions[i][2]][i,:].nzind for i in 1:n]
# 	rowvals = [ones(length(T[actions[i][2]][i,:].nzind))*i for i in 1:n]
# 	zvals = [T[actions[i][2]][i,:].nzval for i in 1:n]
# 	T_pol = sparse(vcat(rowvals...), vcat(colvals...), vcat(zvals...))
# 	return T_pol
# end

# function get_T_policy(T, Q)
# 	n = size(T[1], 1)
# 	actions = argmax(Q, dims=2)
# 	T_pol = [T[actions[i][2]][i,:] for i in 1:n]
# 	return T_pol
# end

# function get_T_policy(T, Q)
# 	n = size(T[1], 1)
# 	T_pol = spzeros(n, n)
# 	actions = argmax(Q, dims=2)
# 	for i = 1:n
# 		T_pol[i,:] = T[actions[i][2]][i,:]
# 	end
# 	return T_pol
# end