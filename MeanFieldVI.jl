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

    T = sparse(rval[1:index-1],cval[1:index-1],zval[1:index-1],nS_sub,nS_sub)
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
# This could possibly be vectorized better
function update_T_τ!(hmdp::hMDP, T, s_τv, Qv)
	nS = size(Qv, 1)
	nτ = size(mdp.T_τ, 1)
	nS_sub = convert(Int64, nS/nτ)
	τs = collect(range(0, step=1, length=nτ))

	T_τ = zeros(nτ, nτ)

	# Printing
	fracBase = 0.2
    frac = fracBase

	for τ⁰ in τs
		τ⁰ind = τ⁰ + 1
		# Compute denominator
		den = sum(s_τ[:, τ⁰ind])
		for τ¹ in τs
			τ⁰ind = τ⁰ + 1
			outer_sum = 0
			for s⁰ in 1:nS_sub
				s_ind = nS_sub*τ⁰ + s⁰
				action = argmax(Qv[s_ind, :])
				inner_sum = 0
				for s¹ in 1:nS_sub
					inner_sum += s_τ[s¹, τ¹ind]*T[action][s⁰, s¹]
				end
				outer_sum += s_τ[s⁰, τ⁰ind]*inner_sum
			end
			T_τ[τ⁰ind, τ¹ind] = outer_sum/den
		end

		if τ⁰ind/nτ > frac
			print(round(frac*100))
            println("% Complete")
            frac += fracBase
        end
	end
	return T_τ
end

function compute_s_τ(mdp::vMDP, nτh)
	nS = mdp.nS
	nτ = size(mdp.T_τ, 1)
	nS_sub = convert(Int64, nS/nτ)
	s_τ = spzeros(nS_sub, nτh)
	for i = 1:nS_sub
		τ = vertical_τ(mdp, i)
		# These next few lines just do 1D interpolation
		if (τ < 0) || (τ > nτh - 1)
			s_τ[i, nτh] = 1.0
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