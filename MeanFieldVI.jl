using SparseArrays

function vanilla_VI(mdp; tol=0.01, max_iter=2000, print_every=1, init_Q=[])
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