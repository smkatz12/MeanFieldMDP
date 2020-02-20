using SparseArrays

function vanilla_VI(mdp; tol=0.001, max_iter=200, print_every=1)
	nS, nA = mdp.nS, mdp.nA
	γ = mdp.γ
	T, R = build_matrices(mdp)
	Q = zeros(nS, nA)
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

		resid = maximum(Q .- Q_prev)
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
	nτ = size(mdp.T_τ, a)
	nS_sub = convert(Int64, nS/nτ)

	R = zeros(nS)
	for i = 1:nτ
		for j = 1:nS_sub
			R[(i-1)*j + j] = reward(mdp, j, i, a)
		end
	end

	rval = zeros(Int32,nS*100)
    cval = zeros(Int32,nS*100)
    zval = zeros(Float32,nS*100)
    index = 1

	for i = 1:nS_sub
		sps, probs = transition(mdp, nS_sub, a)
		for (spi, probi) in zip(sps,probs)
            rval[index] = i
            cval[index] = spi
            zval[index] = probi*p
            index += 1
        end
    end

    T = sparse(rval[1:index-1],cval[1:index-1],zval[1:index-1],nS,nS)
    return (T, R)
end

function bellman_update(mdp, R, T, U)
	nS = mdp.nS
	γ = mdp.γ
	T_τ = mdp.T_τ
	nτ = size(mdp.T_τ, a)

	Qa = R
	for τ_prime in 1:nτ
		Qa[(i-1)*nS+i:i*nS] += γ*sum(T_τ[:,i])*T*U[(i-1)*nS+i:i*nS]
	end
	return Qa
end


