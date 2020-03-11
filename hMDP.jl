# Define the general mdp
struct hMDP
	grid::RectangleGrid
	nS::Int64
	nA::Int64
	γ::Float64
	vh
	turns
	T_τ
end

"""
----------------------------------
Constants
----------------------------------
"""

rs = [0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 510.0, 750.0, 1000.0, 1500.0,
2000.0, 3000.0, 4000.0, 5000.0, 7000.0, 9000.0, 11000.0, 13000.0, 17000.0, 21000.0, 25000.0,
30000.0, 35000.0, 40000.0, 56000.0] # ft
θs = collect(range(-π, stop=π, length=31)) # rad
ψs = collect(range(-π, stop=π, length=31)) # rad
vτs = collect(range(0, step=1, stop=60)) # s

nSh = length(rs)*length(θs)*length(ψs)*length(vτs)

# Actions
COC = 1
SL = 2
SR = 3

hactions = [COC, SL, SR]

nAh = length(hactions)

turns = Dict(COC=>([0.34, 0.33,0.33], [0.0,1.5,-1.5].*π/180.0),
			 SL=>([0.5,0.25,0.25], [3.0,4.0,2.0].*π/180.0),
			 SR=>([0.5,0.25,0.25], [-3.0,-4.0,-2.0].*π/180.0))

# τ stuff
T_vτ_init = zeros(length(vτs), length(vτs))
T_vτ_init[1,1] = 1.0
for i = 2:length(vτs)
	T_vτ_init[i,i-1] = 1
end

vh = 200 # ft/s

function hMDP(;grid=RectangleGrid(rs, θs, ψs), nS=nSh, nA=nAh, γ=0.99, vh=vh, turns=turns, T_τ=T_vτ_init)
	return hMDP(grid, nS, nA, γ, vh, turns, T_τ)
end

"""
----------------------------------
Transition
----------------------------------
"""

function transition(mdp::hMDP, s_ind::Int64, a::Int64)
	states = zeros(72)
	probs = zeros(72)
	startind = 1

	ownProbs, ownTurns = mdp.turns[a]
	intProbs, intTurns = mdp.turns[COC]

	for i = 1:length(ownTurns)
		for j = 1:length(intTurns)
			rnext, θnext, ψnext = next_state_vals(mdp, s_ind, ownTurns[i], intTurns[j])
			s, p = interpolants(mdp.grid, [rnext, θnext, ψnext])
			if length(s) < 8
				s = [s; zeros(8 - length(s))]
				p = [p; zeros(8 - length(p))]
			end
			states[startind:startind+7] = s
			probs[startind:startind+7] = ownProbs[i]*intProbs[j] .* p
		end
	end

	inds = findall(probs .!= 0)
	return states[inds], probs[inds]
end

function next_state_vals(mdp::hMDP, s_ind::Int64, ownTurn, intTurn)
	s_grid = ind2x(mdp.grid, s_ind)
	r, θ, ψ = s_grid[1], s_grid[2], s_grid[3]
	v = mdp.vh

	x₀ = 0.0; y₀ = 0.0; x₁ = r*cos(θ); y₁ = r*sin(θ)
	ẋ₀ = v; ẏ₀ = 0.0; ẋ₁ = v*cos(ψ); ẏ₁ = v*sin(ψ)
	x₀ += ẋ₀; y₀ += ẏ₀; x₁ += ẋ₁; y₁ += ẏ₁

	x₁next = x₁ - x₀; y₁next = y₁ - y₀
	heading₀ = ownTurn; heading₁ = ψ + intTurn

	rnext = √(x₁next^2 + y₁next^2)
	θnext = atan(y₁next, x₁next) - heading₀
	ψnext = heading₁ - heading₀

	# This just normalizes the angle
    θnext = θnext > π  ? θnext - 2*π : θnext
    θnext = θnext < -π ? θnext + 2*π : θnext
    ψnext = ψnext > π  ? ψnext - 2*π : ψnext
    ψnext = ψnext < -π ? ψnext + 2*π : ψnext

    return rnext, θnext, ψnext
end

"""
----------------------------------
Rewards
----------------------------------
"""

function reward(mdp::hMDP, s_ind::Int64, τ_ind::Int64, a::Int64)
	s_grid = ind2x(mdp.grid, s_ind)
	r, θ, ψ, τ = s_grid[1], s_grid[2], s_grid[3], vτs[τ_ind]
	v = mdp.vh

	# Calculating stuff
	x₁ = r*cos(θ); y₁ = r*sin(θ)
	ẋ₁ = v*cos(ψ) - v; ẏ₁ = v*sin(ψ)
	dv2 = ẋ₁^2 + ẏ₁^2
    dCPA = r
    tCPA = 0.0
    if dv2 > 0.0
        tCPA = (-x₁*ẋ₁ - y₁*ẏ₁)/dv2
        xT = x₁ + ẋ₁*tCPA
        yT = y₁ + ẏ₁*tCPA
        if tCPA>0.0
            dCPA = sqrt(xT*xT + yT*yT)
        else
            tCPA=0.0
        end
    end

	reward = 0
	# Penalize nmac
	r ≤ 500.0 && τ < 1 ? reward -= 1 : nothing
	# Penalize closeness
	r > 500.0 ? reward -= 0.5exp(-(r-500.0)/500.0) : nothing
	# Penalize alerting or not alerting
	if a != COC
		reward -= 0.01
	else
		reward -= 0.03exp(-dCPA/500.0)*exp(-tCPA/10.0)
	end

	return reward
end