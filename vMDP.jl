# Define the general mdp
struct vMDP
	grid::RectangleGrid
	nS::Int64
	nA::Int64
	γ::Float64
	ḣRanges
	accels
	T_τ
end

"""
----------------------------------
Constants
----------------------------------
"""

# States
hs = vcat(LinRange(-8000,-4000,3),LinRange(-3000,-1250,6),
	LinRange(-1000,-800,3),LinRange(-700,-150,9),
	LinRange(-100,100,7),LinRange(150,700,9),
	LinRange(800,1000,3),LinRange(1250,3000,6),
	LinRange(4000,8000,3)) # ft
ḣ₀s = vcat(LinRange(-100,-60,3),LinRange(-50,-35,3),
	LinRange(-30,30,10),LinRange(35,50,3),
	LinRange(60,100,3)) #ft/s
ḣ₁s = vcat(LinRange(-100,-60,3),LinRange(-50,-35,3),
	LinRange(-30,30,10),LinRange(35,50,3),
	LinRange(60,100,3)) #ft/s
hτs = collect(range(0, step=1, stop=40)) # s

nSv = length(hs)*length(ḣ₀s)*length(ḣ₁s)*length(hτs)

# Actions
COC = 1
CL1500 = 2
DES1500 = 3

vactions = [COC, CL1500, DES1500]

nAv = length(vactions)

ḣRanges = Dict(COC=>(0, 0),
                CL1500=>(-Inf, 25.0),
                DES1500=>(-25.0, Inf))

accels = Dict(COC=>([0.5,0.25,0.25], [0.0,3.0,-3.0]),
			  CL1500=>([0.5,0.25,0.25], [8.33,9.33,7.33]),
			  DES1500=>([0.5,0.25,0.25], [-8.33,-9.33,-7.33]))

nominal_vertical_accel = 10 # ft/s²

# τ stuff
T_hτ_init = zeros(length(hτs), length(hτs))
T_hτ_init[1,1] = 1.0
for i = 2:length(hτs)
	T_hτ_init[i,i-1] = 1
end

function vMDP(;grid=RectangleGrid(hs, ḣ₀s, ḣ₁s), nS=nSv, nA=nAv, γ=0.99, ḣRanges=ḣRanges, accels=accels, T_τ=T_hτ_init)
	return vMDP(grid, nS, nA, γ, ḣRanges, accels, T_τ)
end

"""
----------------------------------
Transition
----------------------------------
"""

function transition(mdp::vMDP, s_ind::Int64, a::Int64)
    states = zeros(72)
    probs = zeros(72)
    startind = 1

    ownProbs, ownAccels = mdp.accels[a]
    intProbs, intAccels = mdp.accels[COC]

    for i = 1:length(ownAccels)
        for j = 1:length(intAccels)
            hnext, ḣ₀next, ḣ₁next = next_state_vals(mdp, s_ind, a, ownAccels[i], intAccels[j])
            s, p = interpolants(mdp.grid, [hnext, ḣ₀next, ḣ₁next])
            if length(s) < 8
            	s = [s; zeros(8 - length(s))]
            	p = [p; zeros(8 - length(p))]
            end
            states[startind:startind+7] = s
            probs[startind:startind+7]  = ownProbs[i]*intProbs[j] .* p
            startind += 8
        end
    end

	inds = findall(probs .!= 0)
	return states[inds], probs[inds]
end

function next_state_vals(mdp::vMDP, s_ind::Int64, a::Int64, ḧ₀, ḧ₁)
	s_grid = ind2x(mdp.grid, s_ind)
	h, ḣ₀, ḣ₁ = s_grid[1], s_grid[2], s_grid[3]
	ḧ₀ = get_accel(mdp, a, ḣ₀, ḧ₀)
	ḧ₁ = get_accel(mdp, COC, ḣ₁, ḧ₁)

	hnext = h - ḣ₀ - 0.5ḧ₀ + ḣ₁ + 0.5ḧ₁
	ḣ₀next = ḣ₀ + ḧ₀
	ḣ₁next = ḣ₁ + ḧ₁
	return hnext, ḣ₀next, ḣ₁next
end

function get_accel(mdp::vMDP, a::Int64, ḣ::Float64, ḧ::Float64)
	ḣLow, ḣHigh = mdp.ḣRanges[a]
    if ((ḣ ≤ ḣLow) .| (ḣ ≥ ḣHigh)) .& (a != COC)
        ḧ = 0
    elseif a == DES1500
    	if ḣ + ḧ < ḣLow
    		ḧ = ḣLow - ḣ
    	end
    elseif a == CL1500
    	if ḣ + ḧ > ḣHigh
    		ḧ = ḣHigh - ḣ
    	end
    end
    return ḧ
end

"""
----------------------------------
Rewards
----------------------------------
"""

function reward(mdp::vMDP, s_ind::Int64, τ_ind::Int64, a::Int64)
	s_grid = ind2x(mdp.grid, s_ind)
	h, ḣ₀, ḣ₁, τ = s_grid[1], s_grid[2], s_grid[3], hτs[τ_ind]

	r = 0
	# Penalize nmac
	abs(h) < 100 && τ ≤ 1 ? r -= 1 : nothing
	# Penalize alerting
	a != COC ? r -= 0.01 : nothing

	return r
end