using GridInterpolations
using POMDPModelTools

# Define the general mdp
struct vMDP
	grid::RectangleGrid
	nS::Int64
	nA::Int64
	γ::Float64
	ḣRanges
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
τs = collect(range(0, step=1, stop=40)) # s

nS = length(hs)*length(ḣ₀s)*length(ḣ₁s)*length(τs)

# Actions
COC = 1
CL1500 = 2
DES1500 = 3

nA = 3

ḣRanges = Dict(COC=>(-100.0,100.0),
                CL1500=>(-100.0,25.0),
                DES1500=>(-25.0,100.0))

nominal_vertical_accel = 10 # ft/s²

# τ stuff
T_τ_init = zeros(length(τs), length(τs))
T_τ_init[1,1] = 1.0
for i = 2:length(τs)
	T_τ_init[i,i-1] = 1
end

function vMDP(;grid=RectangleGrid(hs, ḣ₀s, ḣ₁s), nS=nS, nA=nA, γ=0.99, ḣRanges=ḣRanges, T_τ=T_τ_init)
	return vMDP(grid, nS, nA, γ, ḣRanges, T_τ)
end

"""
----------------------------------
Transition
----------------------------------
"""

function transition(mdp::vMDP, s_ind::Int64, a::Int64)
	hnext, ḣ₀next, ḣ₁next = next_state_vals(mdp, s_ind, a)
	states, probs = interpolants(mdp.grid, [hnext, ḣ₀next, ḣ₁next])
	return states, probs
end

function next_state_vals(mdp::vMDP, s_ind::Int64, a::Int64)
	s_grid = ind2x(mdp.grid, s_ind)
	h, ḣ₀, ḣ₁ = s_grid[1], s_grid[2], s_grid[3]
	ḧ₀ = get_accel(mdp, a, ḣ₀)
	ḧ₁ = get_accel(mdp, a, ḣ₁)

	hnext = h - ḣ₀ - 0.5ḧ₀ + ḣ₁ + 0.5ḧ₁
	ḣ₀next = ḣ₀ + ḧ₀
	ḣ₁next = ḣ₁ + ḧ₁
	return hnext, ḣ₀next, ḣ₁next
end

function get_accel(mdp::vMDP, a::Int64, ḣ::Float64)
	ḣLow, ḣHigh = mdp.ḣRanges[a]
	ḧ = nominal_vertical_accel
    if (ḣLow >= ḣ) .| (ḣHigh <= ḣ)
        ḧ = 0
    elseif ḣLow > ḣ + ḧ
        ḧ = ḣLow-ḣ
    elseif ḣHigh < ḣ + ḧ
        ḧ = ḣHigh-ḣ
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
	h, ḣ₀, ḣ₁, τ = s_grid[1], s_grid[2], s_grid[3], τs[τ_ind]

	r = 0
	# Penalize nmac
	abs(h) < 100 && τ ≤ 1 ? r -= 1 : nothing
	#Penalize alerting
	a != COC ? r -= 0.05 : nothing

	return r
end