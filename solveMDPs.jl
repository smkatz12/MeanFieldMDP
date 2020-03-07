@everywhere include("vMDP.jl")
@everywhere include("MeanFieldVI.jl")

vmdp = vMDP()
Q = vanilla_VI(vmdp, max_iter=200)