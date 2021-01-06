
module testlucasmodel
include("../src/utils.jl")
include("../src/LucasModel.jl")
using .LucasModel: LucasParameters, LucasHouseholds, ψ_markov!, solveR!
using .LucasModel: solveθbyw̃!, θfoc
using .LucasModel: @unpack_LucasParameters, @unpack_LucasHouseholds
using LinearAlgebra, Test
#########################################
#       Simpliest Lucas
#########################################

p = LucasParameters(
    agrid = [exp(x)-1 for x in 0.05:0.05:5], 
    # ψgrid = [0.1:0.1:0.99;], 
    ψgrid = [1.0;], 
    σx=.0, ρx = 1.0,
    𝔼εy = expectation(Normal(), Gaussian; n = 20)
)
ψ_markov!(p)
solveR!(p)

hh = LucasHouseholds(p)
@unpack_LucasParameters p
@unpack_LucasHouseholds hh
@. hh.V′mat = agrid^(1-γ)/(1-γ)
Vamat = centraldiff(V′mat, agrid)
Vaapproxfunc = partialinterpolate(agrid, Vamat, Linear())
Varealfunc(w, ψ) = w^-γ


θfocerror_real = [θfoc(1.0, x, 1, (Vafunc = Varealfunc, ), p) for x in agrid]
@test norm(θfocerror) < 1e-10

θfocerror_approx = [θfoc(1.0, x, 1, (Vafunc = Vaapproxfunc, ), p) for x in agrid]
@test norm(θfocerror) < 1e-10

θsol_real = [find_zero(x->θfoc(x, a, 1, (Vafunc = Varealfunc, ), p), 0.5) for a in agrid]
@test norm(θsol_real .- 1) < 1e-10

θsol_approx = [find_zero(x->θfoc(x, a, 1, (Vafunc = Vaapproxfunc, ), p), 0.5) for a in agrid]
@test norm(θsol_approx .- 1) < 1e-10



end