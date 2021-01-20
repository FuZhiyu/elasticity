
module SolvingElasticity

#=====================
# Lucas Model with Imperfectly Correlated Labor Income
=====================#
include("LucasModel.jl")
using .LucasModel
p = LucasParameters(
    agrid = [exp(x)-1 for x in 0.01:0.01:5], 
    ψgrid = [0.1:0.02:0.3;], 
    # ψgrid = [0.01:0.005:0.99;], 
    γ = 4,
    σy = 0.1,
    g = 0.04,
    # σx= .1, ρx = 0.95,
    σx= 0, ρx = 1,
    μx = 4,
    β = 0.9,
    𝔼εy = expectation(Normal(), Gaussian; n = 50)
)
solveR!(p)
@show p.Aψ[1,:]
@show financialwealth = @. [(p.pdvec + 1) * p.ψgrid;] # the financial wealth level of the representative household
hh = LucasHouseholds(p)
res = fixedpoint((out, x)->iterationhelper!(out, x, hh, p), hh.cmat, m = 1, ftol = 1e-6, show_trace = true);
;
#======
Test that the solution to the bellman equation is consistent with the steady states:
======#

cfunc = partialinterpolate(p.agrid, hh.cmat)
θfunc = partialinterpolate(p.agrid, hh.θmat)
@show norm(cfunc.(financialwealth, 1:p.nψ) .- 1, Inf)
θ1 = θfunc.(financialwealth, 1:p.nψ)
@show norm(θ1 .- 1, Inf)

#======
solve the elasticity:
======#

hh2 = LucasHouseholds(p)
Δp = 0.001
p.pdvec .*= 1 + Δp
res2 = fixedpoint((out, x)->iterationhelper!(out, x, hh2, p), hh.cmat, m = 1, ftol = 1e-6, show_trace = true)
@assert res2.f_converged
θ2func = partialinterpolate(p.agrid, hh2.θmat)
θ2 = θ2func.(financialwealth, 1:p.nψ)

elas = @. (θ1 - θ2)/Δp

elas_analytical = @. 1/financialwealth/log(p.𝔼R/p.Rfvec)
elas_analytical - elas
elas[6]
financialwealth[6]
@. log(p.𝔼R/p.Rfvec)[6]
end