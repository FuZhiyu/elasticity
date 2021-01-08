
module SolvingElasticity

#=====================
# Lucas Model with Imperfectly Correlated Labor Income
=====================#
using Revise
revise()
include("LucasModel.jl")
using .LucasModel
p = LucasParameters(
    agrid = [exp(x)-1 for x in 0.01:0.01:5], 
    # ψgrid = [0.1:0.1:0.99;], 
    ψgrid = [0.1:0.02:0.5;], 
    γ = 4,
    σY = 0.1,
    g = 0.04,
    β = 0.92,
    # σx= .05, ρx = 0.5,
    σx= .0, ρx = 1.0,
    μx = 0.3,
    𝔼εy = expectation(Normal(), Gaussian; n = 50)
)

solveR!(p)
@show financialwealth = @. [(p.pdvec + 1) * p.ψgrid;] # the financial wealth level of the representative household
hh = LucasHouseholds(p)
res = fixedpoint((out, x)->iterationhelper!(out, x, hh, p), hh.cmat, m = 1, ftol = 1e-9, show_trace = true);
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


end