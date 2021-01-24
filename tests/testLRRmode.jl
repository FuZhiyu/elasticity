
module testLRRmodel
using Test
include("../src/LRRModel.jl")
include("../src/utils.jl")

using .LRRModel

p = LRRParameters(
    σgrid = 0.0078 .+ 1e-4 * [-1; 0; 1], 
    nx = 100,
    nσ = 3, 
    γ = 7.5,
    φ = 0.5
)

p.pdmat .= 50 # initial guess
res = fixedpoint((out, x)->iteratepd!(out, x, p), p.pdmat, show_trace = true, iterations = 500, m = 4)
p.pdmat .= res.zero
solve𝔼R!(p)

sim = simulatemodel(p)

z̄ = mean(log.(sim.pd))
κ = exp(z̄)/(1+ exp(z̄))
A1, re, rf = approximatesolution(p, κ)

@test (mean(sim.re) - re) ./ re < 0.1
@test (mean(log.(sim.Rf)) - rf) ./ rf < 0.1

A1num = (log.(p.pdmat[end, :]) - log.(p.pdmat[1, :])) ./ (p.xgrid[end] - p.xgrid[1])
@test maximum(@. (A1num - A1)/A1) .< 0.1

end