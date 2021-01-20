
module lucasmodelshowcase
#=====================
# Elasticity in Lucas Models

## Lucas model with perfectly correlated income

First, to verify the algorithm is running properly, let's test it using the simple model
with perfectly correlated labor income, i.e., $\sigma_x = 0$ and $\rho_x = 1$:
=====================#
using Plots, LaTeXStrings
include("src/LucasModel.jl")
using .LucasModel
p = LucasParameters(
    agrid = [exp(x)-1 for x in 0.01:0.01:5], ψgrid = [0.1:0.02:0.3;], 
    γ = 4, σy = 0.1, g = 0.04, μx = 4, β = 0.9, 
    σx= 0, ρx = 1
);
#======
Solve the original stationary equilibrium:
======#
solveR!(p)
hh = LucasHouseholds(p)
res = fixedpoint((out, x)->iterationhelper!(out, x, hh, p), hh.cmat, m = 1, ftol = 1e-8);
θfunc = partialinterpolate(p.agrid, hh.θmat);
#=====
Calculate the portfolio share of the representative households: They should be exactly one:
======#
financialwealth = @. [(p.pdvec + 1) * p.ψgrid;] # the financial wealth level of the representative household
θ1 = θfunc.(financialwealth, 1:p.nψ)
maximum(abs.(θ1 .- 1))
#======
Purturbate the original steady state and solve the new bellman equation. The elasticity is calculated
as the sensitivity of portfolio of a representative household to the P/D ratio:
======#
hh2 = LucasHouseholds(p)
Δp = 0.001
p.pdvec .*= 1 + Δp
res2 = fixedpoint((out, x)->iterationhelper!(out, x, hh2, p), hh.cmat, m = 1, ftol = 1e-6, show_trace = true)
@assert res2.f_converged
θ2func = partialinterpolate(p.agrid, hh2.θmat)
θ2 = θ2func.(financialwealth, 1:p.nψ)
elas = @. (θ1 - θ2)/Δp;
#======
Compare the elasticity calculated from numerical methods and the analytical solutions in the paper:
======#
elas_analytical = @. 1/financialwealth/log(p.𝔼R/p.Rfvec)
plot(financialwealth, [elas elas_analytical], 
    xlabel = L"$\frac{W^\epsilon}{Y}$", ylabel = L"\zeta^r", 
    label = ["Numerical" L"$\frac{Y}{W^\epsilon \pi}$:"]
)


#======
## Lucas model with partially correlated labor income

Consider the following parameterization: $\sigma_x = 0.1, \rho_x = 0.95,  \mu_x = 4$.
Under this parameterization, the stationary distribution of $\psi$ looks like:
======#
σx, ρx, μx = 0.1, 0.95, 4
x_sampled = rand(Normal(0, σx^2/(1-ρx^2)), 100000)
ψ_sampled = @. exp.(x_sampled)/(μx + exp.(x_sampled))
histogram(ψ_sampled, label = L"\psi")

#=====
Therefore it's safe to limit the domain within [0.1,0.3] if we are mainly interested in 
the elasticity around the steady state. In implementation, the $\psi$ process is
discretized as a 21-point markov chain.

Figure below present the P/D ratio and $W^\epsilon/Y$ as functions of $\psi$: 
======#
p = LucasParameters(
    agrid = [exp(x)-1 for x in 0.01:0.02:5], ψgrid = [0.1:0.01:0.3;], 
    γ = 4, σy = 0.1, g = 0.04, μx = 4, β = 0.9, σx= 0.1, ρx = 0.95
)
solveR!(p)
financialwealth = @. [(p.pdvec + 1) * p.ψgrid;] # the financial wealth level of the representative household
p1 = plot(p.ψgrid, p.pdvec, ylabel = "P/D ratio", legend = false)
p2 = plot(p.ψgrid, financialwealth, ylabel = L"$\frac{W^\epsilon}{Y}$", legend = false, xlabel = L"\psi")
plot(p1, p2, layout = (2, 1))

#=====
Using a larger or denser grid won't affect the P/D ratio around the mean of $\psi$ significantly.

Following the same procedure as above I solve the elasticity:
======#

hh = LucasHouseholds(p)
res = fixedpoint((out, x)->iterationhelper!(out, x, hh, p), hh.cmat, m = 1, ftol = 1e-6, show_trace = true);
θfunc = partialinterpolate(p.agrid, hh.θmat)
θ1 = θfunc.(financialwealth, 1:p.nψ)

hh2 = LucasHouseholds(p)
Δp = 0.001
p.pdvec .*= 1 + Δp
res2 = fixedpoint((out, x)->iterationhelper!(out, x, hh2, p), hh.cmat, m = 1, ftol = 1e-6, show_trace = true)
@assert res2.f_converged
θ2func = partialinterpolate(p.agrid, hh2.θmat)
θ2 = θ2func.(financialwealth, 1:p.nψ)
elas = @. (θ1 - θ2)/Δp
plot(financialwealth, elas, xlabel = L"$\frac{W^\epsilon}{Y}$", ylabel = L"\zeta^r", legend = false)

#======
Around the steady state of $\psi=0.2$, $W^\epsilon/Y \approx 1.3$ the elasticity is around 12.
======#
end
