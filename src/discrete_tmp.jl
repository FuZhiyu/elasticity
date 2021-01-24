
module LucasModel

p = LucasParameters(
    agrid = [exp(x)-1 for x in 0.01:0.01:5], 
    # ψgrid = [0.1:0.1:0.99;], 
    γ = 4,
    ψgrid = [1.0;], 
    σx=.0, ρx = 1.0,
    σy = 0.1,
    𝔼εy = expectation(Normal(), Gaussian; n = 100)
)

solveR!(p)
p.pdvec
@unpack_LucasParameters p

rhs =  exp((γ - 1) * g - γ * (γ - 1) / 2 * σy^2)
pdconst = 1 / (rhs / β - 1)

end

module LRRModel
p = LRRParameters(
    σgrid = 0.0078 .+ 1e-4 * [-1; 0; 1], 
    nx = 100,
    # maxx = 0.001,
    nσ = 3, 
    # ρ = 1,
    # φe = 0.0,
    γ = 7.5,
    φ = 1.5
)

@unpack_LRRParameters p


res = fixedpoint((out, x)->iteratepd!(out, x, p), p.pdmat, show_trace = true, iterations = 500, m = 4)
p.pdmat .= res.zero
pdfunc = extrapolate(interpolate((xgrid, σgrid), pdmat, Gridded(Linear())), Interpolations.Line())

# out = similar(p.pdmat)
using BenchmarkTools
# @btime iteratepd!(out, pdmat, p)
# Main.@code_warntype __𝔼_pd_inner(pdfunc, 1.0, 1.0, p)
solve𝔼R!(p)
Re = (p.𝔼Rmat - p.Rfmat)
Rf = p.Rfmat

using DataFrames
function simulatemodel(param, T = 100000)
    @unpack_LRRParameters param
    η, e, w = rand(Normal(0, 1), T), rand(Normal(0, 1), T), rand(Normal(0, 1), T)
    xvec, σvec = zeros(T), zeros(T)
    σvec[1] = σ̄
    for i in 2:T
        σvec[i] = σ′func(σvec[i-1], w[i], param)
        xvec[i] = x′func(xvec[i-1], σvec[i], e[i], param)
    end
    dt = DataFrame(t = 1:T, x = xvec, σ = σvec)
    return dt
end

refunc = extrapolate(interpolate((xgrid, σgrid), log.(𝔼Rmat./Rf), Gridded(Linear())), Interpolations.Line())
Refunc = extrapolate(interpolate((xgrid, σgrid), Re, Gridded(Linear())), Interpolations.Line())
Rffunc = extrapolate(interpolate((xgrid, σgrid), Rf, Gridded(Linear())), Interpolations.Line())

sim = simulatemodel(p)
sim.Rf = Rffunc.(sim.x, sim.σ)
sim.year = sim.t .÷ 12
simyear = combine(groupby(sim, :year), :Rf=>prod=>:Rf)
mean(simyear.Rf .- 1) * 100
std(sim.Rf) *sqrt(12)


Rfsim = Rffunc.(xsim, σgrid[2])
rfsim = log.(Rfsim)
mean(rfsim)
std(rfsim)

(mean(Rfsim) - 1) * 12 * 100
std(Rfsim)*sqrt(12)


pdsim = pdfunc.(xsim, σgrid[2])
zsim = log.(pdsim)
std(zsim)

z̄ = mean(log.(pdfunc(xsim, σgrid[2])))
κ = exp(z̄)/(1+ exp(z̄))
κ = 0.997

A1 = (1-1/φ)/(1-κ * ρ)
B = κ * A1 * φe
σa2 = (1 + B^2) * σ̄^2
λmη = -γ
λme = (1-θ) * B
re = -λmη * σ̄^2 + λme * B * σ̄^2 - 0.5 * σa2
resim = refunc.(xsim, σgrid[2])
mean(resim)
Resim = Refunc.(xsim, σgrid[2])
mean(Resim)

using Plots
plot(xgrid, z)

(z[end] - z[1])/(xgrid[end] - xgrid[1])


-log(β) + 1/φ * μ + (1-θ)/θ * re - 1/2/θ * ((λmη^2 + λme^2) * σ̄^2)

mean(rfsim)


iteratepd!(out, pdmat, p)

out

res.zero - pdanalytical

out = similar(pdmat)
iteratepd!(out, pdmat, p)
pdmat
𝔼((η, e, w)->__pd_inner(pd′func, xgrid[1], σgrid[1], η, e, w, p))

end


module HabitModel
using Plots, Test

p = HabitParameters(nshigh = 12, nslow = 18, slowmin = -15, 
agrid = 1 .+ [exp(x) - 1 for x in 0.0:0.02:5]
)

calculatepd!(p)

hh = HabitHouseholds(p)
@unpack_HabitParameters p
@unpack_HabitHouseholds hh

# @. hh.cmat = agrid ./ p.pdvec' + 1

iteratepolicy!(hh, hh.cmat, p)
hh.θmat

cfunc = extrapolate(interpolate((agrid, sgrid), hh.cmat, Gridded(Linear())), Line())
Vafunc = (w, s) -> (cfunc(w, s) + exp(s) - 1)^-γ
θfoc(0.0, agrid[1], 1, (Vafunc = Vafunc, ), p)
θfoc(-1.0, agrid[1], 1, (Vafunc = Vafunc, ), p)




res = fixedpoint((out, x)->iterationhelper!(out, x, hh, p), hh.cmat, m = 0, ftol = 1e-6, show_trace = true);

cmatbackup = copy(hh.cmat)
hh.θmat
hh.cmat

cfunc = extrapolate(interpolate((agrid, sgrid), cmatbackup, Gridded(Linear())), Line())
Vafunc = (w, s) -> (cfunc(w, s) + exp(s) - 1)^-γ

θfoc(0.60, agrid[1], 14, (Vafunc = Vafunc, ), p)
θfoc(0.59, agrid[1], 14, (Vafunc = Vafunc, ), p)
θfoc(0.61, agrid[1], 14, (Vafunc = Vafunc, ), p)


θfocεy.(0.60, agrid[1], 14, nodes(p.𝔼εy), Ref((Vafunc = Vafunc, )), Ref(p)) |> println

p.𝔼εy.nodes[10]

Vafunc.(agrid, sgrid')
cfunc.(agrid, sgrid')
Sgrid[1]


@inline @fastmath function θfocεy(θ, w̃, is, εy, funcs, param::HabitParameters)
    @unpack_HabitParameters param
    @unpack Vafunc = funcs
    s = sgrid[is]
    s′ = s′func(s, εy, param)
    Ygrowth = exp(Δyfunc(εy, param))
    R = Rfunc(s, εy, param)
    Rf = Rfvec[is]
    w′ = w̃ * (θ * R + (1 - θ) * Rf) / Ygrowth
    Va = Vafunc(w′, s′)
    return Va 
end



plot(x->θfoc(x, agrid[1], 14, (Vafunc = Vafunc, ), p), [0.5:0.01:0.7])

cfunc(agrid[1], sgrid[11])


Vafunc(1, sgrid[15])

plot(agrid, Vafunc.(agrid, sgrid[15]))
plot(cmatbackup[:, 14])
plot(Vafunc.(agrid[1:50], sgrid[14]))

θfoc(1e16, agrid[1], 1, (Vafunc = Vafunc, ), p)
p.Rfvec


function Rfunc(s, εy, param)
    @unpack_HabitParameters param
    return 1.08 + εy * σy
end 
p.Rfvec .= 1.02
end


