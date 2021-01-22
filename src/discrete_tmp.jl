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