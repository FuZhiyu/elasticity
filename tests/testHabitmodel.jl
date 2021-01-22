
module testhabitmodel
include("../src/habitmodel.jl")
using Plots, Test
using .HabitModel
p = HabitParameters(nshigh = 12, nslow = 18, slowmin = -15)
calculatepd!(p)
plot(exp.(p.sgrid), p.pdvec./4, seriestypes= :line)
p = HabitParameters(nshigh = 100, nslow = 200)
calculatepd!(p)
plot!(exp.(p.sgrid), p.pdvec./4, seriestypes= :line)

@unpack_HabitParameters p

# test riskfree rate
rf = log.(1 ./ 𝔼M.(p.sgrid, Ref(p)))
@unpack_HabitParameters p
rf2 = @.  -log(β) + γ * g - (γ * (1 - φ) - b) / 2 + b * (s̄ - sgrid)
@test norm(rf - rf2) ≈ 0 atol = 1e-8


p.Sgrid

end