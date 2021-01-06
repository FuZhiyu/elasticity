p = LucasParameters(
    agrid = [exp(x)-1 for x in 0.05:0.05:5], 
    # ψgrid = [0.1:0.1:0.99;], 
    ψgrid = [1.0;], 
    σx=.0, ρx = 1.0,
    𝔼εy = expectation(Normal(), Gaussian; n = 20)
)
p.Aψ .= ψ_markov(p)
solveR!(p)


@unpack_LucasParameters p
hh = LucasHouseholds(p)
@unpack_LucasHouseholds hh
@. hh.cmat = agrid * (1-β)
@. hh.θmat = 1.0

# Rffunc = extrapolate(interpolate((ψgrid, ), Rfvec, Gridded(Linear())), Line())
# pdfunc = extrapolate(interpolate((ψgrid, ), pdvec, Gridded(Linear())), Line())


@. V′mat = agrid^(1-γ)/(1-γ)
# V′func = extrapolate(interpolate((agrid, ψgrid), V′mat, Gridded(Linear())), Line())
V′func = partialinterpolate(agrid, V′mat)
Vamat = diff(V′mat, dims = 1) ./ diff(agrid)
Vafunc = partialinterpolate(agrid[1:end-1], Vamat, Constant())
Varealfunc(w, ψ) = w^-5.0

Varealfunc(50, 1)
θfoc(0.5, 50.0, 1, (Vafunc = Varealfunc, ), p)
θfoc(1.0, 50.0, 1, (Vafunc = Varealfunc, ), p)
θfoc(2.0, 2.0, 1, (Vafunc = Vafunc, ), p)
Varealfunc(1 , 1)


tmpR = εy -> (pdvec[1] + 1) / pdvec[1] * exp(g - 1/2 * σY^2 + σY * εy)
𝔼εy(εy->tmpR(εy)^-γ * (tmpR(εy) - Rfvec[1]))
Rfvec[1]
pd = pdvec[1]
(pd + 1)/pd * (1+g)
β * (pd + 1)/pd - exp((γ-1) * g - γ * (γ-1)/2 * σY^2)

@time solveθbyw̃!(hh.θmat, Vamat, p)
hh.θmat
Main.@code_warntype θfoc(2.0, 50.0, 1, (Vafunc = Vafunc, ), p)


Main.@code_warntype θfoc(2.0, 50.0, 1, (Vafunc = Vafunc, ), p)
θfoc(10, 0.05, 2, (Vafunc = Vafunc, ), p)
Main.@code_warntype θfocεx(10.0, 147.0, 1, 1, (Vafunc = Vafunc, ), p)
Main.@code_warntype θfocεyεx(10.0, 147.0, 1, -1.0, 1, (Vafunc = Vafunc, ), p)

@btime find_zero(x->θfoc(x, 10.0, 1, (Vafunc = Vafunc, ), p), (0.1, 3.0), Roots.A42(), tol = 1e-4)

# Vafunc = extrapolate(interpolate((agrid[1:end-1], ψgrid), Vamat, Gridded(Linear())), Line())
# Vafunc = extrapolate(interpolate((agrid[1:end-1], ψgrid), Vamat, Gridded(Linear())), Line())
# Vaitp = partialinterpolate(agrid[1:end-1], Vamat, Constant())
# # Vaitp = partialinterpolate(agrid[1:end-1], Vamat)
# funcs = (Vafunc = Vafunc,)
# funcs = (Vafunc = Vaitp,)
# funcs = (V′func = V′func,)

@btime θfoc(1, 1.0, 1, funcs, p)
@btime find_zero(x->θfoc(x, 1, 1, $funcs, $p), (0.2, 2.0), $(Roots.A42()), tol = 1e-2)
@btime find_zero(x->θfoc(x, 1, 1, $funcs, $p), 1.0, tol = 1e-2)

function testthreading_single!(θmat, funcs, p)
    for iter in CartesianIndices(θmat)
        ia, iψ = iter[1], iter[2]
        θmat = find_zero(x->θfoc(x, 1, 1, funcs, p), 1.0, tol = 1e-2)
    end
end

using Base.Threads
nthreads()


plist = [
    LucasParameters(agrid = [exp(x)-1 for x in 0.05:0.05:5], ψgrid = [0.1:0.1:0.99;], σx=.5, ρx = 0.8)
    for _ in 1:nthreads()]
for np in plist
    np.Aψ .= ψ_markov(np)
    np.Rfvec .= p.Rfvec
    np.pdvec .= p.pdvec
end
funcslist = [
    (Vafunc = partialinterpolate(agrid[1:end-1], Vamat),) 
    for _ in 1:nthreads()]

funcslist[1]
function testthreading_mul!(θmat, funcslist, plist)
    @threads for iter in CartesianIndices(θmat)
        it = threadid()
        funcs, p = funcslist[it], plist[it]
        ia, iψ = iter[1], iter[2]
        θmat = find_zero(x->θfoc(x, p.agrid[ia], iψ, funcs, p), 1.0, tol = 1e-2)
    end
end

function testthreading_single!(θmat, funcslist, plist)
    for iter in CartesianIndices(θmat)
        it = threadid()
        funcs, p = funcslist[it], plist[it]
        ia, iψ = iter[1], iter[2]
        θmat = find_zero(x->θfoc(x, p.agrid[ia], iψ, funcs, p), 1.0, tol = 1e-2)
    end
end

function testthreading_spawn!(θmat, funcslist, plist)
    for iter in CartesianIndices(θmat)
        ia, iψ = iter[1], iter[2]
        task = Threads.@spawn find_zero(x->θfoc(x, $(p.agrid[ia]), $iψ, $funcs, $p), 1.0, tol = 1e-2)
        fetch(task)
    end
end
@btime testthreading_single!($hh.θmat, $funcslist, $plist)
@btime testthreading_mul!($hh.θmat, $funcslist, $plist)

@btime testthreading_spawn!($hh.θmat, $funcs, $p)
# test scaling 









@btime θobj(1.0, 1.0, 1, funcs, p)
@btime optimize(x->-θobj(x[1], 1.0, 1, funcs,p), 0.1, 2.0)
# funcs = (Rffunc = Rffunc, pdfunc = pdfunc, V′func = V′func, Vafunc = Vafunc)


obj = x -> -f(x[1], EV′(1.0, 0.6, x[1], x[2], funcs, p), p)
# Main.@code_warntype obj([0.1, 1.0])
θfocεyεx(1, 0.5, 0.1, 1.0, 0.0, 0.5, funcs, p)
θfocεyεx(1, 0.5, 0.1, 0.0, 0.0, 0.5, funcs, p)
θfocεyεx(1, 0.5, 0.1, 0.0, 0.0, 0.5, funcs, p)
θfoc(.5, 1, 0.6, funcs, p)
θfoc(1, 1, 0.6, funcs, p)
@btime θfoc(0.1, 1, 0.6, funcs, p)
@btime Vafunc(1.0, 0.5)

@btime ψ′func(0.5, 0.1, p)
testfunc = εx -> ψ′func(0.5, εx, p)
@btime  $(p.𝔼εx)($testfunc)
p.𝔼εx

p = LucasParameters()
hh = LucasHouseholds(p)
@unpack_LucasParameters p
@unpack_LucasHouseholds hh
tmpfunc = partialinterpolate(agrid, hh.V′mat)
V′func = extrapolate(interpolate((agrid, ψgrid), V′mat, Gridded(Linear())), Line())

ψgrid[3]
ψgrid[70]
@btime V′func(1.5, 0.71)
@btime tmpfunc(1.5, 70)


ψgrid
@btime EV′(0.5, 1, 0.1, funcs, p)

@btime V′εyεx(1, 0.5, 0.1, 0.1, 0.1, 0.1, funcs, p)

@btime V′func(1, 0.5)


using Roots

@btime find_zero(x->θfoc(x, 1, 0.6, $funcs, $p), 1.0, tol = 1e-2)



function ψ_markov(param)
    @unpack_LucasParameters param
    A = zeros(nψ, nψ)
    ψgridmid = (ψgrid[1:end-1] + ψgrid[2:end]) / 2
    ψgridmid = [-Inf; ψgridmid; Inf]
    𝔼εx = expectation(Normal(), Gaussian; n = 1000)
    for i in 1:nψ
        for j in 1:nψ
            A[i, j] = 𝔼εx(x-> ψgridmid[j] < ψ′func(ψgrid[i], x, param) <= ψgridmid[j+1])
        end
    end
    return A
end

p = LucasParameters(ρx = 0.8, σx = 0.1, ψgrid = [0.1:0.05:0.99;])
A = ψ_markov(p)

sum(A, dims = 2)
@show A



using Plots
heatmap(A)



@btime Vafunc(agrid)
Vagrid
Va1func = extrapolate(interpolate((agrid[1:end-1], ), Vamat[:, 1], Gridded(Linear())), Line())
@btime Vamat[1,1]

@btime Vafunc(0.3, 0.4)
@btime Va1func(0.3)

@btime linear_interp(agrid[1:end-1], Vamat[:, 1], agrid[1])

@fastmath function linear_interp(grid, value, t, left_extrap = false, right_extrap = true)
    # grid must be sorted! Don't check here for performance.
    n = length(grid)
    if n == 1
        return value[1]
    end
    i = searchsortedfirst(grid, t)
    if i == 1
        if left_extrap
            return value[1] + (t-grid[1]) * (value[2] - value[1])/(grid[2] - grid[1])
        else
            return value[1]
        end
    elseif i> n
        if right_extrap
            return value[n] + (t-grid[n]) * (value[n] - value[n-1])/(grid[n] - grid[n-1])
        else
            return value[n]
        end
    else
        return value[i-1] + (t-grid[i-1]) * (value[i] - value[i-1])/(grid[i] - grid[i-1])
    end
end

end



function testtype()
    @inbounds for i in 1:10
        i
    end
    return 1
end
Main.@code_warntype testtype()