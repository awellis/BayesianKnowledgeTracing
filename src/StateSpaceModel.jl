using Distributions, DataFrames, Query, Turing

@model StateSpaceModel(y) = begin
    N = length(y)
    x = tzeros(Real, N)
    σy = 1.0
    σx = 1.0

    x[1] ~ Normal(0, 1)
    y[1] ~ Normal(x[1], σy)
    for i ∈ 2:N
        x[i] ~ Normal(x[i-1], σx)
        y[i] ~ Normal(x[i], σy)
    end
end

prior = sample(StateSpaceModel(Array{Missing}(missing, 20)), Prior(), 1000)


## solution 1
# prior_x = DataFrame(prior[:x])

f(x) = Symbol(match(r"[0-9]+", string(x)).match)

prior = DataFrame(prior)
prior_x = prior |> @select(startswith("x")) |> DataFrame
rename!(f, prior_x)

prior_y = prior |> @select(startswith("y")) |> DataFrame
rename!(f, prior_y)

prior_x = stack(prior_x, variable_name = "t", value_name = "x")
prior_y = stack(prior_y, variable_name = "t", value_name = "y")

prior_samples = innerjoin(prior_x, prior_y, on = "t")

# stack(prior_samples, Not(:t))


gdf = groupby(prior_samples, :t)

m = combine(gdf, [:x, :y] .=> mean)
lb = combine(gdf, [:x, :y] .=> (x -> quantile(x, [0.025])) .=> [:xlb, :ylb])
ub = combine(gdf, [:x, :y] .=> (x -> quantile(x, [0.975])) .=> [:xub, :yub])

summary = innerjoin(m, lb, ub, on = :t)

plot(plot(summary.x_mean, ribbon = [summary.xlb, summary.xub]),
     plot(summary.y_mean, ribbon = [summary.ylb, summary.yub]))

plot(summary.x_mean, ribbon = [summary.xlb, summary.xub], linewidth = 2, linestyle = :dot, fillalpha=.2)
plot!(summary.y_mean, ribbon = [summary.ylb, summary.yub], inewidth = 2, linestyle = :dash, fillalpha=.1)



## solution 2