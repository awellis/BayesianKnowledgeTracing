using Distributions, Chain, DataFrames, DataFrameMacros, Query, Turing
using AlgebraOfGraphics, CairoMakie

@model StateSpaceModel(y) = begin
    N = length(y)
    x = tzeros(Real, N)
    σx = 1.0

    x[1] ~ Normal(0, 1)
    y[1] ~ BernoulliLogit(x[1])
    for i ∈ 2:N
        x[i] ~ Normal(x[i-1], σx)
        y[i] ~ BernoulliLogit(x[i])
    end
end

y = [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1]
y_missing = similar(y, Missing)
# y_missing = Array{Missing}(missing, 20)

prior = sample(StateSpaceModel(y_missing), Prior(), 1000)


## solution 1
# prior_x = DataFrame(prior[:x])

f(x) = Symbol(match(r"[0-9]+", string(x)).match)

function extract_first_int(str::AbstractString)
    parse(Int32, match(r"\d+", str).match)
end

prior = DataFrame(prior)
prior_x = prior |> @select(startswith("x")) |> DataFrame
# rename!(f, prior_x)

prior_y = prior |> @select(startswith("y")) |> DataFrame
# rename!(f, prior_y)

prior_x = DataFrames.stack(prior_x, variable_name="t", value_name="x")
prior_y = DataFrames.stack(prior_y, variable_name="t", value_name="y")

prior_x = @chain prior_x begin
    @transform(:t = extract_first_int(:t))
    DataFrame
end

prior_y = @chain prior_y begin
    @transform(:t = extract_first_int(:t))
    DataFrame
end


# prior_x = prior_x |>
    # @mutate(t = extract_first_int.(_.t)) |> DataFrame

# prior_y = prior_y |>
    # @mutate(t = extract_first_int.(_.t)) |> DataFrame
# 

prior_samples = innerjoin(prior_x, prior_y, on="t")

# stack(prior_samples, Not(:t))

gdf = groupby(prior_samples, :t)

m = combine(gdf, [:x, :y] .=> mean)
lb = combine(gdf, [:x, :y] .=> (x -> quantile(x, [0.025])) .=> [:xlb, :ylb])
ub = combine(gdf, [:x, :y] .=> (x -> quantile(x, [0.975])) .=> [:xub, :yub])

summary = innerjoin(m, lb, ub, on=:t)

# plot(plot(summary.x_mean, ribbon=[summary.xlb, summary.xub]),
#     plot(summary.y_mean, ribbon=[summary.ylb, summary.yub]))

# plot(summary.x_mean, ribbon=[summary.xlb, summary.xub], linewidth=2, linestyle=:dot, fillalpha=0.2)
# plot!(summary.y_mean, ribbon=[summary.ylb, summary.yub], linewidth=2, linestyle=:dash, fillalpha=0.1)

@chain summary begin
    data(_) *
        mapping(:t, :x_mean, lower = :xlb, upper = :xub) *
        visual(LinesFill)
end

xy = data(summary) * mapping(:t, :x_mean, lower = :xlb, upper = :xub)
layer = visual(LinesFill)
# fig = draw(layer * xy)
draw(layer * xy)




xy = data(summary) * mapping(:t, :y_mean)
layer = visual(Scatter)
# fig = draw(layer * xy)
draw(layer * xy)



## function
function get_x_y()
    chain = DataFrame(chain)
    chain_x = chain |> @select(startswith("x")) |> DataFrame
# rename!(f, prior_x)

    chain_y = chain |> @select(startswith("y")) |> DataFrame
# rename!(f, prior_y)

    chain_x = DataFrames.stack(chain_x, variable_name="t", value_name="x")
    chain_y = DataFrames.stack(chain_y, variable_name="t", value_name="y")

    chain_x = @chain chain_x begin
        @transform(:t = extract_first_int(:t))
        DataFrame
    end

    chain_y = @chain chain_y begin
        @transform(:t = extract_first_int(:t))
        DataFrame
    end


    chain_samples = innerjoin(chain_x, chain_y, on="t")

# stack(prior_samples, Not(:t))

    gdf = groupby(chain_samples, :t)

    m = combine(gdf, [:x, :y] .=> mean)
    lb = combine(gdf, [:x, :y] .=> (x -> quantile(x, [0.025])) .=> [:xlb, :ylb])
    ub = combine(gdf, [:x, :y] .=> (x -> quantile(x, [0.975])) .=> [:xub, :yub])

    summary = innerjoin(m, lb, ub, on=:t)
    summary
end




## posterior

posterior = sample(StateSpaceModel(y), SMC(), 1000)

posterior = DataFrame(posterior)
chain_x = posterior |> @select(startswith("x")) |> DataFrame

chain_x = DataFrames.stack(chain_x , variable_name="t", value_name="x")

chain_x = @chain chain_x begin
    @transform(:t = extract_first_int(:t))
    DataFrame
end
gdf = groupby(chain_x, :t)

m = combine(gdf, [:x] .=> mean)
lb = combine(gdf, [:x] .=> (x -> quantile(x, [0.25])) .=> [:xlb])
ub = combine(gdf, [:x] .=> (x -> quantile(x, [0.75])) .=> [:xub])
summary = innerjoin(m, lb, ub, on=:t)


# summary = @chain summary begin
#     @mutate(:y = y)
#     DataFrame
# end

summary.y = y

@chain summary begin
    data(_) *
        mapping(:t, :x_mean, lower = :xlb, upper = :xub) *
        visual(LinesFill) +
    data(_) *
        mapping(:t, :y) *
        visual(Scatter, markersize = 10, alpha = 0.5)
    draw
end
