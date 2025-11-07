using CSV, DataFrames, StatsBase

# Rank-Biased Overlap (RBO) for two ranked vectors a,b (no ties), depth d, persistence p in (0,1)
function rbo(a::Vector, b::Vector; p=0.98, d=min(length(a), length(b)))
    As, Bs = Set{eltype(a)}(), Set{eltype(b)}()
    sum = 0.0
    for k in 1:d
        push!(As, a[k]); push!(Bs, b[k])
        A = length(intersect(As, Bs))/k
        sum += A * p^(k-1)
    end
    return (1 - p) * sum
end


@inline function flip_any_and_rate(base::AbstractVector, test::AbstractVector; H::Int=3, K::Int=20)
    Hc = min(H, length(base))
    Kc = min(K, length(test))
    base_topH = base[1:Hc]
    test_topK = Set(test[1:Kc])
    misses = count(x -> !(x in test_topK), base_topH)
    return (any = misses > 0 ? 1 : 0, rate = misses / Hc)
end


function compare_rankings(reference, test)
    
    combined = unique(vcat(reference, test))
    idx = Dict(zip(combined, collect(1:length(combined))))
    
    return [
        length(intersect(reference[1:1], test[1:1])),
        length(intersect(reference[1:10], test[1:10])) / 10,
        length(intersect(reference[1:100], test[1:100])) / 100,
        length(intersect(reference[1:1000], test[1:1000])) / 1000,
        corspearman(map(x -> idx[x], reference), map(x -> idx[x], test)),
        rbo(reference, test, p=0.9), # top 10 - 63%
        rbo(reference, test, p=0.99), # top 100 - 63%
        rbo(reference, test, p=0.999), # top 1000 - 63%
        flip_any_and_rate(reference, test, H=1, K=10).any,
        flip_any_and_rate(reference, test, H=1, K=100).any,
        flip_any_and_rate(reference, test, H=1, K=1000).any,
        flip_any_and_rate(reference, test, H=10, K=100).rate,
        flip_any_and_rate(reference, test, H=10, K=1000).rate,
        flip_any_and_rate(reference, test, H=100, K=1000).rate,
    ]

end


models = [
	"laion5b_s13b_b90k",
	"ViT-B-32_laion2b_s34b_b79k",
	"ViT-L-14_laion400m_e32",
	"ViT-L-14_laion2b_s32b_b82k",
	"ViT-H-14_laion2b_s32b_b79k",
	"ViT-H-14_laion2b_s32b_b79k/ViT-H-14_rho50_k1_constrained_FARE2",
 	"EVA02-CLIP-L-14-336",
]



sorted_variants = [
    "base",
    "lower",
    "upper",
    "punctuation",
    "keyboard",
    "remove",
    "repeat",
    "swap",
    "n_only",
    "shuffle_nouns",
    "n+adj_only",
    "shuffle_n+adj",
    "content_only",
    "shuffle_keywords",
    "synonyms",
    "rephrase",
]


df = CSV.read("queries/permutations.csv", DataFrame)
df[!, :query] = collect(1:size(df, 1))

using ProgressBars

stats = DataFrame[]
agg = DataFrame[]

for model in models
    if contains(model, "/")
        _, model = split(model, "/")
    end
    println("Processing model: $model")

    if isfile("analysis/stats/ranking_statistics_$(model).csv")
        println("Skipping $model, results already exist.")
        results = CSV.read("analysis/stats/ranking_statistics_$(model).csv", DataFrame)
    else
        rankings = CSV.read("analysis/knn/$(model)_results.csv", DataFrame)
        rankings = leftjoin(rankings, df[:, [:id, :query, :variant]], on = :query)

        results = DataFrame[]

        max_id = 190
        for id in ProgressBar(1:max_id)
            group = rankings[isequal.(rankings.id, id), :]
            reference = group[group[:, :variant] .== "base", :element]
            queries = unique(group[:, :query])[2:end]

            for q in queries
                test = group[group[:, :query] .== q, :element]
                metrics = compare_rankings(reference, test)
                push!(results, DataFrame(id = id, query = q, o1 = metrics[1], o10 = metrics[2], o100 = metrics[3], o1000 = metrics[4], 
                    spearman = metrics[5],
                    rbo90 = metrics[6], rbo99 = metrics[7], rbo999 = metrics[8], stability = (metrics[6] + metrics[7] + metrics[8]) / 3,
                    flip1_out10 = metrics[9], flip1_out100 = metrics[10], flip1_out1000 = metrics[11], 
                    flip10_out100 = metrics[12], flip10_out1000 = metrics[13], flip100_out1000 = metrics[14],
                ))
            end
        end

        results = DataFrame(vcat(results...) )
        # print results head
        results = leftjoin(results, df[:, [:query, :variant]], on = :query)
        results[!, :model] = [model for i in 1:size(results, 1)]
        CSV.write("analysis/stats/ranking_statistics_$(model).csv", results)
    end
    push!(stats, results)

    aggregated = combine(groupby(results, :variant),
        :o1 => mean, :o10 => mean, :o100 => mean, :o1000 => mean, :spearman => mean, :stability => mean,
        :o1 => std, :o10 => std, :o100 => std, :o1000 => std, :spearman => std, :stability => std,
        :o1 => median, :o10 => median, :o100 => median, :o1000 => median, :spearman => median, :stability => median,
        :o1 => minimum, :o10 => minimum, :o100 => minimum, :o1000 => minimum, :spearman => minimum, :stability => minimum,
        :o1 => maximum, :o10 => maximum, :o100 => maximum, :o1000 => maximum, :spearman => maximum, :stability => maximum,
    )
    aggregated[!, :model] = [model for i in 1:size(aggregated, 1)]

    push!(agg, aggregated)
end

stats = vcat(stats...)
# rename :query to :row
stats = rename(stats, :query => :row)
CSV.write("analysis/stats/ranking_statistics.csv", stats)

agg = vcat(agg...)
CSV.write("analysis/stats/ranking_statistics_aggregated.csv", agg)

# --------- #
# Print table of stats
all_agg = combine(groupby(stats, :model), :stability => mean,
    :rbo90 => mean, :rbo99 => mean, :rbo999 => mean,
    :flip1_out10 => mean, :flip1_out100 => mean, :flip1_out1000 => mean,
    :flip10_out100 => mean, :flip10_out1000 => mean, :flip100_out1000 => mean,
    :o1 => mean, :o10 => mean, :o100 => mean, :o1000 => mean,
    :spearman => mean
)
display(all_agg)
CSV.write("analysis/stats/ranking_statistics_summary.csv", all_agg)