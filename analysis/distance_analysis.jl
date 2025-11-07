using CSV, DataFrames, Distances
using CairoMakie
using LinearAlgebra
using StatsBase


df = CSV.read("queries/permutations.csv", DataFrame)
df[!, :row] = collect(1:size(df, 1))

models = [
	"laion5b_s13b_b90k",
	"ViT-B-32_laion2b_s34b_b79k",
	"ViT-H-14_laion2b_s32b_b79k",
	"ViT-L-14_laion2b_s32b_b82k",
	"ViT-L-14_laion400m_e32",
    "ViT-H-14_rho50_k1_constrained_FARE2",
 	"EVA02-CLIP-L-14-336",
]



# ==== #
using Distances, Statistics
normalized_distances = Dict{String, Float64}()

for (i, model) in enumerate(models)
    v = map(x -> parse.(Float32, split(x, ",")), readlines("features/texts/$(model)_text.csv"))

    # get all base queries
    base_idx = findall(df.variant .== "base")
    rows = df[base_idx, :row]

    # get all vectors for the base queries
    vectors = hcat(v[rows]...)

    # calculate pairwise cosine distances
    dists = pairwise(CosineDist(), vectors; dims = 2)

    # get the mean distance for each base query
    mean_dists = mean(dists; dims = 2)
    println("Model: $model, Mean Cosine Distance: $(mean(mean_dists))")

    normalized_distances[model] = mean(mean_dists)
end

# Save normalized distances to a CSV
norm_df = DataFrame(Model = String[], MeanCosineDistance = Float64[])
for (model, dist) in normalized_distances
    push!(norm_df, (model, dist))
end
CSV.write("analysis/stats/mean_cosine_distances.csv", norm_df)