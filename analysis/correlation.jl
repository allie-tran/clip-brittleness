using CSV
using DataFrames
using Loess, Plots
using StatsBase
using Statistics
using PrettyTables
using StatsPlots
using MixedModels, StatsModels
using DataFrames, StatsPlots
using CategoricalArrays
using Measures

variant_names = Dict(
	"base" => "base",
	"lower" => "lowercase",
	"upper" => "uppercase",
	"punctuation" => "add punctuation",
	"keyboard" => "keyboard layout",
	"remove" => "remove characters",
	"repeat" => "repeat characters",
	"swap" => "swap characters",
	"n_only" => "nouns only",
	"shuffle_nouns" => "shuffle nouns",
	"n+adj_only" => "nouns + adjectives only",
	"shuffle_n+adj" => "shuffle nouns + adjectives",
	"content_only" => "keywords only",
	"shuffle_keywords" => "shuffle keywords",
	"synonyms" => "synonyms",
	"rephrase" => "rephrase",
)

# Plot all models per variant
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

model_short_names = Dict(
	"laion5b_s13b_b90k" => "LAION5B",
	"ViT-B-32_laion2b_s34b_b79k" => "LAION-B32",
	"ViT-L-14_laion2b_s32b_b82k" => "LAION-L14",
	"ViT-L-14_laion400m_e32" => "LAION-L14-400M",
	"ViT-H-14_laion2b_s32b_b79k" => "LAION-H14",
	"ViT-H-14_rho50_k1_constrained_FARE2" => "FARE2-H14",
	"EVA02-CLIP-L-14-336" => "EVA02-L14",
)

sorted_models = [
	"laion5b_s13b_b90k",
	"ViT-B-32_laion2b_s34b_b79k",
	"ViT-L-14_laion400m_e32",
	"ViT-L-14_laion2b_s32b_b82k",
	"ViT-H-14_laion2b_s32b_b79k",
	"ViT-H-14_rho50_k1_constrained_FARE2",
 	"EVA02-CLIP-L-14-336",
]

# Analysis per class
variant_to_class = Dict(
    # Class 1: surface/noise
    "upper" => 1, "punctuation" => 1, "keyboard" => 1, "remove" => 1,
    "repeat" => 1, "swap" => 1, "lower" => 1,

    # Class 2: semantic
    "synonyms" => 3, "rephrase" => 3,

    # Class 3: structural/extractive
    "n_only" => 2, "shuffle_nouns" => 2,
    "n+adj_only" => 2, "shuffle_n+adj" => 2,
    "content_only" => 2, "shuffle_keywords" => 2
)


# ================================ #
# Prepare the data
text_stats = CSV.read("analysis/stats/text_features.csv", DataFrame)
nobase = filter(row -> row.variant != "base", text_stats)
rank_stats = CSV.read("analysis/stats/ranking_statistics.csv", DataFrame)
all_stats = leftjoin(nobase, rank_stats, on = [:id, :variant, :model, :row])
normalized_distance = CSV.read("analysis/stats/mean_cosine_distances.csv", DataFrame)

text_metrics = [:norm_lev, :tok_jacc, :tok_dist, :punct_delta, :len_ratio, :distance]
rank_metrics = [:o1, :o10, :o100, :o1000, :spearman, :rbo90, :rbo99, :rbo999, :stability,
	:flip1_out10, :flip1_out100, :flip1_out1000,
	:flip10_out100, :flip10_out1000, :flip100_out1000]
df = all_stats

# Sort df by model
df.model = categorical(df.model, levels = sorted_models)
df.variant = categorical(df.variant, levels = sorted_variants)
df = sort(df, [:id, :variant, :model])

# Inverse some metrics for consistency
df.o1_inv = 1 .- df.o1
df.o10_inv = 1 .- df.o10
df.o100_inv = 1 .- df.o100
df.o1000_inv = 1 .- df.o1000
df.spearman_inv = 1 .- df.spearman
df.rbo90_inv = 1 .- df.rbo90
df.rbo99_inv = 1 .- df.rbo99
df.rbo999_inv = 1 .- df.rbo999
df.stability_inv = 1 .- df.stability
df.norm_distance = get.(Ref(Dict(zip(normalized_distance.Model, normalized_distance.MeanCosineDistance))), df.model, missing)
df.norm_distance = df.distance ./ df.norm_distance

# ================================ #
# Quick correlation check (optional):
features = [
	:norm_lev, :tok_dist, :punct_delta, :len_ratio, :distance,
	:o1_inv, :o10_inv, :o100_inv, :o1000_inv, :spearman_inv,
	:rbo90_inv, :rbo99_inv, :rbo999_inv, :stability_inv,
	:flip1_out10, :flip1_out100, :flip1_out1000,
	:flip10_out100, :flip10_out1000, :flip100_out1000,
]

fmat = Matrix{Float64}(coalesce.(Matrix(select(df, features)), NaN))
X = Matrix(dropmissing(select(df, features)))
cors = cor(X; dims = 1)   # correlation matrix
# Display the correlation matrix with labels
pretty_table(cors, header = names(df[!, features]), title = "Correlation Matrix of Text Features and Ranking Metrics")

# Heatmap of correlations
gr(size = (1200, 800))  # Set the size of the plot
heatmap(cors, title = "Correlation Heatmap", xlabel = "Features", ylabel = "Features",
	color = :YlGnBu, aspect_ratio = 1,
	xticks = (1:length(features), names(df[!, features])),
	xrotation = 90,
	yticks = (1:length(features), names(df[!, features])),
	cbar_title = "Correlation Coefficient",
	cbar_label = "Correlation",
	cbar_ticks = (-1, 0, 1),
)

# ================================ #
# Merging features with high correlation
# Rank metric: averaging :o1, :o10, :o100, :o1000 and :rbo90, :rbo99, :rbo999
# Flip metrics: avereging :flip1_out10, :flip1_out100, :flip1_out1000,
#               :flip10_out100, :flip10_out1000, :flip100_out
# Edit features: averaging z scores of :norm_lev, :tok_dist, :punct_delta, :len_ratio,
# Distance feature: :distance

# Define the new metrics
df.instability = 1 .- df.rbo99
df.catastrophic_visibility = df.flip10_out100

# ====================================== #

# A) Per-model Distribution
# boxplot per model
gr(size = (1200, 600))  # Set the size of the plot
@df df boxplot(
	:model, :instability;
	ylabel = "Instability (1 - RBO@0.99)",
    xticks = (1:7, [model_short_names[m] for m in sorted_models]),
	xlabel = "Model",
	permute = (:x, :y), # swap x and y axes
	legend = false,
	title = "Instability distributions for permutation variants",
	linecolor = :black,
    margin = 5mm,
	fillalpha = 0.4,
    label= "",
	size = (1000, 400),
)

# add a horizontal zero line
vline!([0.0], color = :red, linestyle = :dash, labels   = "Zero line")
# add a vertical line at the median
vline!([median(df.instability)], color = :blue, linestyle = :dash, label = "Median")
# add a vertical line at the mean
vline!([mean(df.instability)], color = :green, linestyle = :dash, label = "Mean")
# Add legend
plot!(legend = :outerright, leftmargin = 5mm, bottommargin = 5mm)
# Save the plot
savefig("plots/instability_per_model.pdf")

# print the instability means per model
instability_means = combine(groupby(df, :model), :instability => median => :median_instability)
pretty_table(instability_means, header = ["Model", "Median Instability"], title = "Median Instability per Model")

# B) Mixed-effects

# per model
m_models = fit(MixedModel, @formula(instability ~ 1 + model + (1|id) + (1|variant)), df)

# per variant
df.model = string.(df.model)
m_variants = fit(MixedModel, @formula(instability ~ 1 + variant + (1|id) + (1|model)), df)


# ========================================= #
# DISTANCE ANALYSIS
df.model = string.(df.model)
m = fit(MixedModel, @formula(instability ~ 1 + distance + variant + (1|id) + (1|model)), df)

df_sample = df[sample(1:nrow(df), Int(0.1 * nrow(df)), replace = false), :]
df_sample.distance = df_sample.norm_distance

# asign colors to models
colors = Dict(
	"laion5b_s13b_b90k" => "#8da0cb",
	"ViT-B-32_laion2b_s34b_b79k" => "#e78ac3",
	"ViT-L-14_laion400m_e32" => "#ffd92f",
	"ViT-L-14_laion2b_s32b_b82k" => "#e5c494",
	"ViT-H-14_laion2b_s32b_b79k" => "#fc8d62",
	"ViT-H-14_rho50_k1_constrained_FARE2" => "#09bd72",
 	"EVA02-CLIP-L-14-336" => "#b83757",
)


gr(size = (1200, 400))  # Set the size of the plot
plt = @df df_sample StatsPlots.scatter(
	:distance, :instability,
	ms = 3, alpha = 0.1,
	legend = false,
    label = "Data points",
	xlabel = "CLIP distance",
	ylabel = "Instability",
	color="#a3dbe6"
)

for m in unique(df.model)
	subdf = df[df.model .== m, :]
	subdf.distance = subdf.norm_distance

	# fit LOESS for this model
	lo = loess(subdf.distance, subdf.instability; span = 0.3)
	xs = range(minimum(subdf.distance), maximum(subdf.distance), length = 200)
	ys = Loess.predict(lo, xs)

	# add smoothed line
	StatsPlots.plot!(plt, xs, ys, lw = 2, label = model_short_names[m], linestyle = :dash, 
        xlabel = "CLIP distance", ylabel = "Instability",
        title = "Instability vs CLIP distance by model",
		color = colors[m],
		opacity = 0.8
    )

end
plot!(plt, legend = :bottomright, margin = 5mm, bottommargin = 10mm, xlims=(0.0, 0.5))
savefig("plots/instability_distance_analysis.pdf")

# ========================================= #
# Brittleness analysis
function brittleness(instability, distance; eps=1e-6)
    if distance == 0
        return 0.0
    else
        return log(instability / (distance + eps))
    end
end
df.brittleness = map(brittleness, df.instability, df.norm_distance)

function toRomanNumeral(n::Int)
	if n == 1
		return "I"
	elseif n == 2
		return "II"
	elseif n == 3
		return "III"
	else
		return ""
	end
end


# remove base
stats = filter(row -> row.variant != "base", df)
agg = combine(groupby(stats, [:model, :variant]), :brittleness => mean => :mean_distance)
wide = unstack(agg, :variant, :mean_distance)  # rows: model, columns: variant
M = Matrix{Float64}(coalesce.(Matrix(select(wide, Not(:model))), NaN))  # missings -> NaN
plt = heatmap(M, xlabel = "Variant", ylabel = "Model", c = :YlGnBu, title = "Mean brittleness by variant & model", colorbar_title = "Mean brittleness", 
    xticks = (map(x -> Float64(x) + 0.5, 1:size(M, 2)), ["$(variant_names[v]) ($(toRomanNumeral(variant_to_class[v])))" for v in sorted_variants[2:end]]),
	yticks = (map(x -> Float64(x) + 0.5, 1:size(M, 1)), [model_short_names[m] for m in wide.model]),
    xrotation = 45,
    aspect_ratio = 1,
    foreground_color_border = :white,
    tickdirection = :none,
    size = (1147, 600),
    margin = 5mm
)
savefig("plots/brittleness_heatmap.pdf")

# ========================================= #
# EXTRA PLOTS
df.class_idx = get.(Ref(variant_to_class), df.variant, missing)

# Heatmap
agg = combine(groupby(df, [:model, :class_idx]), :brittleness => mean => :mean_distance)
wide = unstack(agg, :class_idx, :mean_distance)  # rows: model, columns: variant
M = Matrix{Float64}(coalesce.(Matrix(select(wide, Not(:model))), NaN))  # missings -> NaN
gr(size = (500, 600))  # Set the size of the plot
plt = heatmap(M, xlabel = "Variant", ylabel = "Model", c = :YlGnBu, title = "Mean brittleness by variant & model", colorbar_title = "Mean brittleness", 
    aspect_ratio = 1,
    xticks = (map(x -> Float64(x), 1:size(M, 2)), ["Class 1", "Class 2", "Class 3"]),
	yticks = (map(x -> Float64(x) + 0.5, 1:size(M, 1)), wide.model))
StatsPlots.plot!(plt, legend = :bottomleft, margin = 5mm, bottommargin = 10mm)

# Grouped boxplot
@df df groupedboxplot(:class_idx, :distance,
    group = :model,
    ylabel="Distance", xrotation=45,
    size = (1200, 400),
    legend = :outerright,
    xlabel = "Class",
    xticks = (1:3, ["Class 1", "Class 2", "Class 3"]),
    margin= 5mm,
)

# Calculate mean catastrophic_visibility per variant and model
agg = combine(groupby(df, [:model, :variant]), :catastrophic_visibility => mean => :mean_catastrophic_visibility)

# Draw spiderchart
using PlotlyJS
fig = PlotlyJS.Plot()
for model in sorted_models
    PlotlyJS.addtraces!(fig, scatterpolar(
        r=agg[agg.model .== model, :mean_catastrophic_visibility],
        theta=agg[agg.model .== model, :variant],
        fill="toself",
        name=model
    ))
end
fig.layout = Layout(
    title = "Catastrophic visibility by variant and model",
    polar = attr(radialaxis = attr(visible = true, range = [0, 0.8])),
    showlegend = true,
    width = 1200,
    height = 600,
)
display(fig)
agg[agg.model .== "laion5b_s13b_b90k", :mean_catastrophic_visibility]