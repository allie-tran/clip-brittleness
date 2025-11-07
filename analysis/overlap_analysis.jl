using CSV
using DataFrames
using StatsPlots
using Statistics
using Measures

model_short_names = Dict(
	"laion5b_s13b_b90k" => "LAION5B",
	"ViT-B-32_laion2b_s34b_b79k" => "LAION-B32",
	"ViT-L-14_laion2b_s32b_b82k" => "LAION-L14",
	"ViT-L-14_laion400m_e32" => "LAION-L14-400M",
	"ViT-H-14_laion2b_s32b_b79k" => "LAION-H14",
	"ViT-H-14_rho50_k1_constrained_FARE2" => "FARE2-H14",
	"EVA02-CLIP-L-14-336" => "EVA02-L14",
)

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

sorted_models = [
	"laion5b_s13b_b90k",
	"ViT-B-32_laion2b_s34b_b79k",
	"ViT-L-14_laion400m_e32",
	"ViT-L-14_laion2b_s32b_b82k",
	"ViT-H-14_laion2b_s32b_b79k",
	"ViT-H-14_rho50_k1_constrained_FARE2",
 	"EVA02-CLIP-L-14-336",
]

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

# Analysis per class
variant_to_class = Dict(
    # Class 1: surface/noise
    "upper" => 1, "punctuation" => 1, "keyboard" => 1, "remove" => 1,
    "repeat" => 1, "swap" => 1, "lower" => 1,

    # Class 2: semantic
    "synonyms" => 2, "rephrase" => 2,

    # Class 3: structural/extractive
    "n_only" => 3, "shuffle_nouns" => 3,
    "n+adj_only" => 3, "shuffle_n+adj" => 3,
    "content_only" => 3, "shuffle_keywords" => 3
)


df = CSV.read("analysis/stats/ranking_statistics.csv", DataFrame)
df.class_idx = get.(Ref(variant_to_class), df.variant, missing)

df_long = stack(df, [:o1, :o10, :o100, :o1000, :spearman],
	[:id, :row, :variant, :model];
	variable_name = :metric,
	value_name = :value)
df_long.query = df_long.row

# Rename model and variant columns for clarity
df_long.model = map(x -> get(model_short_names, x, x), df_long.model)
df_long.variant = map(x -> get(variant_names, x, x), df_long.variant)


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


function plot_line_overlap(model, variant, legend, color)
	# Filter data for the specific model and variant
	model_data = filter(row -> row.model == model && row.variant == variant, df_long)
	
	o1 = model_data[model_data.metric .== "o1", :value]
	o10 = model_data[model_data.metric .== "o10", :value]
	o100 = model_data[model_data.metric .== "o100", :value]
	o1000 = model_data[model_data.metric .== "o1000", :value]
	o1 = mean(o1)
	o10 = mean(o10)
	o100 = mean(o100)
	o1000 = mean(o1000)

	plot!(
		[1, 2, 3, 4],
		[o1, o10, o100, o1000],
		label = model,
		xticks = (1:4, ["o1", "o10", "o100", "o1000"]),
		markershape = :circle,
		markersize = 4,
		opacity = 0.8,
		margin = 5mm,
		legend = legend,
		color= color,
		ylims = (0.0, 1.0),
		markerstrokewidth = 0
	)
end


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

plots = []
for variant in sorted_variants[2:end]  # Skip "base" for this plot
	classname = get(variant_to_class, variant, missing)
	# map to I, II, III
	classname = isnothing(classname) ? "" : " ($(toRomanNumeral(classname)))"
	plt = plot(title= "$(variant_names[variant])$(classname)")
	for (i, model) in enumerate(sorted_models)
		println("Plotting model: $model, variant: $variant")
		# Plot different lines for each model
		plot_line_overlap(model_short_names[model], variant_names[variant], false, colors[model])
	end
	push!(plots, plt)
end

# Add a legend plot
plt = scatter(
	1:length(sorted_models),
	[0.0 for _ in 1:length(sorted_models)],
	group = [model_short_names[m] for m in sorted_models],
	xlims=(0.5, 4.5),
	ylims=(0.5, 1.1),
	legend = :outerright,
	framestyle=:none,
	color=[colors[m] for m in sorted_models],
	opacity = 0.8,
	grid=false,
	showaxis=false,
	markershape = :circle,
	markersize = 4,
	markerstrokewidth = 0
)
push!(plots, plt)  # Add the legend plot at the end

plot(plots..., layout = (4, 4), size = (1400, 600), margin = 5mm)
savefig("plots/ranking_statistics_overlap.pdf")
