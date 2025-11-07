using DataFrames, Statistics
using StringDistances  # pkg> add StringDistances
using Distances
using CSV

# -------- helpers --------
const STOPWORDS = Set([
    "a","an","the","and","or","but","if","then","else","when","while","for","to","of","in","on","at","by","with",
    "from","as","is","are","was","were","be","been","being","this","that","these","those","it","its","into","over",
    "under","about","up","down","out","off"
])

# very simple tokenizer: keep alphanumerics; lowercase; drop 1-char tokens
tokenize(s::AbstractString) = [lowercase(m.match) for m in eachmatch(r"[A-Za-z0-9]+", s) if length(m.match) > 1]
content_tokens(s) = [t for t in tokenize(s) if !(t in STOPWORDS)]

jaccard(a::Vector{<:AbstractString}, b::Vector{<:AbstractString}) = begin
    A, B = Set(a), Set(b)
    if isempty(A) && isempty(B); return 1.0 end
    inter = length(intersect(A,B)); uni = length(union(A,B))
    return inter/uni
end

punct_density(s::AbstractString) = begin
    total = ncodeunits(s)
    total == 0 && return 0.0
    p = count(c -> !isletter(c) && !isnumeric(c) && !isspace(c), s)
    p/total
end

norm_levenshtein(base::AbstractString, pert::AbstractString) = begin
    L = max(length(base), 1)
    evaluate(Levenshtein(), base, pert) / L
end

# main feature function for one pair (base text, perturbed text)
function edit_features(base_text::AbstractString, pert_text::AbstractString)
    nb = content_tokens(base_text)
    np = content_tokens(pert_text)
    tok_jacc = jaccard(nb, np)                    # similarity (1 = same, 0 = disjoint)
    (; 
        norm_lev = norm_levenshtein(base_text, pert_text),
        tok_jacc = tok_jacc,
        tok_dist = 1 - tok_jacc,
        punct_delta = abs(punct_density(pert_text) - punct_density(base_text)),
        len_ratio = length(pert_text) / max(length(base_text), 1)
    )
end


# df_text: one row per (id, variant) with the query text
# expected columns: :id, :variant, :text
# stats: your 440×20 table with one row per (id, query_variant, model), including :variant, :model, :id, and stability metrics like :rbo99
df_text = CSV.read("queries/permutations.csv", DataFrame)

# 2.1 Extract the base text per id
base_texts = filter(:variant => ==("base"), df_text)
rename!(base_texts, :query => :base_text)
select!(base_texts, [:id, :base_text])

# 2.2 Join base text onto all variants, then compute pairwise features
# sort df_text
feat_df = leftjoin(df_text, base_texts, on=:id)
feat_df.row = collect(1:nrow(feat_df))

# compute features (skip variant == "base" if you like; it will give zeros)
function add_features!(df::DataFrame)
    df.norm_lev = similar(df.id, Float64)
    df.tok_jacc = similar(df.id, Float64)
    df.tok_dist = similar(df.id, Float64)
    df.punct_delta = similar(df.id, Float64)
    df.len_ratio = similar(df.id, Float64)

    for i in 1:nrow(df)
        try
            f = edit_features(df.base_text[i], df.query[i])
            df.norm_lev[i] = f.norm_lev
            df.tok_jacc[i] = f.tok_jacc
            df.tok_dist[i] = f.tok_dist
            df.punct_delta[i] = f.punct_delta
            df.len_ratio[i] = f.len_ratio
        catch e
            display(df[i, :])
            println("Error processing row $i: ", e)
            # Reraise
            throw(e)
            df.norm_lev[i] = NaN
            df.tok_jacc[i] = NaN
            df.tok_dist[i] = NaN
            df.punct_delta[i] = NaN
            df.len_ratio[i] = NaN
        end
    end
    return df
end
add_features!(feat_df)

# Text distance from the CLIP models
models = [
	"laion5b_s13b_b90k",
	"ViT-B-32_laion2b_s34b_b79k",
	"ViT-H-14_laion2b_s32b_b79k",
	"ViT-L-14_laion2b_s32b_b82k",
	"ViT-L-14_laion400m_e32",
    "ViT-H-14_rho50_k1_constrained_FARE2",
 	"EVA02-CLIP-L-14-336",
]

df_text[:, :row] = collect(1:nrow(df_text))
stats = DataFrame(
    id = Int[],
    variant = String[],
    model = String[],
    row = Int[],
    distance = Float64[]
)

variants = unique(df_text.variant)

for model in models
    println("Processing model: $model")

    query_vectors = map(x -> parse.(Float32, split(x, ",")), readlines("features/texts/$(model)_text.csv"))
    v = hcat(query_vectors...)

    # create a dict for each variant
    variant_dists = Dict{String, Vector{Float64}}()
    for var in variants
        variant_dists[var] = Float64[]
    end

    dists = [ 0.0 for _ in 1:length(query_vectors) ]

    for g in groupby(df_text, :id)
        base = v[:, g[1, :row]]  # Base vector for the variant

        # append!(category_dists["base"], evaluate(CosineDist(), base, base))
        for (j, var) in enumerate(variants)
            if var == "base"
                continue  # Skip the base variant
            end
            filtered = g[g.variant .== var, :]
            if isempty(filtered)
                continue
            end
        
            # Calculate distance for the current variant
            d = colwise(CosineDist(), base, v[:, filtered.row])
            append!(variant_dists[var], d)
            for i in 1:length(filtered.row)
                dists[filtered.row[i]] = d[i]
            end
        end
    end

    # Add distances to the stats DataFrame
    new_df = DataFrame(
        id = df_text.id,
        variant = df_text.variant,
        model = model,
        row = range(1, length(dists)),
        distance = dists
    )
    stats = vcat(stats, new_df)
end

# Save the text features DataFrame
full_stats = leftjoin(stats, feat_df, on = [:id, :variant, :row])
CSV.write("analysis/stats/text_features.csv", full_stats)
