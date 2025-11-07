using DataStructures

"""
A container that holds the top N (key, score) pairs with the highest scores.
It uses a min-priority queue internally to efficiently manage the top N items.
The 'key' identifies the item, and 'score' determines its priority.
"""
struct TopNFixedLengthPriorityQueue{K, V <: Real}
	n::Int  # The fixed capacity of the queue
	# Internally uses a min-priority queue (ordered by V - the score).
	# It stores key-value pairs where key is K and value (priority) is V.
	# Base.Order.ForwardOrdering ensures it's a min-queue based on score.
	pq::PriorityQueue{K, V, Base.Order.ForwardOrdering}

	"""
	Constructs a new TopNFixedLengthPriorityQueue with a given capacity.
	# Arguments
	- `capacity::Int`: The maximum number of (key, score) pairs to store. Must be positive.
	"""
	function TopNFixedLengthPriorityQueue{K, V}(capacity::Int) where {K, V <: Real}
		if capacity <= 0
			error("Capacity (n) must be a positive integer.")
		end
		# Corrected PriorityQueue constructor call:
		# Pass an instance of the ordering (Base.Order.ForwardOrdering())
		# to the constructor that expects it.
		# This creates a PriorityQueue{K, V, Base.Order.ForwardOrdering}.
		order_instance = Base.Order.ForwardOrdering()
		# Explicitly using the full type for construction to match the field type:
		new{K, V}(capacity, PriorityQueue{K, V, Base.Order.ForwardOrdering}(order_instance))
	end
end

# Helper constructor that allows creating an instance without specifying K, V in the type parameters
# if they can be inferred by Julia or for convenience.
# Example: TopNFixedLengthPriorityQueue(String, Float64, 5)
function TopNFixedLengthPriorityQueue(key_type::Type{K}, value_type::Type{V}, capacity::Int) where {K, V <: Real}
	return TopNFixedLengthPriorityQueue{K, V}(capacity)
end


"""
Adds a (key, score) pair to the container.

If the container is already at full capacity (`n` items):
- If the new item's score is higher than the lowest score currently in the container,
  the item with the lowest score is removed, and the new item is added.
- Otherwise (new item's score is not high enough), the container remains unchanged.

If the key already exists in the container, its score is updated. If this update
or the addition of a new key causes the container to exceed capacity, the item
with the overall lowest score is removed.

# Arguments
- `container`: The TopNFixedLengthPriorityQueue instance.
- `key_item::K`: The key of the item.
- `score_item::V`: The score of the item.

# Returns
- The modified `container`.
"""
function Base.push!(container::TopNFixedLengthPriorityQueue{K, V}, key_item::K, score_item::V) where {K, V <: Real}
	container.pq[key_item] = score_item

	if length(container.pq) > container.n
		dequeue!(container.pq)
	end
	return container
end

"""
Retrieves the top N (key, score) pairs from the container.

The pairs are returned as a `Vector{Pair{K, V}}`, sorted by score in
descending order (i.e., the item with the highest score comes first).

# Arguments
- `container`: The TopNFixedLengthPriorityQueue instance.

# Returns
- A `Vector{Pair{K, V}}` containing the top N items, sorted by score (highest first).
"""
function get_top_n_elements(container::TopNFixedLengthPriorityQueue{K, V}) where {K, V <: Real}
	all_pairs = collect(pairs(container.pq))
	sort!(all_pairs, by = x -> x.second, rev = true)
	return all_pairs
end

"""
Retrieves the (key, score) pairs currently in the container without a specific sort order.
The order reflects the internal storage of the priority queue, which is generally not sorted
beyond ensuring the heap property.

# Arguments
- `container`: The TopNFixedLengthPriorityQueue instance.

# Returns
- A `Vector{Pair{K, V}}` containing the items currently in the queue.
"""
function get_elements_unsorted(container::TopNFixedLengthPriorityQueue{K, V}) where {K, V <: Real}
	return collect(pairs(container.pq))
end

"""
Returns the number of elements currently in the container.
"""
Base.length(container::TopNFixedLengthPriorityQueue) = length(container.pq)

"""
Checks if the container is empty.
"""
Base.isempty(container::TopNFixedLengthPriorityQueue) = isempty(container.pq)

all_dirs = [
	"laion5b_s13b_b90k",
	"ViT-B-32_laion2b_s34b_b79k",
	"ViT-H-14_laion2b_s32b_b79k",
	"ViT-L-14_laion2b_s32b_b82k",
	"ViT-L-14_laion400m_e32",
	"ViT-H-14_laion2b_s32b_b79k/ViT-H-14_rho50_k1_constrained_FARE2",
	"EVA02-CLIP-L-14-336"
]


using DelimitedFiles, CSV, DataFrames
using Distances
using ProgressBars


for base_dir in all_dirs
	
	if contains(base_dir, "/")
		base_dir, text_encoder = split(base_dir, "/")
	else
		text_encoder = base_dir
	end

	base_dir = "features/$base_dir"

	println("Processing directory: $base_dir with text embedding: $text_encoder")
	if !isdir(base_dir)
		error("Directory $base_dir does not exist.")
	end

	query_vectors = map(x -> parse.(Float32, split(x, ",")), readlines("features/texts/$(text_encoder)_text.csv"))
	knn = [TopNFixedLengthPriorityQueue{String, Float32}(1000) for i in 1:length(query_vectors)]

	files = readdir(base_dir)

	iter = ProgressBar(files)
	for f in iter

		h = readdlm("$base_dir/$f", '\t')
		ids = h[:, 1]
		v = hcat(map(l -> parse.(Float32, split(l, ",")), h[:, 2])...)

		Threads.@threads for i in 1:length(query_vectors)

			d = colwise(CosineDist(), query_vectors[i], v)

			for (k, d) in zip(ids, d)
				push!(knn[i], String(k), Float32(1 - d))
			end

		end

		# println(f)
		set_description(iter, "Processed $f")

	end


	dfs = DataFrame[]

	for i in 1:length(knn)
		results = get_top_n_elements(knn[i])
		push!(dfs, DataFrame(query = i, idx = collect(1:length(results)), element = map(x -> x[1], results), score = map(x -> x[2], results)))
	end

	result = vcat(dfs...)

	CSV.write("analysis/knn/$(text_encoder)_results.csv", result)

end
