using Random
using DataFrames
using CSV

function lower(query::String)::String
    return lowercase(query)
end

function all_upper(query::String)::String
    return uppercase(query)
end

qwerty = Dict(
    'q' => ['w', 'a'],
    'w' => ['q', 'a', 's', 'e'],
    'e' => ['w', 's', 'd', 'r'],
    'r' => ['e', 'd', 'f', 't'],
    't' => ['r', 'f', 'g', 'y'],
    'y' => ['t', 'g', 'h', 'u'],
    'u' => ['y', 'h', 'j', 'i'],
    'i' => ['u', 'j', 'k', 'o'],
    'o' => ['i', 'k', 'l', 'p'],
    'p' => ['o', 'l'],
    'a' => ['q', 'w', 's', 'x', 'z'],
    's' => ['w', 'e', 'a', 'd', 'z', 'x'],
    'd' => ['e', 'r', 's', 'f', 'x', 'c'],
    'f' => ['r', 't', 'd', 'g', 'c', 'v'],
    'g' => ['t', 'y', 'f', 'h', 'v', 'b'],
    'h' => ['y', 'u', 'g', 'j', 'b', 'n'],
    'j' => ['u', 'i', 'h', 'k', 'n', 'm'],
    'k' => ['i', 'o', 'j', 'l', 'm'],
    'l' => ['o', 'p', 'k'],
    'z' => ['a', 's', 'x'],
    'x' => ['z', 's', 'd', 'c'],
    'c' => ['d', 'f', 'x', 'v'],
    'v' => ['f', 'g', 'c', 'b'],
    'b' => ['g', 'h', 'v', 'n'],
    'n' => ['h', 'j', 'b', 'm'],
    'm' => ['j', 'k', 'n']
)

function keyboard(query::String)::String
    indices = findall(x -> x in keys(qwerty), lowercase(query))
    if isempty(indices)
        throw("no ascii characters in string")
    end
    index = rand(indices)
    lower = query[index] in keys(qwerty)
    replace = rand(qwerty[lowercase(query)[index]])

    return query[1:(index - 1)] * (lower ? replace : uppercase(replace)) * query[(index + 1):end]
end

function remove_one(query::String)::String
    index = rand(1:length(query))
    return query[1:(index-1)] * query[(index+1):end]
end

function repeat_one(query::String)::String
    index = rand(1:length(query))
    return query[1:index] * query[index:end]
end

function swap_two(query::String)::String
    indices = filter(x -> query[x] != query[x+1], collect(1:(length(query)-1)))
    if isempty(indices)
        throw("no different characters in string")
    end
    index = rand(indices)
    s = query[1:(index-1)] * query[index+1] * query[index]
    if index < length(query) - 1
        s = s * query[(index+2):end]
    end
    return  s
end

dfs = DataFrame[]
i = 0
for line in eachline("queries.txt")
    i += 1
    push!(dfs, DataFrame(id = i, variant = "base", query = line))
    push!(dfs, DataFrame(id = i, variant = "lower", query = lower(line)))
    push!(dfs, DataFrame(id = i, variant = "upper", query = all_upper(line)))

    queries = Set{String}()
    while length(queries) < 5
        push!(queries, keyboard(line))
    end
    for q in queries
        push!(dfs, DataFrame(id = i, variant = "keyboard", query = q))
    end

    queries = Set{String}()
    while length(queries) < 5
        push!(queries, remove_one(line))
    end
    for q in queries
        push!(dfs, DataFrame(id = i, variant = "remove", query = q))
    end

    queries = Set{String}()
    while length(queries) < 5
        push!(queries, repeat_one(line))
    end
    for q in queries
        push!(dfs, DataFrame(id = i, variant = "repeat", query = q))
    end

    queries = Set{String}()
    while length(queries) < 5
        push!(queries, swap_two(line))
    end
    for q in queries
        push!(dfs, DataFrame(id = i, variant = "swap", query = q))
    end

    println(i)

end

    
df = vcat(dfs...)
CSV.write("queries/permutations.csv", df)