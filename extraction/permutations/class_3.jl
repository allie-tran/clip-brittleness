using DotEnv
using HTTP
using JSON3
using CSV

# Load environment variables
DotEnv.load!()

const GEMINI_API_KEY = ENV["GEMINI_API_KEY"]
model = "gemini-2.5-flash-lite"
const GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/$model:generateContent"

function rephrase_query(query::String, n::Int=10)
    rephrases = String[]
    sleep(0.25)  # Optional: sleep to avoid hitting rate limits too quickly
    for i in 1:n
        # prompt = """
        # Rephrase the following sentence without changing its meaning. 
        # Reply with a single sentence.
        # Query: "$query"
        # """
        prompt = """
        Change one word or phrase in the following sentence to their synonyms. Keep the rest of the sentence structure the same.
        Reply with a single sentence, unformatted.
        Query: "$query"
        """
        if length(rephrases) > 0
            prompt *= "Avoid repeating previous rephrases: $(rephrases)"
        end

        payload = JSON3.write(Dict("contents" => [Dict("parts" => [Dict("text" => prompt)])]))
        headers = ["Content-Type" => "application/json"]
        url = "$GEMINI_ENDPOINT?key=$GEMINI_API_KEY"
        
        while true
            try
                response = HTTP.post(url, headers, payload)
                if response.status == 200
                    data = JSON3.read(String(response.body))
                    text = try
                        data["candidates"][1]["content"]["parts"][1]["text"]
                    catch
                        ""
                    end
                    text = strip(text)
                    text = strip(text, '.')  # Remove trailing period
                    text = replace(text, "\n" => " ")  # Remove newlines
                    push!(rephrases, text)
                else
                    println("Error: $(response.status) - $(String(response.body))")
                    error("Failed to get a valid response from the API.")
                end
                break  # Exit loop if request is successful
            catch e
                println("Exception occurred: $e")
                println("Request failed. Retrying...")
                sleep(1)  # Wait before retrying
            end
        end
    end
    return rephrases
end

queries = readlines("queries.txt")
output_file = "queries/permutations.csv"

# Write header if file does not exist
if !isfile(output_file)
    open(output_file, "w") do io
        println(io, "id,variant,query")
    end
end

for (idx, query) in enumerate(queries)
    println("Processing query $idx: $query")
    variations = rephrase_query(query, 5)
    for (i, text) in enumerate(variations)
        # Append each row as we get it
        open(output_file, "a") do io
            println(io, "$idx,rephrase,\"$text\"")
        end
    end
end

println("Saved results to $output_file")