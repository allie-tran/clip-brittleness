import spacy
import os
nlp = spacy.load("en_core_web_sm")

def extract_content_words(text):
    doc = nlp(text)
    content_words = {
        'nouns': [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']],
        'n+adj': [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ']],
        'all_content': [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB', 'NUM'] and not token.is_stop]
    }
    return content_words


with open("queries/queries.txt", "r") as file:
    queries = file.readlines()

output_file = "queries/permutations.csv"
if not os.path.exists(output_file):
    # check if output file exists, if not create and write header
    with open(output_file, "w") as io:
        io.write("id,variant,query\n")

for (i, query) in enumerate(queries):
    variations = extract_content_words(query.strip())
    idx = i + 1  # Start indexing from 1
    # nouns_only
    nouns_only = " ".join(variations['nouns'])
    # Append each row as we get it
    with open(output_file, "a") as io:
        # io.write(f"{idx},nouns,\"{nouns_only}\"\n")
        shuffle_nouns = " ".join(variations['nouns'])
        io.write(f"{idx},shuffle_nouns,\"{shuffle_nouns}\"\n")

    # nouns_and_adjectives
    nouns_and_adjectives = " ".join(variations['n+adj'])
    with open(output_file, "a") as io:
        # io.write(f"{idx},n+adj,\"{nouns_and_adjectives}\"\n")
        shuffle_nouns_and_adjectives = " ".join(variations['n+adj'])
        io.write(f"{idx},shuffle_n+adj,\"{shuffle_nouns_and_adjectives}\"\n")
    
    # all_content
    all_content = " ".join(variations['all_content'])
    with open(output_file, "a") as io:
        # io.write(f"{idx},keywords,\"{all_content}\"\n")
        shuffle_all_content = " ".join(variations['all_content'])
        io.write(f"{idx},shuffle_keywords,\"{shuffle_all_content}\"\n")

    # Adding punctuation variations
    with open(output_file, "a") as io:
        io.write(f"{idx},punctuation,\"{query.strip()}.\"\n")

print("Saved results to", output_file)
