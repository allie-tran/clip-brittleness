# On the Brittleness of CLIP Text Encoders

This is the codes for the analysis described in the paper "On the Brittleness of CLIP Text Encoders", accepted at MMM'26.
The preprint of the paper can be access [here](https://arxiv.org/abs/2511.04247).

## .env file
You need to set these in the `.env` file

```
FRAME_DIR=`path/to/frames`
```

## Structure

**All scripts** need to be run from the root directory to make sure the paths are correct.

- `queries`: store original text queries and permutations
- `extraction`: various codes to:
    - `permutations`: create permutations for each queries in `queries/queries.txt`. Order of operation: `class_1.jl`, `class_2.jl`, `class_3.jl`.
    - `texts`: create text embeddings for each models
    - `videos`: create frame embeddings for each models
- `analysis`: various analysis, including ones that are not in the paper. The main one is in `correlation.jl` and `overlap_analysis.jl`. However, these need to be run beforehand:
    - `text_analysis.jl`
    - `knn_comparison.jl`
    - `ranking_analysis.jl`