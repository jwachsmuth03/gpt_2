| Name | Student number | Tasks | Share |
| --- | --- | --- | --- |
| Enter actual member | Enter number | Confirm actual contribution | Enter percentage |

Complete the agreed contribution table for all members before submission, with shares totaling 100%, and add one sentence about each member's work. No personal details or contributions have been invented.

# Completed Assignments 1, 2, and 3

The assembled notebook `Final_Assignment_Assembly_BJS.ipynb` now contains all five Assignment 2 tasks, with executed outputs. Assignment 1 and Assignment 3 cell sources are unchanged from the previous completed notebook; Assignment 4 remains empty. `assignment2.ipynb` exports only the second section. Keep the notebooks and supporting Python modules together.

## Run

Use Python 3.12, install requirements.txt, select that environment's Python kernel in Jupyter, then Restart Kernel and Run All. The standalone Assignment 2 code uses only the standard library and NumPy for the model; pandas and matplotlib provide tables and plots. The combined requirements also include PyTorch for Assignment 3. No import-path changes are required. Missing corpora download into data/ and are checksum-verified. The archive excludes corpora, generated text split files, and environment files.

Assignment 2 ran on CPU in 10.07 seconds on macOS-15.7.1-arm64-arm-64bit, Python 3.12.14, with seed 42. The full assembled notebook completed with no code-cell errors. Timings vary by machine.

## Assignment 2

Task 1 saves deterministic clean 80/10/10 splits by line (32,000/4,000/4,000), trains BPE on training text only, encodes each line with n-1 BOS and one EOS, and reports line/character counts and unknown-token counts. Assignment 1's 99%-trained merges are not reused because that would leak held-out data.

Task 2 implements orders 1, 2, and 3 with add-one smoothing, tuple keys, and sequence boundaries. It checks five toy sequences, two hand-calculated probabilities, normalization, unseen histories, and unigram empty contexts.

Task 3 uses the same model-independent perplexity function as Assignment 3, checks the token and character denominators, and evaluates a unigram on its training text and a saved file of 200 seeded random content IDs (plus EOS for evaluation).

Task 4 evaluates all 15 validation combinations of n=1/2/3 and k=250/500/1000/2000/4000, displays the complete grid and a logarithmic plot, freezes selection before test evaluation, and reports test perplexity for all three orders at the selected k. Validation selected n=2, k=250.

Task 5 generates ten argmax and ten sampled outputs using the same three prompts as Assignment 3. A wrapper records contexts without changing generate. Repeated contexts establish deterministic argmax cycles; a summary reports loops and EOS stops before the 50-token limit. Seeded generation and refitted count tables are checked for reproducibility.

Character denominators include clean line contents, spaces, and punctuation, but exclude line separators and synthetic markers. EOS counts as a prediction; BOS does not. This convention is identical for Assignments 2 and 3.

## Reuse and integration

The original n_gram.py, byte_pair_encoder.py, and neural_n_gram.py are preserved byte for byte. The ngram.py adapter inherits the original NGram initialization, training setup, and unigram counter. Higher-order tuple counts and the required probability interface avoid the original string-concatenation ambiguity and integer addition. The shared perplexity and generate function bodies are unchanged from the previous completed deliverable.

language_data.py holds the same existing corpus-loading/splitting logic, shared with Assignment 3. Assignment 2 now establishes the evaluation functions, prompts, and split previously called provisional in Assignment 3's preserved historical introduction. Assignments 1 and 3 were executed again successfully; their notebook cell sources remain unchanged.

## Files and sources

Learned BPE files are in models/assignment2/. The full validation grid, test results, generated samples, diagnostic counts, random-ID sanity file, frozen selection, and runtime are in results/assignment2/. Original Assignment 1 and 3 outputs and models are retained.

Shakespeare: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Assignment 1 comparison corpora: https://www.gutenberg.org/ebooks/1342.txt.utf-8 and https://raw.githubusercontent.com/python/cpython/v3.13.0/Lib/test/test_typing.py . Their checksums and byte counts are in results/assignment1/data_sources.json; neither is used to train tokenizers.

Assignment 3 WSJ test corpus: https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt . The notebook verifies its required MD5 and 3,761 lines.

## AI assistance

An AI assistant inspected the assignment and original group code, reused sufficient original methods without modification, added the required interfaces and missing experiment workflow, filled the notebook's five Assignment 2 tasks, and executed the combined notebook to verify the outputs. The group must review and understand the implementation and complete its own factual contribution statement. Earlier AI-assisted Assignment 1 and 3 implementations remain included.
