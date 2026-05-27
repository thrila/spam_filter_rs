# Spam Detector

Naive Bayes spam classifier in Rust for the Enron email dataset.

## What It Does

- Trains a multinomial Naive Bayes model from raw email text.
- Uses lowercase tokenisation and Laplace smoothing.
- Saves and loads a model with a small custom binary format.
- Exposes a CLI for training and prediction.

## Dataset Layout

Place the extracted Enron corpora under `data/`:

- `data/enron1/ham`
- `data/enron1/spam`
- `data/enron2/ham`
- `data/enron2/spam`
- `data/enron3/ham`
- `data/enron3/spam`

The trainer also supports a single corpus directory like `data/enron1/` if you want to train on one set only.

## Train

Train on the default `data/` root:

```powershell
cargo run --bin spam_detector -- train --output model.bin
```

If you want to train from a specific corpus root:

```powershell
cargo run --bin spam_detector -- train --data data/enron1 --output model.bin
```

## Predict

Classify a message file:

```powershell
cargo run --bin spam_detector -- predict --model model.bin --file path\to\email.txt
```

The program prints either `spam` or `ham`.

## Model File Format

`model.bin` is a custom binary snapshot of the trained model. The layout and
serialization details are documented in [docs/model-format.md](docs/model-format.md).

## Tests

Run the test suite:

```powershell
cargo test
```

## Project Files

- `src/lib.rs`: tokenizer, training, prediction, serialization, and tests.
- `src/main.rs`: CLI entrypoint.
- `Cargo.toml`: crate metadata and binary target.

---
[Email dataset](https://www2.aueb.gr/users/ion/data/enron-spam/)
