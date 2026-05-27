use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::error::Error as StdError;
use std::fmt;
use std::fs;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const TRAIN_LABELS: [&str; 2] = ["ham", "spam"];
const MODEL_MAGIC: &[u8; 8] = b"SPAMNB01";
const MODEL_VERSION: u32 = 1;

pub type Result<T> = std::result::Result<T, SpamFilterError>;

#[derive(Debug)]
pub enum SpamFilterError {
    Io(io::Error),
    InvalidCorpus(String),
    InvalidModel(String),
    Parse(String),
}

impl fmt::Display for SpamFilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::InvalidCorpus(message) => write!(f, "invalid corpus: {message}"),
            Self::InvalidModel(message) => write!(f, "invalid model: {message}"),
            Self::Parse(message) => f.write_str(message),
        }
    }
}

impl StdError for SpamFilterError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidCorpus(_) | Self::InvalidModel(_) | Self::Parse(_) => None,
        }
    }
}

impl From<io::Error> for SpamFilterError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NaiveBayesModel {
    pub class_priors: HashMap<String, f64>,
    pub word_probs: HashMap<String, HashMap<String, f64>>,
    pub class_token_totals: HashMap<String, usize>,
    pub vocab_size: usize,
}

pub fn train(corpus_path: &Path) -> Result<NaiveBayesModel> {
    let corpus_path = resolve_corpus_path(corpus_path)?;
    let documents = collect_documents(&corpus_path)?;

    let mut class_doc_counts: HashMap<String, usize> = HashMap::new();
    let mut class_token_totals: HashMap<String, usize> = HashMap::new();
    let mut class_word_counts: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut vocabulary: HashSet<String> = HashSet::new();

    for (label, path) in documents {
        let raw_email = read_text_lossy(&path)?;
        let body = extract_body(&raw_email);
        let tokens = tokenize(body);

        *class_doc_counts.entry(label.clone()).or_insert(0) += 1;
        let token_total = class_token_totals.entry(label.clone()).or_insert(0);
        let word_counts = class_word_counts.entry(label).or_default();

        for token in tokens {
            *token_total += 1;
            vocabulary.insert(token.clone());
            *word_counts.entry(token).or_insert(0) += 1;
        }
    }

    if vocabulary.is_empty() {
        return Err(SpamFilterError::InvalidCorpus(
            "no tokens were found while training".to_string(),
        ));
    }

    let total_documents: usize = class_doc_counts.values().sum();
    if total_documents == 0 {
        return Err(SpamFilterError::InvalidCorpus(
            "no training documents were found".to_string(),
        ));
    }

    let vocab_size = vocabulary.len();
    let mut class_priors = HashMap::new();
    let mut word_probs = HashMap::new();

    let mut labels: Vec<String> = class_doc_counts.keys().cloned().collect();
    labels.sort();

    for label in labels {
        let doc_count = *class_doc_counts
            .get(&label)
            .ok_or_else(|| SpamFilterError::InvalidCorpus(format!("missing class '{label}'")))?;
        let token_total = *class_token_totals.get(&label).ok_or_else(|| {
            SpamFilterError::InvalidCorpus(format!("missing token count for class '{label}'"))
        })?;
        let denominator = (token_total + vocab_size) as f64;
        let counts = class_word_counts.remove(&label).unwrap_or_default();

        class_priors.insert(label.clone(), doc_count as f64 / total_documents as f64);

        let mut probabilities = HashMap::new();
        for (word, count) in counts {
            probabilities.insert(word, (count as f64 + 1.0) / denominator);
        }
        word_probs.insert(label, probabilities);
    }

    let model = NaiveBayesModel {
        class_priors,
        word_probs,
        class_token_totals,
        vocab_size,
    };

    validate_model(&model)?;
    Ok(model)
}

pub fn predict(model: &NaiveBayesModel, email_text: &str) -> String {
    let scores = class_log_scores(model, email_text);
    let mut best: Option<(String, f64)> = None;

    for (label, score) in scores {
        let replace = match &best {
            None => true,
            Some((best_label, best_score)) => {
                score > *best_score || (score == *best_score && label < *best_label)
            }
        };

        if replace {
            best = Some((label, score));
        }
    }

    best.expect("model has no classes; train a model first").0
}

pub fn predict_proba(model: &NaiveBayesModel, email_text: &str) -> HashMap<String, f64> {
    let scores = class_log_scores(model, email_text);
    let max_score = scores
        .iter()
        .map(|(_, score)| *score)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut exp_scores = Vec::with_capacity(scores.len());
    let mut total = 0.0;

    for (label, score) in scores {
        let value = (score - max_score).exp();
        total += value;
        exp_scores.push((label, value));
    }

    if !total.is_finite() || total == 0.0 {
        let uniform = 1.0 / exp_scores.len() as f64;
        return exp_scores
            .into_iter()
            .map(|(label, _)| (label, uniform))
            .collect();
    }

    exp_scores
        .into_iter()
        .map(|(label, value)| (label, value / total))
        .collect()
}

pub fn save_model(model: &NaiveBayesModel, path: &Path) -> Result<()> {
    validate_model(model)?;

    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(MODEL_MAGIC)?;
    write_u32(&mut writer, MODEL_VERSION)?;
    write_u64(&mut writer, model.vocab_size as u64)?;

    let mut labels: Vec<String> = model.class_priors.keys().cloned().collect();
    labels.sort();
    write_u32(&mut writer, labels.len() as u32)?;

    for label in labels {
        let prior = *model
            .class_priors
            .get(&label)
            .ok_or_else(|| SpamFilterError::InvalidModel(format!("missing class '{label}'")))?;
        let token_total = *model.class_token_totals.get(&label).ok_or_else(|| {
            SpamFilterError::InvalidModel(format!("missing token total for class '{label}'"))
        })?;
        let word_probs = model.word_probs.get(&label).ok_or_else(|| {
            SpamFilterError::InvalidModel(format!("missing word probabilities for class '{label}'"))
        })?;

        write_string(&mut writer, &label)?;
        write_f64(&mut writer, prior)?;
        write_u64(&mut writer, token_total as u64)?;

        let mut words: Vec<String> = word_probs.keys().cloned().collect();
        words.sort();
        write_u64(&mut writer, words.len() as u64)?;

        for word in words {
            let probability = *word_probs.get(&word).ok_or_else(|| {
                SpamFilterError::InvalidModel(format!(
                    "missing probability for word '{word}' in class '{label}'"
                ))
            })?;
            write_string(&mut writer, &word)?;
            write_f64(&mut writer, probability)?;
        }
    }

    writer.flush()?;
    Ok(())
}

pub fn load_model(path: &Path) -> Result<NaiveBayesModel> {
    let file = fs::File::open(path).map_err(|err| {
        if err.kind() == io::ErrorKind::NotFound {
            SpamFilterError::InvalidModel(format!("model file '{}' was not found", path.display()))
        } else {
            SpamFilterError::Io(err)
        }
    })?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MODEL_MAGIC {
        return Err(SpamFilterError::InvalidModel(format!(
            "file '{}' is not a spam detector model",
            path.display()
        )));
    }

    let version = read_u32(&mut reader)?;
    if version != MODEL_VERSION {
        return Err(SpamFilterError::InvalidModel(format!(
            "unsupported model version {version} in '{}'",
            path.display()
        )));
    }

    let vocab_size = read_u64(&mut reader)? as usize;
    let class_count = read_u32(&mut reader)? as usize;

    let mut class_priors = HashMap::with_capacity(class_count);
    let mut word_probs = HashMap::with_capacity(class_count);
    let mut class_token_totals = HashMap::with_capacity(class_count);

    for _ in 0..class_count {
        let label = read_string(&mut reader)?;
        let prior = read_f64(&mut reader)?;
        let token_total = read_u64(&mut reader)? as usize;
        let word_count = read_u64(&mut reader)? as usize;

        if class_priors.insert(label.clone(), prior).is_some() {
            return Err(SpamFilterError::InvalidModel(format!(
                "duplicate class '{label}' in model file '{}'",
                path.display()
            )));
        }

        if class_token_totals
            .insert(label.clone(), token_total)
            .is_some()
        {
            return Err(SpamFilterError::InvalidModel(format!(
                "duplicate token total for class '{label}' in model file '{}'",
                path.display()
            )));
        }

        let mut probabilities = HashMap::with_capacity(word_count);
        for _ in 0..word_count {
            let word = read_string(&mut reader)?;
            let probability = read_f64(&mut reader)?;
            if probabilities.insert(word.clone(), probability).is_some() {
                return Err(SpamFilterError::InvalidModel(format!(
                    "duplicate word '{word}' for class '{label}' in model file '{}'",
                    path.display()
                )));
            }
        }

        word_probs.insert(label, probabilities);
    }

    let model = NaiveBayesModel {
        class_priors,
        word_probs,
        class_token_totals,
        vocab_size,
    };

    validate_model(&model)?;
    Ok(model)
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_f64<W: Write>(writer: &mut W, value: f64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<()> {
    let bytes = value.as_bytes();
    let length = u32::try_from(bytes.len()).map_err(|_| {
        SpamFilterError::InvalidModel(format!("string too long to serialise: '{value}'"))
    })?;
    write_u32(writer, length)?;
    writer.write_all(bytes)?;
    Ok(())
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f64<R: Read>(reader: &mut R) -> Result<f64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}

fn read_string<R: Read>(reader: &mut R) -> Result<String> {
    let length = read_u32(reader)? as usize;
    let mut bytes = vec![0u8; length];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|_| {
        SpamFilterError::InvalidModel("model file contained invalid UTF-8".to_string())
    })
}

pub fn extract_body(text: &str) -> &str {
    let mut offset = 0usize;
    let mut saw_header = false;

    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(|c| c == '\r' || c == '\n');
        if trimmed.trim().is_empty() {
            return &text[offset + line.len()..];
        }

        if is_header_line(trimmed) {
            saw_header = true;
            offset += line.len();
            continue;
        }

        if saw_header {
            return &text[offset..];
        }

        return text;
    }

    if saw_header { "" } else { text }
}

pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lower in ch.to_lowercase() {
                current.push(lower);
            }
        } else {
            push_token(&mut tokens, &mut current);
        }
    }

    push_token(&mut tokens, &mut current);
    tokens
}

pub fn cli_main() -> Result<()> {
    match parse_command()? {
        Command::Help => {
            print_usage();
            Ok(())
        }
        Command::Train { data, output } => {
            let model = train(&data)?;
            save_model(&model, &output)?;
            Ok(())
        }
        Command::Predict { model, file } => {
            let model = load_model(&model)?;
            let email_text = read_text_lossy(&file)?;
            let label = predict(&model, &email_text);
            println!("{label}");
            Ok(())
        }
    }
}

fn class_log_scores(model: &NaiveBayesModel, email_text: &str) -> Vec<(String, f64)> {
    assert!(
        !model.class_priors.is_empty(),
        "model has no classes; train a model first"
    );

    let tokens = tokenize(extract_body(email_text));
    let mut labels: Vec<String> = model.class_priors.keys().cloned().collect();
    labels.sort();

    let mut scores = Vec::with_capacity(labels.len());
    for label in labels {
        let prior = *model
            .class_priors
            .get(&label)
            .expect("model missing class prior");
        let token_total = model.class_token_totals.get(&label).copied().unwrap_or(0);
        let denominator = (token_total + model.vocab_size).max(1) as f64;
        let default_probability = 1.0 / denominator;

        let mut score = prior.ln();
        if let Some(word_probs) = model.word_probs.get(&label) {
            for token in &tokens {
                let probability = word_probs
                    .get(token)
                    .copied()
                    .unwrap_or(default_probability);
                score += probability.ln();
            }
        } else {
            score += (default_probability.ln()) * tokens.len() as f64;
        }

        scores.push((label, score));
    }

    scores
}

fn validate_model(model: &NaiveBayesModel) -> Result<()> {
    if model.class_priors.is_empty() {
        return Err(SpamFilterError::InvalidModel(
            "model has no classes".to_string(),
        ));
    }

    if model.word_probs.is_empty() {
        return Err(SpamFilterError::InvalidModel(
            "model has no word probabilities".to_string(),
        ));
    }

    if model.vocab_size == 0 {
        return Err(SpamFilterError::InvalidModel(
            "model has an empty vocabulary".to_string(),
        ));
    }

    for (label, prior) in &model.class_priors {
        if !prior.is_finite() || *prior < 0.0 {
            return Err(SpamFilterError::InvalidModel(format!(
                "invalid class prior for '{label}'"
            )));
        }
        if !model.class_token_totals.contains_key(label) {
            return Err(SpamFilterError::InvalidModel(format!(
                "missing token total for class '{label}'"
            )));
        }
        if !model.word_probs.contains_key(label) {
            return Err(SpamFilterError::InvalidModel(format!(
                "missing word probabilities for class '{label}'"
            )));
        }
    }

    Ok(())
}

fn push_token(tokens: &mut Vec<String>, current: &mut String) {
    if current.chars().count() >= 2 {
        tokens.push(std::mem::take(current));
    } else {
        current.clear();
    }
}

fn is_header_line(line: &str) -> bool {
    let Some((name, value)) = line.split_once(':') else {
        return false;
    };

    let name = name.trim();
    !name.is_empty()
        && !value.trim().is_empty()
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-')
}

fn resolve_corpus_path(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    let candidate = Path::new("data").join(path);
    if candidate.exists() {
        return Ok(candidate);
    }

    Err(SpamFilterError::InvalidCorpus(format!(
        "corpus path '{}' was not found",
        path.display()
    )))
}

fn collect_documents(corpus_path: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut documents = Vec::new();

    for label in TRAIN_LABELS {
        let label_dir = corpus_path.join(label);
        if !label_dir.is_dir() {
            return Err(SpamFilterError::InvalidCorpus(format!(
                "missing required directory '{}'",
                label_dir.display()
            )));
        }

        let mut files = Vec::new();
        collect_files(&label_dir, &mut files)?;
        files.sort();

        if files.is_empty() {
            return Err(SpamFilterError::InvalidCorpus(format!(
                "no email files were found under '{}'",
                label_dir.display()
            )));
        }

        for file in files {
            documents.push((label.to_string(), file));
        }
    }

    Ok(documents)
}

fn collect_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if root.is_file() {
        files.push(root.to_path_buf());
        return Ok(());
    }

    if !root.is_dir() {
        return Err(SpamFilterError::InvalidCorpus(format!(
            "expected '{}' to be a directory",
            root.display()
        )));
    }

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }

    Ok(())
}

fn read_text_lossy(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn parse_command() -> Result<Command> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Ok(Command::Help);
    }

    let subcommand = &args[0];
    let rest = &args[1..];

    match subcommand.as_str() {
        "train" => parse_train_command(rest),
        "predict" => parse_predict_command(rest),
        other => Err(SpamFilterError::Parse(format!(
            "unknown subcommand '{other}'"
        ))),
    }
}

fn parse_train_command(args: &[String]) -> Result<Command> {
    let mut data = PathBuf::from("data/enron1");
    let mut output = PathBuf::from("model.bin");
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--data" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| SpamFilterError::Parse("--data requires a value".to_string()))?;
                data = PathBuf::from(value);
            }
            "--output" => {
                index += 1;
                let value = args.get(index).ok_or_else(|| {
                    SpamFilterError::Parse("--output requires a value".to_string())
                })?;
                output = PathBuf::from(value);
            }
            other => {
                return Err(SpamFilterError::Parse(format!(
                    "unexpected argument '{other}'"
                )));
            }
        }

        index += 1;
    }

    Ok(Command::Train { data, output })
}

fn parse_predict_command(args: &[String]) -> Result<Command> {
    let mut model = None;
    let mut file = None;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--model" => {
                index += 1;
                let value = args.get(index).ok_or_else(|| {
                    SpamFilterError::Parse("--model requires a value".to_string())
                })?;
                model = Some(PathBuf::from(value));
            }
            "--file" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| SpamFilterError::Parse("--file requires a value".to_string()))?;
                file = Some(PathBuf::from(value));
            }
            other => {
                return Err(SpamFilterError::Parse(format!(
                    "unexpected argument '{other}'"
                )));
            }
        }

        index += 1;
    }

    let model = model.ok_or_else(|| SpamFilterError::Parse("--model is required".to_string()))?;
    let file = file.ok_or_else(|| SpamFilterError::Parse("--file is required".to_string()))?;

    Ok(Command::Predict { model, file })
}

fn print_usage() {
    eprintln!(
        "Usage:\n  spam_detector train [--data PATH] [--output FILE]\n  spam_detector predict --model FILE --file FILE\n\nDefaults:\n  --data    data/enron1\n  --output  model.bin"
    );
}

enum Command {
    Help,
    Train { data: PathBuf, output: PathBuf },
    Predict { model: PathBuf, file: PathBuf },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn tokenize_lowercases_and_splits() {
        let tokens = tokenize("Hello, WORLD! a 1 B2; e-mail.");
        assert_eq!(tokens, vec!["hello", "world", "b2", "mail"]);
    }

    #[test]
    fn extract_body_skips_headers() {
        let email =
            "From: sender@example.com\r\nSubject: hello\r\n\r\nBody line one\r\nBody line two";
        assert_eq!(extract_body(email), "Body line one\r\nBody line two");
    }

    #[test]
    fn extract_body_handles_headers_without_blank_line() {
        let email = "Subject: hello there\nThis is the body\nWith another line";
        assert_eq!(extract_body(email), "This is the body\nWith another line");
    }

    #[test]
    fn laplace_smoothing_keeps_unseen_words_positive() {
        let temp = TestDir::new("laplace");
        write_email(
            temp.path().join("ham").join("ham1.txt"),
            "Subject: team lunch\n\nplease review the schedule",
        );
        write_email(
            temp.path().join("spam").join("spam1.txt"),
            "Subject: cheap pills\n\nbuy cheap pills now",
        );

        let model = train(temp.path()).expect("training should succeed");
        let proba = predict_proba(&model, "Subject: unknown\n\nxylophone");

        assert_eq!(proba.len(), 2);
        for value in proba.values() {
            assert!(*value > 0.0);
            assert!(value.is_finite());
        }
        let sum: f64 = proba.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn log_probability_is_correct() {
        let mut class_priors = HashMap::new();
        class_priors.insert("ham".to_string(), 0.25);
        class_priors.insert("spam".to_string(), 0.75);

        let mut ham_words = HashMap::new();
        ham_words.insert("offer".to_string(), 0.5);
        let mut spam_words = HashMap::new();
        spam_words.insert("offer".to_string(), 0.6);

        let mut word_probs = HashMap::new();
        word_probs.insert("ham".to_string(), ham_words);
        word_probs.insert("spam".to_string(), spam_words);

        let mut class_token_totals = HashMap::new();
        class_token_totals.insert("ham".to_string(), 2);
        class_token_totals.insert("spam".to_string(), 3);

        let model = NaiveBayesModel {
            class_priors,
            word_probs,
            class_token_totals,
            vocab_size: 4,
        };

        let scores = class_log_scores(&model, "Subject: offer\n\noffer unknown");
        let ham_score = scores
            .iter()
            .find(|(label, _)| label == "ham")
            .map(|(_, score)| *score)
            .expect("ham score");
        let spam_score = scores
            .iter()
            .find(|(label, _)| label == "spam")
            .map(|(_, score)| *score)
            .expect("spam score");

        let expected_ham = 0.25f64.ln() + 0.5f64.ln() + (1.0 / 6.0f64).ln();
        let expected_spam = 0.75f64.ln() + 0.6f64.ln() + (1.0 / 7.0f64).ln();

        assert!((ham_score - expected_ham).abs() < 1e-12);
        assert!((spam_score - expected_spam).abs() < 1e-12);
    }

    #[test]
    fn end_to_end_training_prediction_and_serialization() {
        let temp = TestDir::new("end_to_end");
        write_email(
            temp.path().join("ham").join("nested").join("ham1.txt"),
            "Subject: project update\n\nplease review the meeting notes and report",
        );
        write_email(
            temp.path().join("ham").join("ham2.txt"),
            "Subject: lunch schedule\n\nlet us meet for lunch tomorrow",
        );
        write_email(
            temp.path().join("spam").join("spam1.txt"),
            "Subject: free money\n\nclaim your free prize now",
        );
        write_email(
            temp.path().join("spam").join("nested").join("spam2.txt"),
            "Subject: cheap pills\n\nbuy cheap pills and free offers",
        );

        let model = train(temp.path()).expect("training should succeed");
        assert_eq!(
            predict(&model, "Subject: prize\n\nfree prize claim now"),
            "spam"
        );
        assert_eq!(
            predict(
                &model,
                "Subject: report\n\nplease review the report and notes"
            ),
            "ham"
        );

        let model_path = temp.path().join("model.bin");
        save_model(&model, &model_path).expect("model save should succeed");
        let loaded = load_model(&model_path).expect("model load should succeed");
        assert_eq!(
            predict(&loaded, "Subject: prize\n\nfree prize claim now"),
            "spam"
        );
        assert_eq!(
            predict(
                &loaded,
                "Subject: report\n\nplease review the report and notes"
            ),
            "ham"
        );
    }

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(prefix: &str) -> Self {
            let mut path = std::env::temp_dir();
            let suffix = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should move forward")
                .as_nanos();
            path.push(format!(
                "spam_filter_{prefix}_{}_{}",
                std::process::id(),
                suffix
            ));
            fs::create_dir_all(path.join("ham")).expect("ham dir");
            fs::create_dir_all(path.join("spam")).expect("spam dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn write_email(path: PathBuf, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("parent dir");
        }
        fs::write(path, content).expect("write email");
    }
}
