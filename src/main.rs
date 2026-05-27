fn main() {
    if let Err(err) = spam_filter::cli_main() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
