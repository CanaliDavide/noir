use std::time::Instant;

use regex::Regex;

use noir::operator::source::FileSource;
use noir::BatchMode;
use noir::EnvironmentConfig;
use noir::StreamEnvironment;

fn main() {
    tracing_subscriber::fmt::init();

    let (config, args) = EnvironmentConfig::from_args();
    if args.len() != 1 {
        panic!("Pass the dataset path as an argument");
    }
    let path = &args[0];

    let mut env = StreamEnvironment::new(config);

    env.spawn_remote_workers();

    let source = FileSource::new(path);
    let tokenizer = Tokenizer::new();
    let result = env
        .stream(source)
        .batch_mode(BatchMode::fixed(1024))
        .flat_map(move |line| tokenizer.tokenize(line))
        .group_by_count(|word| word.clone())
        .collect_vec();
    let start = Instant::now();
    env.execute();
    let elapsed = start.elapsed();

    if let Some(_r) = result.get() {
        println!("OK");
        // _r.iter()
        //     .sorted_by_key(|t| t.1)
        //     .rev()
        //     .take(10)
        //     .for_each(|(k, v)| eprintln!("{:>10}:{:>10}", k, v));
    }
    eprintln!("Elapsed: {:?}", elapsed);
}

#[derive(Clone)]
struct Tokenizer {
    re: Regex,
}

impl Tokenizer {
    fn new() -> Self {
        Self {
            re: Regex::new(r"[A-Za-z]+").unwrap(),
        }
    }
    fn tokenize(&self, value: String) -> Vec<String> {
        self.re
            .find_iter(&value)
            .map(|t| t.as_str().to_lowercase())
            .collect()
    }
}
