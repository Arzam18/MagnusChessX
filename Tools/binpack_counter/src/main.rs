use std::{env, fs::File, path::Path, process::ExitCode};

use sfbinpack::CompressedTrainingDataEntryReader;

fn count_positions(path: &Path) -> Result<u64, String> {
    let file = File::open(path)
        .map_err(|error| format!("could not open {}: {error}", path.display()))?;
    let mut reader = CompressedTrainingDataEntryReader::new(file)
        .map_err(|error| format!("could not decode {}: {error}", path.display()))?;

    let mut count = 0_u64;
    while reader.has_next() {
        let entry = reader.next();
        if !(-1..=1).contains(&entry.result) {
            return Err(format!(
                "invalid WDL result {} at position {} in {}",
                entry.result,
                count + 1,
                path.display()
            ));
        }
        count += 1;
    }

    Ok(count)
}
fn main() -> ExitCode {
    let mut args = env::args_os();
    let program = args.next().unwrap_or_default();
    let Some(path) = args.next() else {
        eprintln!("usage: {} FILE.binpack", Path::new(&program).display());
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("error: exactly one .binpack file is required");
        return ExitCode::from(2);
    }

    match count_positions(Path::new(&path)) {
        Ok(count) if count > 0 => {
            println!("{count}");
            ExitCode::SUCCESS
        }
        Ok(_) => {
            eprintln!("error: non-empty binpack decoded to zero positions");
            ExitCode::from(1)
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(1)
        }
    }
}
