use std::{
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use bullet_lib::value::loader::DataLoader;
use bulletformat::{BulletFormat, ChessBoard};
use serde::{Deserialize, Serialize};
use sfbinpack::CompressedTrainingDataEntryReader;

use super::{SplitMix64, convert_to_bulletformat, fisher_yates_shuffle};

pub const DATASET_CACHE_SHARDS: usize = 64;
pub const DATASET_CACHE_RECORD_SIZE: usize = std::mem::size_of::<ChessBoard>();

const CACHE_VERSION: u32 = 1;
const PARTITION_ALGORITHM: &str = "splitmix64-global-position-v1";
const PARTITION_SEED: u64 = 0x4D4E_5545_5032_0001;
const EPOCH_ORDER_DOMAIN: u64 = 0xD157_5EED_E0C4_0001;
const SHARD_ORDER_DOMAIN: u64 = 0xD157_5EED_5A4D_0001;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DatasetSourceManifest {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub positions: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DatasetShardManifest {
    pub path: String,
    pub count: u64,
    pub size_bytes: u64,
    pub source_counts: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DatasetCacheManifest {
    pub version: u32,
    pub partition_algorithm: String,
    pub source_fingerprint: String,
    pub total_positions: u64,
    pub total_bytes: u64,
    pub record_size: usize,
    pub shard_count: usize,
    pub sources: Vec<DatasetSourceManifest>,
    pub shards: Vec<DatasetShardManifest>,
}

#[derive(Clone, Debug)]
pub struct DatasetCache {
    directory: PathBuf,
    manifest: DatasetCacheManifest,
}

impl DatasetCache {
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn manifest(&self) -> &DatasetCacheManifest {
        &self.manifest
    }

    pub fn source_fingerprint(&self) -> &str {
        &self.manifest.source_fingerprint
    }

    pub fn cache_fingerprint(&self) -> String {
        format!(
            "v{}:{}:{}",
            self.manifest.version,
            self.manifest.partition_algorithm,
            self.manifest.source_fingerprint
        )
    }

    pub fn total_positions(&self) -> u64 {
        self.manifest.total_positions
    }
}

#[derive(Clone, Debug)]
struct InspectedSource {
    path: PathBuf,
    display_path: String,
    size_bytes: u64,
    sha256: String,
}

/// Hashes and fingerprints the source files without decoding them. Paths are
/// canonicalised and sorted so caller ordering cannot alter the cache identity.
pub fn fingerprint_sources(source_paths: &[PathBuf]) -> io::Result<String> {
    Ok(inspect_sources(source_paths)?.1)
}

/// Returns an existing valid cache or builds and atomically activates a new one.
/// A matching but corrupt cache is rejected instead of silently regenerated.
pub fn ensure_dataset_cache(
    source_paths: &[PathBuf],
    cache_dir: &Path,
) -> io::Result<DatasetCache> {
    let (sources, source_fingerprint) = inspect_sources(source_paths)?;
    if cache_dir.exists() && !cache_dir.is_dir() {
        return invalid_input(format!(
            "dataset cache path is not a directory: {}",
            cache_dir.display()
        ));
    }

    if let Some(manifest) = reusable_manifest(cache_dir, &sources, &source_fingerprint)? {
        return Ok(DatasetCache {
            directory: cache_dir.to_path_buf(),
            manifest,
        });
    }

    build_dataset_cache(&sources, &source_fingerprint, cache_dir)
}

fn reusable_manifest(
    cache_dir: &Path,
    sources: &[InspectedSource],
    source_fingerprint: &str,
) -> io::Result<Option<DatasetCacheManifest>> {
    let manifest_path = cache_dir.join("manifest.toml");
    if !manifest_path.exists() {
        return Ok(None);
    }
    let manifest = read_manifest(&manifest_path)?;
    if manifest.version != CACHE_VERSION
        || manifest.partition_algorithm != PARTITION_ALGORITHM
        || manifest.source_fingerprint != source_fingerprint
    {
        return Ok(None);
    }
    validate_manifest_sources(&manifest, sources)?;
    validate_dataset_cache(cache_dir, &manifest)?;
    Ok(Some(manifest))
}

/// Validates all manifest invariants and exact shard sizes before training.
pub fn validate_dataset_cache(cache_dir: &Path, manifest: &DatasetCacheManifest) -> io::Result<()> {
    if manifest.version != CACHE_VERSION {
        return invalid_data(format!(
            "unsupported dataset cache version {}; expected {CACHE_VERSION}",
            manifest.version
        ));
    }
    if manifest.partition_algorithm != PARTITION_ALGORITHM {
        return invalid_data(format!(
            "unsupported partition algorithm {:?}",
            manifest.partition_algorithm
        ));
    }
    if manifest.record_size != DATASET_CACHE_RECORD_SIZE {
        return invalid_data(format!(
            "cache record size is {}, expected {DATASET_CACHE_RECORD_SIZE}",
            manifest.record_size
        ));
    }
    if manifest.shard_count != DATASET_CACHE_SHARDS || manifest.shards.len() != DATASET_CACHE_SHARDS
    {
        return invalid_data(format!(
            "cache has {} shard entries, expected {DATASET_CACHE_SHARDS}",
            manifest.shards.len()
        ));
    }
    if manifest.sources.is_empty() {
        return invalid_data("cache manifest has no sources");
    }

    let mut total_positions = 0_u64;
    let mut total_bytes = 0_u64;
    let mut source_totals = vec![0_u64; manifest.sources.len()];
    for (index, shard) in manifest.shards.iter().enumerate() {
        let expected_name = shard_file_name(index);
        if shard.path != expected_name || !is_plain_file_name(&shard.path) {
            return invalid_data(format!(
                "invalid path {:?} for shard {index}; expected {expected_name:?}",
                shard.path
            ));
        }
        if shard.source_counts.len() != manifest.sources.len() {
            return invalid_data(format!(
                "shard {index} has {} source counts, expected {}",
                shard.source_counts.len(),
                manifest.sources.len()
            ));
        }
        let source_count = checked_sum(&shard.source_counts, "shard source counts")?;
        if source_count != shard.count {
            return invalid_data(format!(
                "shard {index} source counts sum to {source_count}, expected {}",
                shard.count
            ));
        }
        let expected_size = shard
            .count
            .checked_mul(DATASET_CACHE_RECORD_SIZE as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "shard size overflow"))?;
        if shard.size_bytes != expected_size {
            return invalid_data(format!(
                "shard {index} manifest size is {}, expected {expected_size}",
                shard.size_bytes
            ));
        }
        let actual_size = fs::metadata(cache_dir.join(&shard.path))
            .map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!("failed to stat cache shard {:?}: {error}", shard.path),
                )
            })?
            .len();
        if actual_size != expected_size {
            return invalid_data(format!(
                "shard {index} file size is {actual_size}, expected {expected_size}"
            ));
        }
        for (total, count) in source_totals.iter_mut().zip(&shard.source_counts) {
            *total = total.checked_add(*count).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "source count overflow")
            })?;
        }
        total_positions = total_positions
            .checked_add(shard.count)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "position count overflow"))?;
        total_bytes = total_bytes.checked_add(expected_size).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "cache byte count overflow")
        })?;
    }

    if total_positions != manifest.total_positions {
        return invalid_data(format!(
            "shards contain {total_positions} positions, manifest says {}",
            manifest.total_positions
        ));
    }
    if total_bytes != manifest.total_bytes {
        return invalid_data(format!(
            "shards contain {total_bytes} bytes, manifest says {}",
            manifest.total_bytes
        ));
    }
    for (index, (source, actual)) in manifest.sources.iter().zip(source_totals).enumerate() {
        if source.positions != actual {
            return invalid_data(format!(
                "source {index} shard counts sum to {actual}, manifest says {}",
                source.positions
            ));
        }
    }
    let source_positions = checked_sum(
        &manifest
            .sources
            .iter()
            .map(|source| source.positions)
            .collect::<Vec<_>>(),
        "source positions",
    )?;
    if source_positions != manifest.total_positions {
        return invalid_data(format!(
            "sources contain {source_positions} positions, manifest says {}",
            manifest.total_positions
        ));
    }

    Ok(())
}

#[derive(Clone)]
pub struct DatasetShuffleLoader {
    directory: PathBuf,
    manifest: DatasetCacheManifest,
    shard_paths: Vec<String>,
    seed: u64,
    virtual_epochs: usize,
}

impl DatasetShuffleLoader {
    pub fn new(cache: DatasetCache, seed: u64, virtual_epochs: usize) -> io::Result<Self> {
        if virtual_epochs == 0 {
            return invalid_input("dataset virtual epochs must be greater than zero");
        }
        validate_dataset_cache(&cache.directory, &cache.manifest)?;
        let shard_paths = cache
            .manifest
            .shards
            .iter()
            .map(|shard| {
                cache
                    .directory
                    .join(&shard.path)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        Ok(Self {
            directory: cache.directory,
            manifest: cache.manifest,
            shard_paths,
            seed,
            virtual_epochs,
        })
    }

    pub fn positions_per_epoch(&self) -> u64 {
        self.manifest.total_positions
    }
}

impl DataLoader<ChessBoard> for DatasetShuffleLoader {
    fn data_file_paths(&self) -> &[String] {
        &self.shard_paths
    }

    fn count_positions(&self) -> Option<u64> {
        self.manifest
            .total_positions
            .checked_mul(self.virtual_epochs as u64)
    }

    fn map_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, start_position: usize, mut f: F) {
        let total_positions = self.manifest.total_positions;
        if total_positions == 0 {
            return;
        }
        let start_position = start_position as u64;
        let first_epoch = start_position / total_positions;
        if first_epoch >= self.virtual_epochs as u64 {
            return;
        }
        let first_epoch_offset = start_position % total_positions;
        let directory = self.directory.clone();
        let manifest = self.manifest.clone();
        let seed = self.seed;
        let virtual_epochs = self.virtual_epochs as u64;
        let (sender, receiver) = mpsc::sync_channel::<io::Result<Vec<ChessBoard>>>(1);

        thread::spawn(move || {
            for epoch in first_epoch..virtual_epochs {
                let order = epoch_shard_order(seed, epoch, manifest.shards.len());
                let mut skip = if epoch == first_epoch {
                    first_epoch_offset
                } else {
                    0
                };

                for shard_index in order {
                    let shard = &manifest.shards[shard_index];
                    if skip >= shard.count {
                        skip -= shard.count;
                        continue;
                    }

                    let loaded = read_and_shuffle_shard(
                        &directory.join(&shard.path),
                        shard.count,
                        shard_shuffle_seed(seed, epoch, shard_index),
                    )
                    .map(|mut boards| {
                        if skip > 0 {
                            boards.drain(..skip as usize);
                            skip = 0;
                        }
                        boards
                    });
                    let failed = loaded.is_err();
                    if sender.send(loaded).is_err() || failed {
                        return;
                    }
                }
            }
        });

        while let Ok(result) = receiver.recv() {
            let boards = result
                .unwrap_or_else(|error| panic!("failed to load dataset shuffle cache: {error}"));
            if !boards.is_empty() && f(&boards) {
                break;
            }
        }
    }
}

fn inspect_sources(source_paths: &[PathBuf]) -> io::Result<(Vec<InspectedSource>, String)> {
    if source_paths.is_empty() {
        return invalid_input("dataset shuffle requires at least one binpack source");
    }
    let mut paths = source_paths
        .iter()
        .map(fs::canonicalize)
        .collect::<io::Result<Vec<_>>>()?;
    paths.sort_by_key(|a| path_sort_key(a));
    paths.dedup();
    if paths.len() != source_paths.len() {
        return invalid_input("dataset shuffle source list contains duplicate files");
    }

    let mut sources = Vec::with_capacity(paths.len());
    for path in paths {
        let metadata = fs::metadata(&path)?;
        if !metadata.is_file() {
            return invalid_input(format!("dataset source is not a file: {}", path.display()));
        }
        let sha256 = sha256_file(&path)?;
        sources.push(InspectedSource {
            display_path: path.to_string_lossy().into_owned(),
            path,
            size_bytes: metadata.len(),
            sha256,
        });
    }

    let mut fingerprint = Sha256::new();
    fingerprint.update(b"mnue-p2-dataset-sources-v1\0");
    for source in &sources {
        fingerprint.update(source.display_path.as_bytes());
        fingerprint.update(&[0]);
        fingerprint.update(&source.size_bytes.to_le_bytes());
        fingerprint.update(source.sha256.as_bytes());
        fingerprint.update(&[0]);
    }
    Ok((sources, hex(&fingerprint.finalize())))
}

fn build_dataset_cache(
    sources: &[InspectedSource],
    source_fingerprint: &str,
    cache_dir: &Path,
) -> io::Result<DatasetCache> {
    let parent = cache_dir.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("cache directory has no parent: {}", cache_dir.display()),
        )
    })?;
    fs::create_dir_all(parent)?;
    let cache_name = cache_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "cache directory needs a UTF-8 file name",
            )
        })?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let staging = parent.join(format!(
        ".{cache_name}.staging-{}-{nonce}",
        std::process::id()
    ));
    let backup = parent.join(format!(
        ".{cache_name}.replaced-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&staging)?;

    let build_result = build_staging_cache(sources, source_fingerprint, &staging);
    let manifest = match build_result {
        Ok(manifest) => manifest,
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
    };
    let source_paths = sources
        .iter()
        .map(|source| source.path.clone())
        .collect::<Vec<_>>();
    let (verified_sources, verified_fingerprint) = match inspect_sources(&source_paths) {
        Ok(verified) => verified,
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
    };
    if verified_fingerprint != source_fingerprint
        || verified_sources
            .iter()
            .zip(sources)
            .any(|(verified, original)| {
                verified.display_path != original.display_path
                    || verified.size_bytes != original.size_bytes
                    || verified.sha256 != original.sha256
            })
    {
        let _ = fs::remove_dir_all(&staging);
        return invalid_data("dataset sources changed while the cache was being built; retry");
    }

    let had_old_cache = cache_dir.exists();
    if had_old_cache {
        fs::rename(cache_dir, &backup).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("failed to move stale cache aside: {error}"),
            )
        })?;
    }
    if let Err(error) = fs::rename(&staging, cache_dir) {
        if had_old_cache {
            let _ = fs::rename(&backup, cache_dir);
        }
        return Err(io::Error::new(
            error.kind(),
            format!("failed to atomically activate dataset cache: {error}"),
        ));
    }
    if had_old_cache {
        fs::remove_dir_all(&backup).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "new cache is active, but failed to remove replaced cache {}: {error}",
                    backup.display()
                ),
            )
        })?;
    }

    validate_dataset_cache(cache_dir, &manifest)?;
    Ok(DatasetCache {
        directory: cache_dir.to_path_buf(),
        manifest,
    })
}

fn build_staging_cache(
    sources: &[InspectedSource],
    source_fingerprint: &str,
    staging: &Path,
) -> io::Result<DatasetCacheManifest> {
    let mut writers = (0..DATASET_CACHE_SHARDS)
        .map(|index| {
            File::create(staging.join(shard_file_name(index)))
                .map(|file| BufWriter::with_capacity(1024 * 1024, file))
        })
        .collect::<io::Result<Vec<_>>>()?;
    let mut shard_buffers = (0..DATASET_CACHE_SHARDS)
        .map(|_| Vec::with_capacity(8192))
        .collect::<Vec<Vec<ChessBoard>>>();
    let mut shard_counts = vec![0_u64; DATASET_CACHE_SHARDS];
    let mut shard_source_counts = vec![vec![0_u64; sources.len()]; DATASET_CACHE_SHARDS];
    let mut source_positions = vec![0_u64; sources.len()];
    let mut global_ordinal = 0_u64;

    eprintln!(
        "Building normal P2 dataset shuffle cache from {} binpack files into {} shards...",
        sources.len(),
        DATASET_CACHE_SHARDS
    );
    for (source_index, source) in sources.iter().enumerate() {
        eprintln!(
            "  decoding source {}/{}: {}",
            source_index + 1,
            sources.len(),
            source.path.display()
        );
        let file = File::open(&source.path)?;
        let mut reader = CompressedTrainingDataEntryReader::new(file).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to initialise {}: {error}", source.path.display()),
            )
        })?;
        while reader.has_next() {
            let entry = reader.next();
            if !(-1..=1).contains(&entry.result) {
                return invalid_data(format!(
                    "invalid WDL result {} at source {} position {}",
                    entry.result,
                    source.path.display(),
                    source_positions[source_index]
                ));
            }
            let board = convert_to_bulletformat(&entry);
            let shard_index = partition_shard(global_ordinal);
            shard_buffers[shard_index].push(board);
            if shard_buffers[shard_index].len() == 8192 {
                ChessBoard::write_to_bin(&mut writers[shard_index], &shard_buffers[shard_index])?;
                shard_buffers[shard_index].clear();
            }
            shard_counts[shard_index] += 1;
            shard_source_counts[shard_index][source_index] += 1;
            source_positions[source_index] += 1;
            global_ordinal = global_ordinal.checked_add(1).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "position count overflow")
            })?;
        }
    }
    for (writer, buffer) in writers.iter_mut().zip(&mut shard_buffers) {
        ChessBoard::write_to_bin(writer, buffer)?;
        buffer.clear();
    }
    for writer in &mut writers {
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    drop(writers);

    let total_bytes = global_ordinal
        .checked_mul(DATASET_CACHE_RECORD_SIZE as u64)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "cache byte count overflow"))?;
    let manifest = DatasetCacheManifest {
        version: CACHE_VERSION,
        partition_algorithm: PARTITION_ALGORITHM.to_string(),
        source_fingerprint: source_fingerprint.to_string(),
        total_positions: global_ordinal,
        total_bytes,
        record_size: DATASET_CACHE_RECORD_SIZE,
        shard_count: DATASET_CACHE_SHARDS,
        sources: sources
            .iter()
            .zip(source_positions)
            .map(|(source, positions)| DatasetSourceManifest {
                path: source.display_path.clone(),
                size_bytes: source.size_bytes,
                sha256: source.sha256.clone(),
                positions,
            })
            .collect(),
        shards: (0..DATASET_CACHE_SHARDS)
            .map(|index| DatasetShardManifest {
                path: shard_file_name(index),
                count: shard_counts[index],
                size_bytes: shard_counts[index] * DATASET_CACHE_RECORD_SIZE as u64,
                source_counts: shard_source_counts[index].clone(),
            })
            .collect(),
    };
    validate_dataset_cache(staging, &manifest)?;

    let manifest_text = toml::to_string_pretty(&manifest).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to encode cache manifest: {error}"),
        )
    })?;
    let mut file = File::create(staging.join("manifest.toml"))?;
    file.write_all(manifest_text.as_bytes())?;
    file.sync_all()?;
    eprintln!(
        "  cache complete: {} positions, {} bytes",
        manifest.total_positions, manifest.total_bytes
    );
    Ok(manifest)
}

fn validate_manifest_sources(
    manifest: &DatasetCacheManifest,
    sources: &[InspectedSource],
) -> io::Result<()> {
    if manifest.sources.len() != sources.len() {
        return invalid_data(format!(
            "cache manifest has {} sources, expected {}",
            manifest.sources.len(),
            sources.len()
        ));
    }
    for (index, (manifest_source, source)) in manifest.sources.iter().zip(sources).enumerate() {
        if manifest_source.path != source.display_path
            || manifest_source.size_bytes != source.size_bytes
            || manifest_source.sha256 != source.sha256
        {
            return invalid_data(format!("cache source metadata mismatch at index {index}"));
        }
    }
    Ok(())
}

fn read_manifest(path: &Path) -> io::Result<DatasetCacheManifest> {
    let text = fs::read_to_string(path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to read cache manifest {}: {error}", path.display()),
        )
    })?;
    toml::from_str(&text).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid cache manifest {}: {error}", path.display()),
        )
    })
}

fn read_and_shuffle_shard(path: &Path, count: u64, seed: u64) -> io::Result<Vec<ChessBoard>> {
    let count = usize::try_from(count).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "shard does not fit address space",
        )
    })?;
    let mut boards = vec![ChessBoard::default(); count];
    let bytes = unsafe {
        // ChessBoard is repr(C), exactly 32 bytes, Copy, and already fully
        // initialised. Reading its native cache bytes back cannot create an
        // invalid Rust value because every field accepts every bit pattern.
        std::slice::from_raw_parts_mut(
            boards.as_mut_ptr().cast::<u8>(),
            count * DATASET_CACHE_RECORD_SIZE,
        )
    };
    let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    reader.read_exact(bytes)?;
    let mut trailing = [0_u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return invalid_data(format!("cache shard {} has trailing bytes", path.display()));
    }
    let mut rng = SplitMix64::new(seed);
    fisher_yates_shuffle(&mut boards, &mut rng);
    Ok(boards)
}

fn epoch_shard_order(seed: u64, epoch: u64, shard_count: usize) -> Vec<usize> {
    let mut order = (0..shard_count).collect::<Vec<_>>();
    let mut rng = SplitMix64::new(mix64(
        seed ^ EPOCH_ORDER_DOMAIN ^ epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15),
    ));
    fisher_yates_shuffle(&mut order, &mut rng);
    order
}

fn shard_shuffle_seed(seed: u64, epoch: u64, shard_index: usize) -> u64 {
    mix64(
        seed ^ SHARD_ORDER_DOMAIN
            ^ epoch.wrapping_mul(0xD134_2543_DE82_EF95)
            ^ (shard_index as u64).wrapping_mul(0xA24B_AED4_963E_E407),
    )
}

fn partition_shard(global_ordinal: u64) -> usize {
    (mix64(global_ordinal ^ PARTITION_SEED) % DATASET_CACHE_SHARDS as u64) as usize
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn shard_file_name(index: usize) -> String {
    format!("shard-{index:03}.bullet")
}

fn is_plain_file_name(path: &str) -> bool {
    Path::new(path).components().count() == 1
        && Path::new(path).file_name().and_then(|name| name.to_str()) == Some(path)
}

fn path_sort_key(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/").to_lowercase()
}

fn checked_sum(values: &[u64], label: &str) -> io::Result<u64> {
    values.iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(*value)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("{label} overflow")))
    })
}

fn invalid_data<T>(message: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

fn invalid_input<T>(message: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut sha = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        sha.update(&buffer[..read]);
    }
    Ok(hex(&sha.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    block_len: usize,
    length_bytes: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0; 64],
            block_len: 0,
            length_bytes: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.length_bytes = self
            .length_bytes
            .checked_add(input.len() as u64)
            .expect("SHA-256 input length overflow");
        if self.block_len > 0 {
            let take = (64 - self.block_len).min(input.len());
            self.block[self.block_len..self.block_len + take].copy_from_slice(&input[..take]);
            self.block_len += take;
            input = &input[take..];
            if self.block_len == 64 {
                let block = self.block;
                self.compress(&block);
                self.block_len = 0;
            } else {
                return;
            }
        }
        while input.len() >= 64 {
            let block: &[u8; 64] = input[..64].try_into().expect("slice length is fixed");
            self.compress(block);
            input = &input[64..];
        }
        self.block[..input.len()].copy_from_slice(input);
        self.block_len = input.len();
    }

    fn finalize(mut self) -> [u8; 32] {
        let length_bits = self.length_bytes.wrapping_mul(8);
        self.block[self.block_len] = 0x80;
        self.block_len += 1;
        if self.block_len > 56 {
            self.block[self.block_len..].fill(0);
            let block = self.block;
            self.compress(&block);
            self.block = [0; 64];
        } else {
            self.block[self.block_len..56].fill(0);
        }
        self.block[56..].copy_from_slice(&length_bits.to_be_bytes());
        let block = self.block;
        self.compress(&block);

        let mut output = [0_u8; 32];
        for (chunk, value) in output.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&value.to_be_bytes());
        }
        output
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut words = [0_u32; 64];
        for (word, bytes) in words.iter_mut().zip(block.chunks_exact(4)) {
            *word = u32::from_be_bytes(bytes.try_into().expect("word is four bytes"));
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let big1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(big1)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let big0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = big0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (state, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *state = state.wrapping_add(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::time::Instant;

    use super::*;
    use sfbinpack::{
        CompressedTrainingDataEntryWriter, TrainingDataEntry,
        chess::{
            coords::Square,
            r#move::{Move, MoveType},
            piece::Piece,
            position::Position,
        },
    };

    fn temp_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mnue-dataset-{name}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        path
    }

    fn synthetic_cache(counts: &[u64]) -> DatasetCache {
        assert_eq!(counts.len(), DATASET_CACHE_SHARDS);
        let directory = temp_dir("loader");
        let mut total = 0_u64;
        let mut shards = Vec::new();
        for (index, count) in counts.iter().copied().enumerate() {
            let boards = (0..count)
                .map(|offset| ChessBoard {
                    occ: total + offset + 1,
                    ..ChessBoard::default()
                })
                .collect::<Vec<_>>();
            let path = shard_file_name(index);
            let mut output = BufWriter::new(File::create(directory.join(&path)).unwrap());
            ChessBoard::write_to_bin(&mut output, &boards).unwrap();
            output.flush().unwrap();
            shards.push(DatasetShardManifest {
                path,
                count,
                size_bytes: count * DATASET_CACHE_RECORD_SIZE as u64,
                source_counts: vec![count],
            });
            total += count;
        }
        let manifest = DatasetCacheManifest {
            version: CACHE_VERSION,
            partition_algorithm: PARTITION_ALGORITHM.to_string(),
            source_fingerprint: "synthetic".to_string(),
            total_positions: total,
            total_bytes: total * DATASET_CACHE_RECORD_SIZE as u64,
            record_size: DATASET_CACHE_RECORD_SIZE,
            shard_count: DATASET_CACHE_SHARDS,
            sources: vec![DatasetSourceManifest {
                path: "synthetic.binpack".to_string(),
                size_bytes: 0,
                sha256: "synthetic".to_string(),
                positions: total,
            }],
            shards,
        };
        DatasetCache {
            directory,
            manifest,
        }
    }

    fn write_test_binpack(path: &Path, count: usize) {
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .unwrap();
        let mut writer = CompressedTrainingDataEntryWriter::new(file).unwrap();
        for index in 0..count {
            writer
                .write_entry(&TrainingDataEntry {
                    pos: Position::from_fen(
                        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    )
                    .unwrap(),
                    mv: Move::new(
                        Square::new(12),
                        Square::new(28),
                        MoveType::Normal,
                        Piece::none(),
                    ),
                    score: index as i16,
                    ply: index as u16,
                    result: [-1, 0, 1][index % 3],
                })
                .unwrap();
        }
        drop(writer);
    }

    fn sequence(loader: &DatasetShuffleLoader, start: usize) -> Vec<u64> {
        let mut values = Vec::new();
        loader.map_chunks(start, |chunk| {
            values.extend(chunk.iter().map(|board| board.occ));
            false
        });
        values
    }

    #[test]
    fn sha256_matches_standard_vectors() {
        let mut empty = Sha256::new();
        empty.update(b"");
        assert_eq!(
            hex(&empty.finalize()),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let mut abc = Sha256::new();
        abc.update(b"a");
        abc.update(b"bc");
        assert_eq!(
            hex(&abc.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn partition_uses_all_64_shards_and_preserves_total() {
        let mut counts = [0_u64; DATASET_CACHE_SHARDS];
        for ordinal in 0..1_000_000 {
            counts[partition_shard(ordinal)] += 1;
        }
        assert_eq!(counts.iter().sum::<u64>(), 1_000_000);
        assert!(
            counts
                .iter()
                .all(|count| *count > 14_000 && *count < 17_500)
        );
    }

    #[test]
    fn fixed_partition_keeps_contiguous_sources_proportional_in_every_shard() {
        let source_sizes = [253_900_u64, 241_100, 505_000];
        let mut source_counts = [[0_u64; 3]; DATASET_CACHE_SHARDS];
        let mut ordinal = 0_u64;
        for (source_index, size) in source_sizes.into_iter().enumerate() {
            for _ in 0..size {
                source_counts[partition_shard(ordinal)][source_index] += 1;
                ordinal += 1;
            }
        }
        for shard in source_counts {
            let total = shard.iter().sum::<u64>() as f64;
            for (actual, expected) in shard.into_iter().zip([0.2539, 0.2411, 0.5050]) {
                assert!((actual as f64 / total - expected).abs() < 0.02);
            }
        }
    }

    #[test]
    fn changed_source_fingerprint_invalidates_cache_before_shard_validation() {
        let directory = temp_dir("invalidate");
        let manifest = DatasetCacheManifest {
            version: CACHE_VERSION,
            partition_algorithm: PARTITION_ALGORITHM.to_string(),
            source_fingerprint: "old-fingerprint".to_string(),
            total_positions: 0,
            total_bytes: 0,
            record_size: DATASET_CACHE_RECORD_SIZE,
            shard_count: 0,
            sources: Vec::new(),
            shards: Vec::new(),
        };
        fs::write(
            directory.join("manifest.toml"),
            toml::to_string(&manifest).unwrap(),
        )
        .unwrap();
        assert!(
            reusable_manifest(&directory, &[], "new-fingerprint")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn staging_manifest_is_never_treated_as_active() {
        let parent = temp_dir("staging");
        let active = parent.join("cache");
        let staging = parent.join(".cache.staging-interrupted");
        fs::create_dir(&staging).unwrap();
        fs::write(staging.join("manifest.toml"), "version = 1").unwrap();
        assert!(
            reusable_manifest(&active, &[], "fingerprint")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn cache_build_reuses_unchanged_sources_and_rebuilds_changed_sources() {
        let directory = temp_dir("cache-build");
        let source = directory.join("source.binpack");
        let cache_dir = directory.join("cache");
        write_test_binpack(&source, 3);

        let first = ensure_dataset_cache(std::slice::from_ref(&source), &cache_dir).unwrap();
        assert_eq!(first.total_positions(), 3);
        let first_fingerprint = first.source_fingerprint().to_string();
        let reused = ensure_dataset_cache(std::slice::from_ref(&source), &cache_dir).unwrap();
        assert_eq!(reused.source_fingerprint(), first_fingerprint);

        write_test_binpack(&source, 4);
        let rebuilt = ensure_dataset_cache(std::slice::from_ref(&source), &cache_dir).unwrap();
        assert_eq!(rebuilt.total_positions(), 4);
        assert_ne!(rebuilt.source_fingerprint(), first_fingerprint);
        validate_dataset_cache(rebuilt.directory(), rebuilt.manifest()).unwrap();
    }

    #[test]
    #[ignore = "set MNUE_P2_ACCEPTANCE_DATA_DIR to build and verify the 5.31 GB real cache"]
    fn actual_generated_dataset_acceptance() {
        let data_dir = PathBuf::from(
            std::env::var("MNUE_P2_ACCEPTANCE_DATA_DIR")
                .expect("MNUE_P2_ACCEPTANCE_DATA_DIR is required"),
        );
        let cache_dir = std::env::var("MNUE_P2_ACCEPTANCE_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| data_dir.join(".mnue-p2-shuffle-cache"));
        let mut sources = fs::read_dir(&data_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("binpack"))
            })
            .collect::<Vec<_>>();
        sources.sort();

        let started = Instant::now();
        let cache = ensure_dataset_cache(&sources, &cache_dir).unwrap();
        eprintln!(
            "real cache build/reuse validation took {:?}",
            started.elapsed()
        );
        let manifest = cache.manifest();
        assert_eq!(manifest.total_positions, 165_936_222);
        assert_eq!(manifest.total_bytes, 5_309_959_104);
        assert_eq!(manifest.shards.len(), DATASET_CACHE_SHARDS);

        let source_groups = manifest
            .sources
            .iter()
            .map(|source| {
                let name = Path::new(&source.path)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_ascii_lowercase();
                if name.contains("endgame") {
                    0
                } else if name.contains("opening") {
                    1
                } else {
                    2
                }
            })
            .collect::<Vec<_>>();
        let expected = [0.2539_f64, 0.2411, 0.5050];
        for (shard_index, shard) in manifest.shards.iter().enumerate() {
            let mut groups = [0_u64; 3];
            for (source_index, count) in shard.source_counts.iter().enumerate() {
                groups[source_groups[source_index]] += count;
            }
            for (group, expected_ratio) in groups.into_iter().zip(expected) {
                let actual_ratio = group as f64 / shard.count as f64;
                assert!(
                    (actual_ratio - expected_ratio).abs() < 0.005,
                    "shard {shard_index} group ratio {actual_ratio:.6} differs from {expected_ratio:.6}"
                );
            }
        }
    }

    #[test]
    #[ignore = "set MNUE_P2_ACCEPTANCE_DATA_DIR to benchmark the 5.31 GB real cache"]
    fn actual_dataset_loader_throughput_acceptance() {
        let data_dir = PathBuf::from(
            std::env::var("MNUE_P2_ACCEPTANCE_DATA_DIR")
                .expect("MNUE_P2_ACCEPTANCE_DATA_DIR is required"),
        );
        let cache_dir = std::env::var("MNUE_P2_ACCEPTANCE_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| data_dir.join(".mnue-p2-shuffle-cache"));
        let mut sources = fs::read_dir(&data_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("binpack"))
            })
            .collect::<Vec<_>>();
        sources.sort();
        let cache = ensure_dataset_cache(&sources, &cache_dir).unwrap();
        let total = cache.total_positions();

        let dataset_loader = DatasetShuffleLoader::new(cache, 1, 1).unwrap();
        let started = Instant::now();
        let mut dataset_positions = 0_u64;
        dataset_loader.map_chunks(0, |chunk| {
            dataset_positions += chunk.len() as u64;
            false
        });
        let dataset_seconds = started.elapsed().as_secs_f64();
        assert_eq!(dataset_positions, total);
        let dataset_rate = dataset_positions as f64 / dataset_seconds;

        let path_strings = sources
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let path_refs = path_strings.iter().map(String::as_str).collect::<Vec<_>>();
        let threads = std::env::var("MNUE_P2_ACCEPTANCE_THREADS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(4);
        let native_loader = bullet_lib::value::loader::SfBinpackLoader::new_concat_multiple(
            &path_refs,
            256,
            threads,
            |_: &TrainingDataEntry| true,
        );
        let started = Instant::now();
        let mut native_positions = 0_u64;
        native_loader.map_chunks(0, |chunk| {
            native_positions += chunk.len() as u64;
            native_positions >= total
        });
        let native_seconds = started.elapsed().as_secs_f64();
        let native_rate = native_positions as f64 / native_seconds;

        eprintln!(
            "dataset loader: {dataset_positions} positions in {dataset_seconds:.3}s ({dataset_rate:.0} pos/s)"
        );
        eprintln!(
            "native loader: {native_positions} positions in {native_seconds:.3}s ({native_rate:.0} pos/s)"
        );
        assert!(
            dataset_rate >= native_rate * 0.95,
            "dataset loader is more than 5% slower than native: {dataset_rate:.0} vs {native_rate:.0} pos/s"
        );
    }

    #[test]
    fn loader_replays_seed_and_changes_each_epoch_without_duplicates() {
        let cache = synthetic_cache(&[3; DATASET_CACHE_SHARDS]);
        let loader = DatasetShuffleLoader::new(cache.clone(), 91, 2).unwrap();
        let replay = DatasetShuffleLoader::new(cache, 91, 2).unwrap();
        let values = sequence(&loader, 0);
        assert_eq!(values, sequence(&replay, 0));
        let per_epoch = loader.positions_per_epoch() as usize;
        assert_ne!(&values[..per_epoch], &values[per_epoch..]);
        for epoch in values.chunks_exact(per_epoch) {
            assert_eq!(epoch.len(), per_epoch);
            assert_eq!(
                epoch.iter().copied().collect::<HashSet<_>>().len(),
                per_epoch
            );
        }
    }

    #[test]
    fn resume_suffix_is_position_for_position_identical() {
        let cache = synthetic_cache(&[5; DATASET_CACHE_SHARDS]);
        let loader = DatasetShuffleLoader::new(cache, 0x1234, 3).unwrap();
        let full = sequence(&loader, 0);
        for start in [0, 1, 4, 5, 127, 319, 320, 517, full.len()] {
            assert_eq!(sequence(&loader, start), full[start..]);
        }
    }

    #[test]
    fn invalid_shard_size_is_rejected() {
        let cache = synthetic_cache(&[1; DATASET_CACHE_SHARDS]);
        let first = cache.directory.join(&cache.manifest.shards[0].path);
        File::options()
            .write(true)
            .open(first)
            .unwrap()
            .set_len(0)
            .unwrap();
        let error = DatasetShuffleLoader::new(cache, 1, 1)
            .err()
            .expect("corrupt cache must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn source_fingerprint_changes_with_contents_and_ignores_input_order() {
        let directory = temp_dir("fingerprint");
        let first = directory.join("a.binpack");
        let second = directory.join("b.binpack");
        fs::write(&first, b"a").unwrap();
        fs::write(&second, b"b").unwrap();
        let original = fingerprint_sources(&[first.clone(), second.clone()]).unwrap();
        assert_eq!(
            original,
            fingerprint_sources(&[second.clone(), first.clone()]).unwrap()
        );
        fs::write(&second, b"changed").unwrap();
        assert_ne!(original, fingerprint_sources(&[first, second]).unwrap());
    }
}
