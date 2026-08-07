use std::{
    cell::RefCell,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

use bullet_compiler::tensor::TValue;
use bullet_gpu::runtime::Device;
use bullet_lib::{
    nn::{
        ExecutionContext, ModelBuilder, Shape,
        optimiser::{AdamWOptimiser, AdamWParams, Optimiser},
    },
    value::loader::{DataLoader as ValueDataLoader, GameResult, LoadableDataType},
};
use bullet_trainer::{
    Trainer,
    model::save::{ModelWeights, SavedFormat},
    run::{
        dataloader::{DataLoader, DataLoadingError, PreparedBatchHost},
        schedule::{TrainingSchedule, TrainingSteps},
    },
};
use bulletformat::ChessBoard;
use mnue_binpack_loader::{
    DatasetCache, DatasetShuffleLoader, ResamplingSfBinpackLoader, StartPositionDataLoader,
    ensure_dataset_cache, fingerprint_sources,
};
use mnue_data_schedule::{DataSchedule, InputDataKind, build_data_schedule, print_startup_log};

use super::{
    Arch, BINPACK_BUFFER_MB, Config, MNUE_MAGIC, MNUE_VERSION, QA, QB, RunManifest, SCALE,
    ShuffleMode, accept_all_binpack, dataset_cache_dir, die, install_run_manifest,
    metadata_data_fingerprint, new_scaled_affine, resolve_shuffle_mode, resume_run_manifest,
    validate_fresh_output_root, validate_resume_checkpoint, validate_resume_output_targets,
    write_i32, write_u32,
};

pub const FEATURE_MAPPING: &str = "canonical-hmirror-v1";

pub const COARSE_BUCKETS: usize = 16;
pub const FINE_BUCKETS: usize = 32;
pub const OUTPUT_BUCKETS: usize = 64;
pub const COARSE_HIDDEN: usize = 768;
pub const FINE_HIDDEN: usize = 256;
pub const HIDDEN: usize = COARSE_HIDDEN + FINE_HIDDEN;
pub const COARSE_INPUTS: usize = COARSE_BUCKETS * 10 * 64;
pub const FINE_INPUTS: usize = FINE_BUCKETS * 11 * 64;
pub const TOTAL_INPUTS: usize = COARSE_INPUTS + FINE_INPUTS;
pub const MAX_ACTIVE: usize = 32;
pub const ARCH_ID: u32 = 9;
pub const RAW_PAYLOAD_BYTES: usize = COARSE_INPUTS * COARSE_HIDDEN * 2
    + COARSE_HIDDEN * 2
    + FINE_INPUTS * FINE_HIDDEN * 2
    + FINE_HIDDEN * 2
    + OUTPUT_BUCKETS * 2 * HIDDEN * 2
    + OUTPUT_BUCKETS * 2;
pub const HEADERED_BYTES: usize = RAW_PAYLOAD_BYTES + 40;

const COARSE_STM: &str = "coarse_stm";
const COARSE_NTM: &str = "coarse_ntm";
const FINE_STM: &str = "fine_stm";
const FINE_NTM: &str = "fine_ntm";
const BUCKETS: &str = "buckets";
const TARGETS: &str = "targets";

type H32Trainer = Trainer<ExecutionContext, AdamWOptimiser, ()>;

#[derive(Clone, Debug, PartialEq, Eq)]
struct EncodedPosition {
    coarse_stm: Vec<usize>,
    coarse_ntm: Vec<usize>,
    fine_stm: Vec<usize>,
    fine_ntm: Vec<usize>,
    output_bucket: usize,
}

#[inline]
fn file_of(sq: u8) -> usize {
    usize::from(sq & 7)
}

#[inline]
fn mirror_for_king(ksq: u8) -> bool {
    file_of(ksq) >= 4
}

#[inline]
fn canonical_square(sq: u8, mirror: bool) -> usize {
    usize::from(if mirror { sq ^ 7 } else { sq })
}

#[inline]
fn canonical_king(ksq: u8) -> usize {
    canonical_square(ksq, mirror_for_king(ksq))
}

#[inline]
pub(crate) fn coarse_bucket(ksq: u8) -> usize {
    let king = canonical_king(ksq);
    (king >> 3) / 2 * 4 + (king & 7)
}

#[inline]
pub(crate) fn fine_bucket(ksq: u8) -> usize {
    let king = canonical_king(ksq);
    (king >> 3) * 4 + (king & 7)
}

#[inline]
fn coarse_feature(bucket: usize, rel_colour: usize, piece: usize, sq: usize) -> usize {
    (((bucket * 2 + rel_colour) * 5 + piece) * 64) + sq
}

#[inline]
fn fine_feature(bucket: usize, plane: usize, sq: usize) -> usize {
    ((bucket * 11 + plane) * 64) + sq
}

#[inline]
fn non_king_piece(piece: u8) -> Option<usize> {
    match piece & 7 {
        0..=4 => Some(usize::from(piece & 7)),
        _ => None,
    }
}

fn phase4_from_units(units: usize) -> usize {
    match units.min(24) {
        18..=24 => 0,
        12..=17 => 1,
        6..=11 => 2,
        _ => 3,
    }
}

fn phase4(pos: &ChessBoard) -> usize {
    phase4_from_units(
        pos.into_iter()
            .map(|(piece, _)| match piece & 7 {
                1 | 2 => 1,
                3 => 2,
                4 => 4,
                _ => 0,
            })
            .sum(),
    )
}

fn encode(pos: &ChessBoard) -> EncodedPosition {
    let stm_king = pos.our_ksq();
    let ntm_king = pos.opp_ksq();
    let stm_mirror = mirror_for_king(stm_king);
    let ntm_mirror = mirror_for_king(ntm_king);
    let stm_coarse_bucket = coarse_bucket(stm_king);
    let ntm_coarse_bucket = coarse_bucket(ntm_king);
    let stm_fine_bucket = fine_bucket(stm_king);
    let ntm_fine_bucket = fine_bucket(ntm_king);

    let mut encoded = EncodedPosition {
        coarse_stm: Vec::with_capacity(30),
        coarse_ntm: Vec::with_capacity(30),
        fine_stm: Vec::with_capacity(31),
        fine_ntm: Vec::with_capacity(31),
        output_bucket: phase4(pos) * COARSE_BUCKETS + stm_coarse_bucket,
    };

    for (piece, square) in *pos {
        let stm_colour = usize::from((piece & 8) != 0);
        let ntm_colour = stm_colour ^ 1;
        let stm_sq = canonical_square(square, stm_mirror);
        let ntm_sq = canonical_square(square ^ 56, ntm_mirror);

        if let Some(piece_index) = non_king_piece(piece) {
            encoded.coarse_stm.push(coarse_feature(
                stm_coarse_bucket,
                stm_colour,
                piece_index,
                stm_sq,
            ));
            encoded.coarse_ntm.push(coarse_feature(
                ntm_coarse_bucket,
                ntm_colour,
                piece_index,
                ntm_sq,
            ));
            encoded.fine_stm.push(fine_feature(
                stm_fine_bucket,
                stm_colour * 5 + piece_index,
                stm_sq,
            ));
            encoded.fine_ntm.push(fine_feature(
                ntm_fine_bucket,
                ntm_colour * 5 + piece_index,
                ntm_sq,
            ));
        } else if piece & 7 == 5 {
            if stm_colour == 1 {
                encoded
                    .fine_stm
                    .push(fine_feature(stm_fine_bucket, 10, stm_sq));
            }
            if ntm_colour == 1 {
                encoded
                    .fine_ntm
                    .push(fine_feature(ntm_fine_bucket, 10, ntm_sq));
            }
        }
    }

    debug_assert!(encoded.coarse_stm.len() <= 30);
    debug_assert!(encoded.coarse_ntm.len() <= 30);
    debug_assert!(encoded.fine_stm.len() <= 31);
    debug_assert!(encoded.fine_ntm.len() <= 31);
    encoded
}

fn fill_sparse(dst: &mut [i32], indices: &[usize]) {
    assert!(indices.len() <= dst.len());
    dst.fill(-1);
    for (slot, &index) in dst.iter_mut().zip(indices) {
        *slot = i32::try_from(index).expect("H32 feature index exceeds i32");
    }
}

fn target(pos: &ChessBoard) -> f32 {
    let result = match pos.result() {
        GameResult::Loss => 0.0,
        GameResult::Draw => 0.5,
        GameResult::Win => 1.0,
    };
    let score = f32::from(pos.score());
    let score_target = 1.0 / (1.0 + (-score / SCALE as f32).exp());
    0.75 * result + 0.25 * score_target
}

fn prepare_batch(data: &[ChessBoard], threads: usize) -> PreparedBatchHost {
    let len = data.len();
    let mut coarse_stm = vec![-1; len * MAX_ACTIVE];
    let mut coarse_ntm = vec![-1; len * MAX_ACTIVE];
    let mut fine_stm = vec![-1; len * MAX_ACTIVE];
    let mut fine_ntm = vec![-1; len * MAX_ACTIVE];
    let mut buckets = vec![0; len];
    let mut targets = vec![0.0; len];
    let workers = threads.max(1).min(len.max(1));
    let chunk_len = len.div_ceil(workers);

    std::thread::scope(|scope| {
        for ((((((positions, coarse_stm), coarse_ntm), fine_stm), fine_ntm), buckets), targets) in
            data.chunks(chunk_len)
                .zip(coarse_stm.chunks_mut(chunk_len * MAX_ACTIVE))
                .zip(coarse_ntm.chunks_mut(chunk_len * MAX_ACTIVE))
                .zip(fine_stm.chunks_mut(chunk_len * MAX_ACTIVE))
                .zip(fine_ntm.chunks_mut(chunk_len * MAX_ACTIVE))
                .zip(buckets.chunks_mut(chunk_len))
                .zip(targets.chunks_mut(chunk_len))
        {
            scope.spawn(move || {
                for (index, pos) in positions.iter().enumerate() {
                    let encoded = encode(pos);
                    let offset = index * MAX_ACTIVE;
                    fill_sparse(
                        &mut coarse_stm[offset..offset + MAX_ACTIVE],
                        &encoded.coarse_stm,
                    );
                    fill_sparse(
                        &mut coarse_ntm[offset..offset + MAX_ACTIVE],
                        &encoded.coarse_ntm,
                    );
                    fill_sparse(
                        &mut fine_stm[offset..offset + MAX_ACTIVE],
                        &encoded.fine_stm,
                    );
                    fill_sparse(
                        &mut fine_ntm[offset..offset + MAX_ACTIVE],
                        &encoded.fine_ntm,
                    );
                    buckets[index] = i32::try_from(encoded.output_bucket).unwrap();
                    targets[index] = target(pos);
                }
            });
        }
    });

    PreparedBatchHost {
        batch_size: len,
        inputs: [
            (COARSE_STM.to_string(), TValue::I32(coarse_stm)),
            (COARSE_NTM.to_string(), TValue::I32(coarse_ntm)),
            (FINE_STM.to_string(), TValue::I32(fine_stm)),
            (FINE_NTM.to_string(), TValue::I32(fine_ntm)),
            (BUCKETS.to_string(), TValue::I32(buckets)),
            (TARGETS.to_string(), TValue::F32(targets)),
        ]
        .into(),
    }
}

#[derive(Clone)]
struct H32BatchLoader<D> {
    inner: D,
    start_position: usize,
    threads: usize,
}

impl<D> DataLoader for H32BatchLoader<D>
where
    D: ValueDataLoader<ChessBoard>,
{
    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(
        self,
        batch_size: usize,
        mut callback: F,
    ) -> Result<(), DataLoadingError> {
        let mut pending = Vec::with_capacity(batch_size);
        self.inner.map_chunks(self.start_position, |mut chunk| {
            while !chunk.is_empty() {
                let needed = batch_size - pending.len();
                let take = needed.min(chunk.len());
                pending.extend_from_slice(&chunk[..take]);
                chunk = &chunk[take..];
                if pending.len() == batch_size {
                    let stop = callback(prepare_batch(&pending, self.threads));
                    pending.clear();
                    if stop {
                        return true;
                    }
                }
            }
            false
        });
        Ok(())
    }
}

fn build_trainer(config: &Config) -> H32Trainer {
    let builder = ModelBuilder::default();
    let targets = builder.new_dense_input(TARGETS, Shape::new(1, 1));
    let buckets = builder.new_sparse_input(BUCKETS, Shape::new(OUTPUT_BUCKETS, 1), 1);
    let coarse_stm = builder.new_sparse_input(COARSE_STM, Shape::new(COARSE_INPUTS, 1), MAX_ACTIVE);
    let coarse_ntm = builder.new_sparse_input(COARSE_NTM, Shape::new(COARSE_INPUTS, 1), MAX_ACTIVE);
    let fine_stm = builder.new_sparse_input(FINE_STM, Shape::new(FINE_INPUTS, 1), MAX_ACTIVE);
    let fine_ntm = builder.new_sparse_input(FINE_NTM, Shape::new(FINE_INPUTS, 1), MAX_ACTIVE);

    let coarse = new_scaled_affine(
        &builder,
        "coarse",
        COARSE_INPUTS,
        COARSE_HIDDEN,
        config.l0_init_scale,
    );
    let fine = new_scaled_affine(
        &builder,
        "fine",
        FINE_INPUTS,
        FINE_HIDDEN,
        config.fine_init_scale,
    );
    let head = builder.new_affine("head", 2 * HIDDEN, OUTPUT_BUCKETS);

    let stm = coarse
        .forward(coarse_stm)
        .screlu()
        .concat(fine.forward(fine_stm).screlu());
    let ntm = coarse
        .forward(coarse_ntm)
        .screlu()
        .concat(fine.forward(fine_ntm).screlu());
    let output = head.forward(stm.concat(ntm)).select(buckets);
    let loss = output.sigmoid().squared_error(targets);
    let device = Device::<ExecutionContext>::new(0).expect("failed to create Bullet device");
    let model = builder.build(device, loss, output);
    let optimiser =
        Optimiser::new(model, AdamWParams::default()).expect("failed to create AdamW optimiser");
    Trainer {
        optimiser,
        state: (),
    }
}

fn formats(quantised: bool) -> Vec<SavedFormat> {
    let mut formats = vec![
        SavedFormat::id("coarsew"),
        SavedFormat::id("coarseb"),
        SavedFormat::id("finew"),
        SavedFormat::id("fineb"),
        SavedFormat::id("headw").transpose(),
        SavedFormat::id("headb"),
    ];
    if quantised {
        formats = vec![
            SavedFormat::id("coarsew").round().quantise::<i16>(QA),
            SavedFormat::id("coarseb").round().quantise::<i16>(QA),
            SavedFormat::id("finew").round().quantise::<i16>(QA),
            SavedFormat::id("fineb").round().quantise::<i16>(QA),
            SavedFormat::id("headw")
                .round()
                .quantise::<i16>(QB)
                .transpose(),
            SavedFormat::id("headb").round().quantise::<i16>(QA * QB),
        ];
    }
    formats
}

fn write_weight_file(trainer: &H32Trainer, path: &Path, quantised: bool) -> io::Result<usize> {
    let store = ModelWeights::from(&trainer.optimiser.model);
    let mut file = fs::File::create(path)?;
    let mut bytes = 0;
    for format in formats(quantised) {
        let buffer = format.write_to_byte_buffer(&store)?;
        bytes += buffer.len();
        file.write_all(&buffer)?;
    }
    file.flush()?;
    Ok(bytes)
}

fn write_headered(raw: &Path, output: &Path) -> io::Result<()> {
    let payload = fs::read(raw)?;
    if payload.len() != RAW_PAYLOAD_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unexpected H32 payload size: got {}, expected {RAW_PAYLOAD_BYTES}",
                payload.len()
            ),
        ));
    }
    let mut file = fs::File::create(output)?;
    write_u32(&mut file, MNUE_MAGIC)?;
    write_u32(&mut file, MNUE_VERSION)?;
    write_u32(&mut file, ARCH_ID)?;
    write_u32(&mut file, TOTAL_INPUTS as u32)?;
    write_u32(&mut file, HIDDEN as u32)?;
    write_u32(&mut file, FINE_BUCKETS as u32)?;
    write_u32(&mut file, OUTPUT_BUCKETS as u32)?;
    write_i32(&mut file, SCALE)?;
    write_i32(&mut file, i32::from(QA))?;
    write_i32(&mut file, i32::from(QB))?;
    file.write_all(&payload)?;
    Ok(())
}

fn save_checkpoint(
    trainer: &H32Trainer,
    config: &Config,
    superbatch: usize,
    losses: &[(usize, usize, f32)],
) -> io::Result<()> {
    let checkpoint =
        PathBuf::from(&config.output_dir).join(format!("{}-{superbatch}", config.name));
    fs::create_dir_all(&checkpoint)?;
    let optimiser = checkpoint.join("optimiser_state");
    fs::create_dir_all(&optimiser)?;
    trainer
        .optimiser
        .write_to_checkpoint(optimiser.to_string_lossy().as_ref())
        .map_err(|error| io::Error::other(format!("optimiser checkpoint failed: {error:?}")))?;

    write_weight_file(trainer, &checkpoint.join("raw.bin"), false)?;
    let quantised = checkpoint.join("quantised.bin");
    let bytes = write_weight_file(trainer, &quantised, true)?;
    if bytes != RAW_PAYLOAD_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("H32 quantised payload is {bytes} bytes, expected {RAW_PAYLOAD_BYTES}"),
        ));
    }
    let mut log = fs::File::create(checkpoint.join("log.txt"))?;
    for &(sb, batch, loss) in losses {
        writeln!(log, "{sb},{batch},{loss}")?;
    }
    let network = PathBuf::from(&config.net_dir).join(format!("{}-{superbatch}.mnue", config.name));
    write_headered(&quantised, &network)?;
    println!("Saved [{}-{superbatch}]", config.name);
    println!("Wrote {}", network.display());
    Ok(())
}

fn low_level_schedule(config: &Config) -> TrainingSchedule<'static> {
    let initial = config.lr;
    let final_lr = config.lr_final;
    let final_superbatch = config.superbatches;
    TrainingSchedule {
        steps: TrainingSteps {
            batch_size: config.batch_size,
            batches_per_superbatch: config.batches_per_superbatch(),
            start_superbatch: config.start_superbatch,
            end_superbatch: config.superbatches,
        },
        lr_schedule: Box::new(move |_, superbatch| {
            if superbatch >= final_superbatch {
                final_lr
            } else {
                let progress = superbatch as f32 / final_superbatch as f32;
                initial * (final_lr / initial).powf(progress)
            }
        }),
        log_rate: 128,
    }
}

fn train_with_loader<D>(trainer: &mut H32Trainer, config: &Config, loader: D)
where
    D: ValueDataLoader<ChessBoard>,
{
    let start_position = usize::try_from(config.completed_positions())
        .unwrap_or_else(|_| die("H32 start position exceeds usize"));
    let loader = H32BatchLoader {
        inner: loader,
        start_position,
        threads: config.threads,
    };
    let losses = RefCell::new(Vec::new());
    let mut loss_sum = 0.0_f32;
    let mut loss_count = 0_usize;
    trainer
        .train_custom(
            low_level_schedule(config),
            loader,
            |_, superbatch, batch, loss| {
                loss_sum += loss;
                loss_count += 1;
                if batch % 32 == 0 || batch == config.batches_per_superbatch() {
                    losses
                        .borrow_mut()
                        .push((superbatch, batch, loss_sum / loss_count as f32));
                    loss_sum = 0.0;
                    loss_count = 0;
                }
            },
            |trainer, superbatch| {
                if superbatch % config.save_rate == 0 || superbatch == config.superbatches {
                    save_checkpoint(trainer, config, superbatch, &losses.borrow()).unwrap_or_else(
                        |error| die(format!("failed to save H32 checkpoint: {error}")),
                    );
                }
            },
        )
        .unwrap_or_else(|error| die(format!("H32 training failed: {error:?}")));
}

fn prepare_run(config: &Config) -> (DataSchedule, ShuffleMode, Option<DatasetCache>) {
    let data_schedule = build_data_schedule(config.data_schedule_options())
        .unwrap_or_else(|error| die(error.to_string()));
    let resume_manifest = if let Some(path) = &config.resume_from {
        resume_run_manifest(Path::new(path)).unwrap_or_else(|error| die(error))
    } else {
        None
    };
    let (shuffle_mode, legacy_resume) =
        resolve_shuffle_mode(config, &data_schedule, resume_manifest.as_ref())
            .unwrap_or_else(|error| die(error));
    if legacy_resume {
        die("P2-H32 has no legacy checkpoint format; start a fresh H32 run");
    }
    if let Some(path) = &config.resume_from {
        validate_resume_checkpoint(Path::new(path), config.start_superbatch);
    }
    validate_resume_output_targets(config);
    validate_fresh_output_root(config);

    if let Some(manifest) = &resume_manifest {
        let provisional = RunManifest::for_run(
            config,
            shuffle_mode,
            manifest.data_fingerprint.clone(),
            manifest.cache_fingerprint.clone(),
        );
        manifest
            .validate_resume(&provisional)
            .unwrap_or_else(|error| die(error));
        if shuffle_mode == ShuffleMode::Dataset {
            let fingerprint = fingerprint_sources(&data_schedule.chunk_paths)
                .unwrap_or_else(|error| die(format!("failed to fingerprint data: {error}")));
            if fingerprint != manifest.data_fingerprint {
                die("H32 resume data fingerprint does not match checkpoint manifest");
            }
        }
    }

    let dataset_cache = if shuffle_mode == ShuffleMode::Dataset {
        let cache_dir = dataset_cache_dir(config).unwrap_or_else(|error| die(error));
        let cache = ensure_dataset_cache(&data_schedule.chunk_paths, &cache_dir)
            .unwrap_or_else(|error| die(format!("dataset cache error: {error}")));
        let available = u128::from(cache.total_positions())
            .checked_mul(config.chunk_virtual_epochs as u128)
            .unwrap_or_else(|| die("dataset virtual epoch capacity overflow"));
        if config.effective_positions() > available {
            let required = config
                .effective_positions()
                .div_ceil(u128::from(cache.total_positions()));
            die(format!(
                "--chunk-virtual-epochs must be at least {required} for this H32 run"
            ));
        }
        Some(cache)
    } else {
        None
    };

    let fingerprint_paths = if data_schedule.kind == InputDataKind::SfBinpack {
        &data_schedule.chunk_paths
    } else {
        &data_schedule.scheduled_paths
    };
    let data_fingerprint = dataset_cache
        .as_ref()
        .map(|cache| cache.source_fingerprint().to_string())
        .unwrap_or_else(|| {
            metadata_data_fingerprint(fingerprint_paths).unwrap_or_else(|error| die(error))
        });
    let cache_fingerprint = dataset_cache.as_ref().map(DatasetCache::cache_fingerprint);
    let manifest = RunManifest::for_run(config, shuffle_mode, data_fingerprint, cache_fingerprint);
    if let Some(previous) = &resume_manifest {
        previous
            .validate_resume(&manifest)
            .unwrap_or_else(|error| die(error));
    }
    install_run_manifest(Path::new(&config.output_dir), &manifest)
        .unwrap_or_else(|error| die(error));

    (data_schedule, shuffle_mode, dataset_cache)
}

pub fn run(config: Config) {
    assert_eq!(TOTAL_INPUTS, 32_768);
    assert_eq!(RAW_PAYLOAD_BYTES, 27_527_296);
    assert_eq!(HEADERED_BYTES, 27_527_336);
    let (data_schedule, shuffle_mode, dataset_cache) = prepare_run(&config);
    fs::create_dir_all(&config.output_dir).expect("failed to create H32 output directory");
    fs::create_dir_all(&config.net_dir).expect("failed to create H32 network directory");

    println!("MNUE-P2-H32 hybrid run");
    println!("Architecture           : {}", Arch::P2H32.cli_name());
    println!("Feature mapping        : {FEATURE_MAPPING}");
    println!("Coarse branch          : {COARSE_INPUTS} -> {COARSE_HIDDEN}");
    println!("Fine branch            : {FINE_INPUTS} -> {FINE_HIDDEN}");
    println!("Output buckets         : {OUTPUT_BUCKETS}");
    println!("Shuffle mode           : {}", shuffle_mode.cli_name());
    println!("Requested positions    : {}", config.positions);
    println!("Effective positions    : {}", config.effective_positions());
    println!("Batch size             : {}", config.batch_size);
    println!(
        "Batches/superbatch     : {}",
        config.batches_per_superbatch()
    );
    println!("Superbatches           : {}", config.superbatches);
    println!("Start superbatch       : {}", config.start_superbatch);
    println!("Coarse init scale      : {}", config.l0_init_scale);
    println!("Fine init scale        : {}", config.fine_init_scale);
    println!("Raw network bytes      : {RAW_PAYLOAD_BYTES}");
    if let Some(cache) = &dataset_cache {
        println!("Shuffle cache dir      : {}", cache.directory().display());
    }
    print_startup_log(&data_schedule);

    let mut trainer = build_trainer(&config);
    if let Some(path) = &config.resume_from {
        println!("Loading H32 optimiser state from {path}");
        trainer
            .optimiser
            .load_from_checkpoint(
                Path::new(path)
                    .join("optimiser_state")
                    .to_string_lossy()
                    .as_ref(),
            )
            .unwrap_or_else(|error| die(format!("failed to load H32 checkpoint: {error:?}")));
    }

    let path_strings = data_schedule.path_strings();
    let path_refs = path_strings.iter().map(String::as_str).collect::<Vec<_>>();
    match data_schedule.kind {
        InputDataKind::Direct => {
            let loader = bullet_lib::value::loader::DirectSequentialDataLoader::new(&path_refs);
            train_with_loader(&mut trainer, &config, loader);
        }
        InputDataKind::SfBinpack if shuffle_mode == ShuffleMode::Dataset => {
            let cache = dataset_cache.expect("H32 dataset cache must exist");
            let loader =
                DatasetShuffleLoader::new(cache, data_schedule.seed, data_schedule.virtual_epochs)
                    .unwrap_or_else(|error| die(format!("dataset cache error: {error}")));
            train_with_loader(&mut trainer, &config, loader);
        }
        InputDataKind::SfBinpack if shuffle_mode == ShuffleMode::Interleaved => {
            let loader = StartPositionDataLoader::new(ResamplingSfBinpackLoader::new(
                data_schedule.chunk_path_strings(),
                BINPACK_BUFFER_MB,
                config.threads,
                accept_all_binpack,
                data_schedule.seed,
                data_schedule.virtual_epochs,
                data_schedule.chunk_sample,
            ));
            train_with_loader(&mut trainer, &config, loader);
        }
        InputDataKind::SfBinpack => {
            let loader = StartPositionDataLoader::new(
                bullet_lib::value::loader::SfBinpackLoader::new_concat_multiple(
                    &path_refs,
                    BINPACK_BUFFER_MB,
                    config.threads,
                    accept_all_binpack,
                ),
            );
            train_with_loader(&mut trainer, &config, loader);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board(fen: &str) -> ChessBoard {
        format!("{fen} | 0 | 0.5").parse().unwrap()
    }

    #[test]
    fn dimensions_and_payload_are_exact() {
        assert_eq!(COARSE_INPUTS, 10_240);
        assert_eq!(FINE_INPUTS, 22_528);
        assert_eq!(TOTAL_INPUTS, 32_768);
        assert_eq!(RAW_PAYLOAD_BYTES, 27_527_296);
        assert_eq!(HEADERED_BYTES, 27_527_336);
    }

    #[test]
    fn mirrored_positions_encode_identically() {
        let left = board("4k3/8/8/2q5/5N2/8/1P6/2K5 w - - 0 1");
        let right = board("3k4/8/8/5q2/2N5/8/6P1/5K2 w - - 0 1");
        assert_eq!(encode(&left), encode(&right));
    }

    #[test]
    fn each_perspective_has_exactly_one_enemy_king_feature() {
        let encoded = encode(&board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"));
        let king_plane = |index: &&usize| ((*index / 64) % 11) == 10;
        assert_eq!(encoded.fine_stm.iter().filter(king_plane).count(), 1);
        assert_eq!(encoded.fine_ntm.iter().filter(king_plane).count(), 1);
        assert_eq!(encoded.coarse_stm.len(), 1);
        assert_eq!(encoded.coarse_ntm.len(), 1);
    }

    #[test]
    fn phase4_boundaries_select_expected_output_buckets() {
        for (units, expected) in [
            (0, 3),
            (5, 3),
            (6, 2),
            (11, 2),
            (12, 1),
            (17, 1),
            (18, 0),
            (24, 0),
        ] {
            assert_eq!(phase4_from_units(units), expected);
        }
        let bare = board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        assert_eq!(phase4(&bare), 3);
        assert_eq!(
            encode(&bare).output_bucket,
            3 * 16 + coarse_bucket(bare.our_ksq())
        );
        let full = board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert_eq!(phase4(&full), 0);
    }

    #[test]
    fn canonical_buckets_fold_files_and_preserve_rank_resolution() {
        assert_eq!(fine_bucket(0), fine_bucket(7));
        assert_eq!(fine_bucket(8), fine_bucket(15));
        assert_ne!(fine_bucket(0), fine_bucket(8));
        assert_eq!(coarse_bucket(0), coarse_bucket(8));
        assert_ne!(coarse_bucket(0), coarse_bucket(16));
    }

    #[test]
    fn target_matches_existing_p2_blend() {
        let draw = board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        assert!((target(&draw) - 0.5).abs() < f32::EPSILON);
    }
}
