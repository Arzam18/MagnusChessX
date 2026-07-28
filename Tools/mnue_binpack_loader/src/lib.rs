use std::{fs::File, sync::mpsc, thread};

use bullet_lib::value::loader::DataLoader;
use bulletformat::ChessBoard;
use sfbinpack::{
    CompressedTrainingDataEntryReader, TrainingDataEntry,
    chess::{color::Color, piecetype::PieceType},
};

const SOURCE_INTERLEAVE_POSITIONS: usize = 16 * 1024;
const SHUFFLE_SEED_DOMAIN: u64 = 0xB17B_1A5E_DA7A_5EED;

#[derive(Clone)]
pub struct StartPositionDataLoader<D> {
    inner: D,
}

impl<D> StartPositionDataLoader<D> {
    pub fn new(inner: D) -> Self {
        Self { inner }
    }
}

impl<T, D> DataLoader<T> for StartPositionDataLoader<D>
where
    T: 'static,
    D: DataLoader<T>,
{
    fn data_file_paths(&self) -> &[String] {
        self.inner.data_file_paths()
    }

    fn count_positions(&self) -> Option<u64> {
        self.inner.count_positions()
    }

    fn map_chunks<F: FnMut(&[T]) -> bool>(&self, start_position: usize, mut f: F) {
        let mut remaining = start_position;

        self.inner.map_chunks(0, |chunk| {
            if remaining >= chunk.len() {
                remaining -= chunk.len();
                return false;
            }

            let chunk = &chunk[remaining..];
            remaining = 0;
            !chunk.is_empty() && f(chunk)
        });
    }
}

#[derive(Clone)]
pub struct ResamplingSfBinpackLoader<T: Fn(&TrainingDataEntry) -> bool> {
    chunk_paths: Vec<String>,
    buffer_size: usize,
    threads: usize,
    filter: T,
    seed: u64,
    virtual_epochs: usize,
    chunk_sample: Option<usize>,
}

impl<T: Fn(&TrainingDataEntry) -> bool> ResamplingSfBinpackLoader<T> {
    pub fn new(
        chunk_paths: Vec<String>,
        buffer_size_mb: usize,
        threads: usize,
        filter: T,
        seed: u64,
        virtual_epochs: usize,
        chunk_sample: Option<usize>,
    ) -> Self {
        assert!(!chunk_paths.is_empty());
        assert!(threads > 0);
        assert!(virtual_epochs > 0);
        if let Some(sample) = chunk_sample {
            assert!(sample > 0 && sample <= chunk_paths.len());
        }

        Self {
            chunk_paths,
            buffer_size: buffer_size_mb * 1024 * 1024 / std::mem::size_of::<ChessBoard>() / 2,
            threads,
            filter,
            seed,
            virtual_epochs,
            chunk_sample,
        }
    }
}

impl<T> DataLoader<ChessBoard> for ResamplingSfBinpackLoader<T>
where
    T: Fn(&TrainingDataEntry) -> bool + Clone + Send + Sync + 'static,
{
    fn data_file_paths(&self) -> &[String] {
        &self.chunk_paths
    }

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, mut f: F) {
        let chunk_paths = self.chunk_paths.clone();
        let buffer_size = self.buffer_size;
        let threads = self.threads;
        let filter = self.filter.clone();
        let seed = self.seed;
        let virtual_epochs = self.virtual_epochs;
        let chunk_sample = self.chunk_sample;

        let reader_buffer_size = 16384 * threads;
        let (reader_sender, reader_receiver) = mpsc::sync_channel::<Vec<TrainingDataEntry>>(8);
        let (reader_msg_sender, reader_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut buffer = Vec::with_capacity(reader_buffer_size);
            let mut schedule =
                ResamplingChunkSchedule::new(chunk_paths, seed, virtual_epochs, chunk_sample);
            let chunks_per_epoch = schedule.chunks_per_epoch();

            'dataloading: loop {
                let pool = schedule.next_pool();
                for epoch in pool.chunks(chunks_per_epoch) {
                    let mut readers = epoch
                        .iter()
                        .map(|path| InterleavedReader::open(path))
                        .collect::<Vec<_>>();

                    while !readers.is_empty() {
                        let lane_index = least_progress_index(readers.iter().enumerate().map(
                            |(index, lane)| (index, lane.reader.read_bytes(), lane.total_bytes),
                        ))
                        .expect("an active binpack reader must exist");

                        for _ in 0..SOURCE_INTERLEAVE_POSITIONS {
                            if !readers[lane_index].reader.has_next() {
                                break;
                            }

                            buffer.push(readers[lane_index].reader.next());

                            if buffer.len() == reader_buffer_size {
                                if reader_msg_receiver.try_recv().unwrap_or(false)
                                    || reader_sender.send(buffer).is_err()
                                {
                                    break 'dataloading;
                                }

                                buffer = Vec::with_capacity(reader_buffer_size);
                            }
                        }

                        if !readers[lane_index].reader.has_next() {
                            readers.remove(lane_index);
                        }
                    }
                }
            }
        });

        let (converted_sender, converted_receiver) =
            mpsc::sync_channel::<Vec<ChessBoard>>(4 * threads);
        let (converted_msg_sender, converted_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let filter = &filter;
            let mut should_break = false;
            'dataloading: while let Ok(unfiltered) = reader_receiver.recv() {
                if should_break || converted_msg_receiver.try_recv().unwrap_or(false) {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                thread::scope(|s| {
                    let chunk_size = unfiltered.len().div_ceil(threads);
                    let mut handles = Vec::new();

                    for chunk in unfiltered.chunks(chunk_size) {
                        let handle = s.spawn(move || {
                            let mut buffer = Vec::with_capacity(chunk_size);

                            for entry in chunk {
                                if filter(entry) {
                                    buffer.push(convert_to_bulletformat(entry));
                                }
                            }

                            buffer
                        });

                        handles.push(handle);
                    }

                    for handle in handles {
                        let converted = handle.join().unwrap();
                        if converted_sender.send(converted).is_err() {
                            should_break = true;
                            break;
                        }
                    }
                });

                if should_break {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }
            }
        });

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut shuffle_buffer = Vec::with_capacity(buffer_size);
            let mut shuffle_rng = SplitMix64::new(seed ^ SHUFFLE_SEED_DOMAIN);

            'dataloading: while let Ok(converted) = converted_receiver.recv() {
                for entry in converted {
                    shuffle_buffer.push(entry);

                    if shuffle_buffer.len() == buffer_size {
                        fisher_yates_shuffle(&mut shuffle_buffer, &mut shuffle_rng);

                        if buffer_msg_receiver.try_recv().unwrap_or(false)
                            || buffer_sender.send(shuffle_buffer).is_err()
                        {
                            converted_msg_sender.send(true).unwrap();
                            break 'dataloading;
                        }

                        shuffle_buffer = Vec::with_capacity(buffer_size);
                    }
                }
            }
        });

        'dataloading: while let Ok(shuffle_buffer) = buffer_receiver.recv() {
            if f(&shuffle_buffer) {
                buffer_msg_sender.send(true).unwrap();
                break 'dataloading;
            }
        }
    }
}

struct InterleavedReader {
    reader: CompressedTrainingDataEntryReader<File>,
    total_bytes: u64,
}

impl InterleavedReader {
    fn open(path: &str) -> Self {
        let file = File::open(path).unwrap_or_else(|error| {
            panic!("failed to open binpack {path}: {error}");
        });
        let total_bytes = file
            .metadata()
            .unwrap_or_else(|error| {
                panic!("failed to read binpack metadata for {path}: {error}");
            })
            .len()
            .max(1);
        let reader = CompressedTrainingDataEntryReader::new(file).unwrap_or_else(|error| {
            panic!("failed to initialise binpack reader for {path}: {error}");
        });

        Self {
            reader,
            total_bytes,
        }
    }
}

fn least_progress_index<I>(progress: I) -> Option<usize>
where
    I: IntoIterator<Item = (usize, u64, u64)>,
{
    progress
        .into_iter()
        .min_by(|a, b| {
            let a_scaled = u128::from(a.1) * u128::from(b.2);
            let b_scaled = u128::from(b.1) * u128::from(a.2);
            a_scaled.cmp(&b_scaled).then_with(|| a.0.cmp(&b.0))
        })
        .map(|(index, _, _)| index)
}

struct ResamplingChunkSchedule {
    chunks: Vec<String>,
    rng: SplitMix64,
    virtual_epochs: usize,
    chunk_sample: Option<usize>,
}

impl ResamplingChunkSchedule {
    fn new(
        chunks: Vec<String>,
        seed: u64,
        virtual_epochs: usize,
        chunk_sample: Option<usize>,
    ) -> Self {
        Self {
            chunks,
            rng: SplitMix64::new(seed),
            virtual_epochs,
            chunk_sample,
        }
    }

    fn chunks_per_epoch(&self) -> usize {
        self.chunk_sample.unwrap_or(self.chunks.len())
    }

    fn next_pool(&mut self) -> Vec<String> {
        let per_epoch = self.chunks_per_epoch();
        let mut pool = Vec::with_capacity(per_epoch * self.virtual_epochs);
        for _ in 0..self.virtual_epochs {
            let mut epoch = self.chunks.clone();
            fisher_yates_shuffle(&mut epoch, &mut self.rng);
            pool.extend(epoch.into_iter().take(per_epoch));
        }
        pool
    }
}

fn convert_to_bulletformat(entry: &TrainingDataEntry) -> ChessBoard {
    let mut bbs = [0; 8];

    let stm = usize::from(entry.pos.side_to_move().ordinal());
    let pc_bb = |pt| {
        entry.pos.pieces_bb_color(Color::Black, pt).bits()
            | entry.pos.pieces_bb_color(Color::White, pt).bits()
    };

    bbs[0] = entry.pos.pieces_bb(Color::White).bits();
    bbs[1] = entry.pos.pieces_bb(Color::Black).bits();
    bbs[2] = pc_bb(PieceType::Pawn);
    bbs[3] = pc_bb(PieceType::Knight);
    bbs[4] = pc_bb(PieceType::Bishop);
    bbs[5] = pc_bb(PieceType::Rook);
    bbs[6] = pc_bb(PieceType::Queen);
    bbs[7] = pc_bb(PieceType::King);

    let mut score = entry.score;
    let mut result = f32::from(1 + entry.result) / 2.0;

    if stm > 0 {
        score = -score;
        result = 1.0 - result;
    }

    ChessBoard::from_raw(bbs, stm, score, result).expect("Binpack must be malformed!")
}

fn fisher_yates_shuffle<T>(items: &mut [T], rng: &mut SplitMix64) {
    for i in (1..items.len()).rev() {
        let j = rng.next_bounded(i + 1);
        items.swap(i, j);
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_bounded(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        (self.next_u64() % upper as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct FixedChunkLoader {
        paths: Vec<String>,
        chunks: Vec<Vec<u8>>,
    }

    impl DataLoader<u8> for FixedChunkLoader {
        fn data_file_paths(&self) -> &[String] {
            &self.paths
        }

        fn map_chunks<F: FnMut(&[u8]) -> bool>(&self, start_position: usize, mut f: F) {
            assert_eq!(start_position, 0);
            for chunk in &self.chunks {
                if f(chunk) {
                    break;
                }
            }
        }
    }

    #[test]
    fn start_position_loader_skips_across_chunk_boundaries() {
        let loader = StartPositionDataLoader::new(FixedChunkLoader {
            paths: vec!["test".to_string()],
            chunks: vec![vec![0, 1], vec![2, 3, 4], vec![5, 6, 7]],
        });
        let mut seen = Vec::new();

        loader.map_chunks(4, |chunk| {
            seen.extend_from_slice(chunk);
            false
        });

        assert_eq!(seen, vec![4, 5, 6, 7]);
    }

    #[test]
    fn resampling_pools_are_not_repeated() {
        let mut schedule = ResamplingChunkSchedule::new(
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            1,
            3,
            None,
        );
        let first = schedule.next_pool();
        let second = schedule.next_pool();
        assert_ne!(first, second);
    }

    #[test]
    fn sample_mode_keeps_distinct_chunks_per_epoch() {
        let mut schedule = ResamplingChunkSchedule::new(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            7,
            5,
            Some(2),
        );
        let pool = schedule.next_pool();
        assert_eq!(pool.len(), 10);
        for epoch in pool.chunks(2) {
            assert_eq!(epoch.len(), 2);
            assert_ne!(epoch[0], epoch[1]);
        }
    }

    #[test]
    fn weighted_progress_selection_interleaves_unequal_sources() {
        let totals = [10_u64, 30_u64];
        let mut read = [0_u64; 2];
        let mut order = Vec::new();

        while read != totals {
            let index = least_progress_index(
                read.iter()
                    .zip(totals)
                    .enumerate()
                    .filter(|(_, (done, total))| done < &total)
                    .map(|(index, (done, total))| (index, *done, total)),
            )
            .unwrap();
            read[index] += 1;
            order.push(index);
        }

        assert_eq!(order.iter().filter(|&&index| index == 0).count(), 10);
        assert_eq!(order.iter().filter(|&&index| index == 1).count(), 30);
        assert!(order[..8].contains(&0));
        assert!(order[..8].contains(&1));
    }

    #[test]
    fn shuffle_rng_advances_between_buffers_and_replays_from_seed() {
        let seed = 1234 ^ SHUFFLE_SEED_DOMAIN;
        let mut first_rng = SplitMix64::new(seed);
        let mut first = (0..64).collect::<Vec<_>>();
        let mut second = first.clone();
        fisher_yates_shuffle(&mut first, &mut first_rng);
        fisher_yates_shuffle(&mut second, &mut first_rng);

        let mut replay_rng = SplitMix64::new(seed);
        let mut replay_first = (0..64).collect::<Vec<_>>();
        let mut replay_second = replay_first.clone();
        fisher_yates_shuffle(&mut replay_first, &mut replay_rng);
        fisher_yates_shuffle(&mut replay_second, &mut replay_rng);

        assert_ne!(first, second);
        assert_eq!(first, replay_first);
        assert_eq!(second, replay_second);
    }
}
