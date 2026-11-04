/*
MIT License

Copyright (c) 2026 Magnus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "Mnue.h"
#include "board/Position.h"
#include "board/MoveGen.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

// Symbols provided by mnue/MnueEmbedded.S — the embedded MNUE P2-family blob in .rodata.
// Weak: resolve to nullptr when the object is not linked (e.g. non-GCC builds).
extern "C" const char mnue_p2_embedded_data[] __attribute__((weak));
extern "C" const char mnue_p2_embedded_end[]   __attribute__((weak));

namespace magnus::mnue {
namespace {

using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using u32 = std::uint32_t;

constexpr u32 kMnueMagic = 0x45554E4Du; // "MNUE" in little-endian file bytes
constexpr u32 kMnueVersion = 1;
constexpr int kQa = 255;
constexpr int kQb = 64;
constexpr int kScale = 400;
constexpr int kClip = 255;
constexpr int kMnueMaterialMax = 78;
constexpr int kMnueCpMaxRaw = 32768;
// Display calibration only: search/eval use raw scores, while UCI cp should
// report ordinary opening advantages around the expected pawn-unit scale.
constexpr double kMnueCpBase = 104.0;
constexpr double kMnueCpMaterial = 12.0;
constexpr double kMnueCpEndgameDiscount = 6.0;
constexpr double kMnueCpMinDenominator = 96.0;
constexpr double kMnueCpMaxDenominator = 128.0;

constexpr int kSearchMaterialScaleBase = 856;
constexpr int kSearchMaterialScaleRange = 279;
constexpr int kSearchMaterialScaleDenominator = 1024;
constexpr int kSearchFiftyMoveScaleBase = 200;
constexpr int kSearchFiftyMoveClockClamp = 100;

constexpr std::array<double, 4> kWdlA{
    2.37727934,
    -12.88614270,
    28.77964720,
    82.47173049
};
constexpr std::array<double, 4> kWdlB{
    -0.44227540,
    4.20132633,
    -11.59643736,
    32.47824518
};
constexpr int kWdlPlyClamp = 240;
constexpr double kWdlPlyDivisor = 64.0;
constexpr double kWdlCpClamp = 2000.0;
constexpr int kWdlLowMaterialThreshold = 4;
constexpr double kWdlLowMaterialABonus = 130.0;
constexpr double kWdlLowMaterialBBonus = 22.0;

[[nodiscard]] constexpr double wdl_horner(
    const std::array<double, 4>& coefficients,
    double x
) noexcept {
    return ((coefficients[0] * x + coefficients[1]) * x + coefficients[2])
        * x + coefficients[3];
}

template<class Layout>
struct Network {
    bool loaded = false;
    int scale = kScale;
    std::string path;
    std::string error;

    // Row-major by feature: w0[feature][hidden]. This matches the existing
    // Chess768 loader/inference convention and keeps feature delta contiguous.
    std::vector<i16> w0;
    std::vector<i16> b0;

    // w1[bucket][perspective][hidden], perspective 0 = stm/us, 1 = nstm/them.
    std::vector<i16> w1;
    std::vector<std::int8_t> w1_i8;
    std::vector<i16> b1;
    int w1_max_abs = 0;
    bool forward_vnni_safe = false;
    bool forward_madd_safe = false;
    bool forward_i32_safe = false;

    [[nodiscard]] bool valid() const noexcept {
        return w0.size() == static_cast<std::size_t>(Layout::InputSize) * Layout::HiddenSize
            && b0.size() == static_cast<std::size_t>(Layout::HiddenSize)
            && w1.size() == static_cast<std::size_t>(Layout::OutputBuckets) * 2 * Layout::HiddenSize
            && b1.size() == static_cast<std::size_t>(Layout::OutputBuckets);
    }
};

struct H32Network {
    bool loaded = false;
    int scale = kScale;
    std::string path;
    std::string error;
    std::vector<i16> coarse_w;
    std::vector<i16> coarse_b;
    std::vector<i16> fine_w;
    std::vector<i16> fine_b;
    std::vector<i16> w1;
    std::vector<std::int8_t> w1_i8;
    std::vector<i16> b1;
    int w1_max_abs = 0;
    bool forward_vnni_safe = false;
    bool forward_madd_safe = false;
    bool forward_i32_safe = false;

    [[nodiscard]] bool valid() const noexcept {
        return coarse_w.size()
                == static_cast<std::size_t>(P2H32Layout::CoarseInputSize)
                    * P2H32Layout::CoarseHiddenSize
            && fine_w.size()
                == static_cast<std::size_t>(P2H32Layout::FineInputSize)
                    * P2H32Layout::FineHiddenSize
            && coarse_b.size() == P2H32Layout::CoarseHiddenSize
            && fine_b.size() == P2H32Layout::FineHiddenSize
            && w1.size()
                == static_cast<std::size_t>(P2H32Layout::OutputBuckets)
                    * 2 * P2H32Layout::HiddenSize
            && b1.size() == P2H32Layout::OutputBuckets;
    }
};

struct FileHeader {
    u32 magic;
    u32 version;
    u32 arch;
    u32 input_size;
    u32 hidden_size;
    u32 input_buckets;
    u32 output_buckets;
    i32 scale;
    i32 qa;
    i32 qb;
};

static_assert(sizeof(FileHeader) == 40);

Network<P2Layout> g_p2;
Network<P2K32Layout> g_p2k32;
H32Network g_p2h32;
Network<P2ProLayout> g_p2pro;
Network<P4Layout> g_p4;
std::string g_error;
std::atomic<u32> g_p2_generation{1};

enum class P2ActiveKind : std::uint8_t {
    None,
    P2,
    P2K32,
    P2H32,
    P2Pro
};

P2ActiveKind g_p2_active = P2ActiveKind::None;
const std::string kEmptyPath;

[[nodiscard]] const char* p2_kind_eval_name(P2ActiveKind kind) noexcept {
    switch (kind) {
        case P2ActiveKind::P2:    return "mnue-p2";
        case P2ActiveKind::P2K32: return "mnue-p2-k32";
        case P2ActiveKind::P2H32: return "mnue-p2-h32";
        case P2ActiveKind::P2Pro: return "mnue-p2pro";
        case P2ActiveKind::None:  return "mnue-p2";
    }
    return "mnue-p2";
}

[[nodiscard]] const char* p2_kind_arch_name(P2ActiveKind kind) noexcept {
    switch (kind) {
        case P2ActiveKind::P2:    return "P2";
        case P2ActiveKind::P2K32: return "P2-K32";
        case P2ActiveKind::P2H32: return "P2-H32";
        case P2ActiveKind::P2Pro: return "P2Pro";
        case P2ActiveKind::None:  return "P2";
    }
    return "P2";
}

[[nodiscard]] const char* p2_kind_short_name(P2ActiveKind kind) noexcept {
    switch (kind) {
        case P2ActiveKind::P2:    return "p2";
        case P2ActiveKind::P2K32: return "p2-k32";
        case P2ActiveKind::P2H32: return "p2-h32";
        case P2ActiveKind::P2Pro: return "p2pro";
        case P2ActiveKind::None:  return "p2";
    }
    return "p2";
}

template<typename T>
[[nodiscard]] bool read_exact(std::istream& in, T* data, std::size_t count) {
    if (count == 0)
        return true;
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(sizeof(T) * count));
    return static_cast<bool>(in);
}

template<class Layout>
[[nodiscard]] constexpr std::uintmax_t payload_bytes() noexcept {
    return static_cast<std::uintmax_t>(
        static_cast<std::size_t>(Layout::InputSize) * Layout::HiddenSize +
        static_cast<std::size_t>(Layout::HiddenSize) +
        static_cast<std::size_t>(Layout::OutputBuckets) * 2 * Layout::HiddenSize +
        static_cast<std::size_t>(Layout::OutputBuckets)
    ) * sizeof(i16);
}

template<class Layout>
[[nodiscard]] constexpr std::uintmax_t file_bytes() noexcept {
    return sizeof(FileHeader) + payload_bytes<Layout>();
}

[[nodiscard]] constexpr std::uintmax_t h32_payload_bytes() noexcept {
    return static_cast<std::uintmax_t>(27'527'296);
}

[[nodiscard]] constexpr std::uintmax_t h32_file_bytes() noexcept {
    return sizeof(FileHeader) + h32_payload_bytes();
}

template<class Layout>
void clear_network(Network<Layout>& net) noexcept {
    net.loaded = false;
    net.scale = kScale;
    net.path.clear();
    net.error.clear();
    net.w0.clear();
    net.b0.clear();
    net.w1.clear();
    net.w1_i8.clear();
    net.b1.clear();
    net.w1_max_abs = 0;
    net.forward_vnni_safe = false;
    net.forward_madd_safe = false;
    net.forward_i32_safe = false;
}

void clear_network(H32Network& net) noexcept {
    net.loaded = false;
    net.scale = kScale;
    net.path.clear();
    net.error.clear();
    net.coarse_w.clear();
    net.coarse_b.clear();
    net.fine_w.clear();
    net.fine_b.clear();
    net.w1.clear();
    net.w1_i8.clear();
    net.b1.clear();
    net.w1_max_abs = 0;
    net.forward_vnni_safe = false;
    net.forward_madd_safe = false;
    net.forward_i32_safe = false;
}

template<class Layout>
void resize_network(Network<Layout>& net) {
    net.w0.resize(static_cast<std::size_t>(Layout::InputSize) * Layout::HiddenSize);
    net.b0.resize(static_cast<std::size_t>(Layout::HiddenSize));
    net.w1.resize(static_cast<std::size_t>(Layout::OutputBuckets) * 2 * Layout::HiddenSize);
    net.b1.resize(static_cast<std::size_t>(Layout::OutputBuckets));
}

void resize_network(H32Network& net) {
    net.coarse_w.resize(
        static_cast<std::size_t>(P2H32Layout::CoarseInputSize)
        * P2H32Layout::CoarseHiddenSize
    );
    net.coarse_b.resize(P2H32Layout::CoarseHiddenSize);
    net.fine_w.resize(
        static_cast<std::size_t>(P2H32Layout::FineInputSize)
        * P2H32Layout::FineHiddenSize
    );
    net.fine_b.resize(P2H32Layout::FineHiddenSize);
    net.w1.resize(
        static_cast<std::size_t>(P2H32Layout::OutputBuckets)
        * 2 * P2H32Layout::HiddenSize
    );
    net.b1.resize(P2H32Layout::OutputBuckets);
}

template<class Layout>
[[nodiscard]] constexpr int max_safe_i32_w1_abs() noexcept {
    constexpr i64 lane_terms = (Layout::HiddenSize + 7) / 8;
    constexpr i64 max_lane_sum_per_weight =
        static_cast<i64>(kClip) * static_cast<i64>(kClip) * lane_terms;
    return static_cast<int>(
        static_cast<i64>(std::numeric_limits<i32>::max()) / max_lane_sum_per_weight
    );
}

template<class Layout, class Net>
void refresh_network_traits(Net& net) noexcept {
    int max_abs = 0;
    bool w1_i8_safe = true;
    for (const i16 w : net.w1) {
        const int abs_w = w < 0 ? -static_cast<int>(w) : static_cast<int>(w);
        max_abs = std::max(max_abs, abs_w);
        if (w < -128 || w > 127)
            w1_i8_safe = false;
    }

    net.w1_max_abs = max_abs;
    net.forward_vnni_safe = w1_i8_safe;
    net.forward_madd_safe = max_abs <= 128;
    net.forward_i32_safe = max_abs <= max_safe_i32_w1_abs<Layout>();

    if (w1_i8_safe) {
        net.w1_i8.resize(net.w1.size());
        for (std::size_t i = 0; i < net.w1.size(); ++i)
            net.w1_i8[i] = static_cast<std::int8_t>(net.w1[i]);
    } else {
        net.w1_i8.clear();
    }
}

template<class Layout>
[[nodiscard]] bool read_network_payload(std::istream& in, Network<Layout>& net) {
    resize_network(net);

    if (!read_exact(in, net.w0.data(), net.w0.size()) ||
        !read_exact(in, net.b0.data(), net.b0.size()) ||
        !read_exact(in, net.w1.data(), net.w1.size()) ||
        !read_exact(in, net.b1.data(), net.b1.size())) {
        clear_network(net);
        return false;
    }

    refresh_network_traits<Layout>(net);
    return net.valid();
}

[[nodiscard]] bool read_network_payload(std::istream& in, H32Network& net) {
    resize_network(net);
    if (!read_exact(in, net.coarse_w.data(), net.coarse_w.size())
        || !read_exact(in, net.coarse_b.data(), net.coarse_b.size())
        || !read_exact(in, net.fine_w.data(), net.fine_w.size())
        || !read_exact(in, net.fine_b.data(), net.fine_b.size())
        || !read_exact(in, net.w1.data(), net.w1.size())
        || !read_exact(in, net.b1.data(), net.b1.size())) {
        clear_network(net);
        return false;
    }
    refresh_network_traits<P2H32Layout>(net);
    return net.valid();
}


[[nodiscard]] u32 next_p2_generation() noexcept {
    u32 next = g_p2_generation.load(std::memory_order_relaxed) + 1;
    if (next == 0)
        next = 1;
    g_p2_generation.store(next, std::memory_order_release);
    return next;
}

[[nodiscard]] inline std::uint8_t p2_perspective_mask(Color perspective) noexcept {
    return static_cast<std::uint8_t>(1u << static_cast<unsigned>(perspective));
}


// Core parsing: reads MNUE header + payload from any std::istream.
// `known_size` is the total byte size when available (0 = unknown).
// Used by both file-based load_network() and embedded load_p2_embedded().
template<class Layout>
[[nodiscard]] bool load_network_from_stream(
    std::istream& in, Network<Layout>& net, const std::string& path,
    std::uintmax_t known_size = 0) {
    clear_network(net);

    FileHeader header{};
    if (!read_exact(in, &header, 1)) {
        net.error = "could not read MNUE header";
        g_error = net.error;
        return false;
    }

    if (header.magic != kMnueMagic || header.version != kMnueVersion) {
        // Headerless fallback: only when size matches raw payload exactly.
        // A file that happens to have enough bytes but is the wrong format
        // must NOT be silently accepted.
        if (known_size == 0) {
            in.clear();
            in.seekg(0, std::ios::end);
            const auto endpos = in.tellg();
            if (endpos >= 0)
                known_size = static_cast<std::uintmax_t>(endpos);
            in.seekg(0, std::ios::beg);
        }
        if (known_size == payload_bytes<Layout>()) {
            in.clear();
            in.seekg(0, std::ios::beg);
            if (read_network_payload(in, net)) {
                net.loaded = true;
                net.path = path;
                g_error.clear();
                return true;
            }
        }

        net.error = "bad MNUE magic/version";
        g_error = net.error;
        return false;
    }

    if (header.arch != Layout::ArchId ||
        header.input_size != Layout::InputSize ||
        header.hidden_size != Layout::HiddenSize ||
        header.input_buckets != Layout::InputBuckets ||
        header.output_buckets != Layout::OutputBuckets ||
        header.qa != kQa ||
        header.qb != kQb) {
        net.error = "MNUE header dimensions do not match requested layout";
        g_error = net.error;
        return false;
    }

    net.scale = header.scale;
    if (!read_network_payload(in, net)) {
        net.error = "truncated MNUE weight file";
        g_error = net.error;
        return false;
    }

    if (!net.valid()) {
        clear_network(net);
        net.error = "internal MNUE size check failed";
        g_error = net.error;
        return false;
    }

    net.loaded = true;
    net.path = path;
    g_error.clear();
    return true;
}

template<class Layout>
[[nodiscard]] bool load_network(Network<Layout>& net, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        net.error = "could not open MNUE file: " + path;
        g_error = net.error;
        return false;
    }
    std::error_code ec;
    const auto file_sz = std::filesystem::file_size(path, ec);
    return load_network_from_stream(
        in, net, path,
        ec ? static_cast<std::uintmax_t>(0) : file_sz
    );
}

[[nodiscard]] bool load_h32_from_stream(
    std::istream& in,
    H32Network& net,
    const std::string& path,
    std::uintmax_t known_size = 0
) {
    clear_network(net);
    FileHeader header{};
    if (!read_exact(in, &header, 1)) {
        net.error = "could not read H32 header";
        g_error = net.error;
        return false;
    }

    if (header.magic != kMnueMagic || header.version != kMnueVersion) {
        if (known_size == h32_payload_bytes()) {
            in.clear();
            in.seekg(0, std::ios::beg);
            if (read_network_payload(in, net)) {
                net.loaded = true;
                net.path = path;
                g_error.clear();
                return true;
            }
        }
        net.error = "bad H32 magic/version";
        g_error = net.error;
        return false;
    }

    if (header.arch != P2H32Layout::ArchId
        || header.input_size != P2H32Layout::InputSize
        || header.hidden_size != P2H32Layout::HiddenSize
        || header.input_buckets != P2H32Layout::InputBuckets
        || header.output_buckets != P2H32Layout::OutputBuckets
        || header.qa != kQa
        || header.qb != kQb) {
        net.error = "MNUE header dimensions do not match P2-H32";
        g_error = net.error;
        return false;
    }

    net.scale = header.scale;
    if (!read_network_payload(in, net)) {
        net.error = "truncated P2-H32 weight file";
        g_error = net.error;
        return false;
    }
    net.loaded = true;
    net.path = path;
    g_error.clear();
    return true;
}

[[nodiscard]] bool load_network(H32Network& net, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        net.error = "could not open MNUE file: " + path;
        g_error = net.error;
        return false;
    }
    std::error_code ec;
    const auto file_sz = std::filesystem::file_size(path, ec);
    return load_h32_from_stream(
        in,
        net,
        path,
        ec ? static_cast<std::uintmax_t>(0) : file_sz
    );
}

[[nodiscard]] inline Square flip_vertical_sq(Square sq) noexcept {
    return static_cast<Square>(static_cast<int>(sq) ^ 56);
}

[[nodiscard]] inline Square relative_square(Color perspective, Square sq) noexcept {
    return perspective == WHITE ? sq : flip_vertical_sq(sq);
}

[[nodiscard]] inline int file_of(Square sq) noexcept {
    return static_cast<int>(sq) & 7;
}

[[nodiscard]] inline int rank_of(Square sq) noexcept {
    return static_cast<int>(sq) >> 3;
}

[[nodiscard]] inline Square king_square_of(const Position& pos, Color color) noexcept {
    const Square sq = king_square(pos, color);
    return sq == NO_SQ ? static_cast<Square>(0) : sq;
}

[[nodiscard]] inline int nonking_piece_index(PieceType pt) noexcept {
    switch (pt) {
        case PAWN:   return 0;
        case KNIGHT: return 1;
        case BISHOP: return 2;
        case ROOK:   return 3;
        case QUEEN:  return 4;
        default:     return -1;
    }
}

[[nodiscard]] inline int king_zone16(Square relative_king_sq) noexcept {
    const int file_group = file_of(relative_king_sq) / 2; // 0..3
    const int rank_group = rank_of(relative_king_sq) / 2; // 0..3
    return rank_group * 4 + file_group;                  // 0..15
}

[[nodiscard]] inline int king_zone32(Square relative_king_sq) noexcept {
    const int file = file_of(relative_king_sq);
    const int file_group = std::min(file, 7 - file); // 0..3, horizontal mirror
    const int rank = rank_of(relative_king_sq);      // 0..7
    return rank * 4 + file_group;                    // 0..31
}

struct H32Perspective {
    std::uint16_t coarse_base;
    std::uint16_t fine_base;
    std::uint8_t coarse_bucket;
    std::uint8_t fine_bucket;
    std::uint8_t square_xor;
    bool mirror;
};

static_assert(sizeof(H32Perspective) == 8);

struct H32FeatureDelta {
    static constexpr std::uint16_t None = std::numeric_limits<std::uint16_t>::max();
    std::uint16_t coarse = None;
    std::uint16_t fine = None;

    [[nodiscard]] int coarse_index() const noexcept {
        return coarse == None ? -1 : static_cast<int>(coarse);
    }

    [[nodiscard]] int fine_index() const noexcept {
        return fine == None ? -1 : static_cast<int>(fine);
    }
};

static_assert(sizeof(H32FeatureDelta) == 2 * sizeof(std::uint16_t));

[[nodiscard]] inline H32Perspective h32_perspective(
    const Position& pos,
    Color perspective
) noexcept {
    static constexpr auto transforms = [] {
        std::array<std::array<H32Perspective, SQ_NB>, COLOR_NB> table{};
        for (int persp = WHITE; persp <= BLACK; ++persp) {
            for (int square = 0; square < SQ_NB; ++square) {
                const int vertical_xor = persp == BLACK ? 56 : 0;
                const int relative = square ^ vertical_xor;
                const bool mirror = (relative & 7) >= 4;
                const int canonical = relative ^ (mirror ? 7 : 0);
                const int file = canonical & 7;
                const int rank = canonical >> 3;
                const int coarse_bucket = (rank / 2) * 4 + file;
                const int fine_bucket = rank * 4 + file;
                table[static_cast<std::size_t>(persp)]
                     [static_cast<std::size_t>(square)] = {
                    .coarse_base = static_cast<std::uint16_t>(
                        coarse_bucket * 10 * 64
                    ),
                    .fine_base = static_cast<std::uint16_t>(
                        fine_bucket * 11 * 64
                    ),
                    .coarse_bucket = static_cast<std::uint8_t>(coarse_bucket),
                    .fine_bucket = static_cast<std::uint8_t>(fine_bucket),
                    .square_xor = static_cast<std::uint8_t>(
                        vertical_xor ^ (mirror ? 7 : 0)
                    ),
                    .mirror = mirror
                };
            }
        }
        return table;
    }();
    return transforms[static_cast<std::size_t>(perspective)]
                     [static_cast<std::size_t>(king_square_of(pos, perspective))];
}

[[nodiscard]] inline int h32_phase4(const Position& pos) noexcept {
    static constexpr std::array<std::uint8_t, 25> buckets{
        3, 3, 3, 3, 3, 3,
        2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0
    };
    return buckets[static_cast<std::size_t>(
        std::clamp(mnue_phase_units(pos), 0, 24)
    )];
}

[[nodiscard]] inline int h32_output_bucket(const Position& pos) noexcept {
    const Color stm = static_cast<Color>(pos.side_to_move);
    const Square relative_king = relative_square(stm, king_square_of(pos, stm));
    const int file = file_of(relative_king);
    const int canonical_file = file < 4 ? file : 7 - file;
    const int coarse_bucket = (rank_of(relative_king) / 2) * 4 + canonical_file;
    return h32_phase4(pos) * P2H32Layout::CoarseInputBuckets
        + coarse_bucket;
}

[[nodiscard]] inline int h32_coarse_feature_index(
    Color perspective,
    const H32Perspective& transform,
    Piece pc,
    Square sq
) noexcept {
    const int pt = nonking_piece_index(type_of(pc));
    if (pt < 0)
        return -1;
    const int rel_color = color_of(pc) == perspective ? 0 : 1;
    const int canonical = static_cast<int>(sq) ^ transform.square_xor;
    return (((transform.coarse_bucket * 2 + rel_color) * 5 + pt) * 64)
        + canonical;
}

[[nodiscard]] inline int h32_fine_feature_index(
    Color perspective,
    const H32Perspective& transform,
    Piece pc,
    Square sq
) noexcept {
    if (pc == PIECE_NONE)
        return -1;
    const PieceType type = type_of(pc);
    const int rel_color = color_of(pc) == perspective ? 0 : 1;
    int plane = -1;
    if (type == KING) {
        if (rel_color == 0)
            return -1;
        plane = 10;
    } else {
        const int pt = nonking_piece_index(type);
        if (pt < 0)
            return -1;
        plane = rel_color * 5 + pt;
    }
    const int canonical = static_cast<int>(sq) ^ transform.square_xor;
    return ((transform.fine_bucket * 11 + plane) * 64) + canonical;
}

[[nodiscard]] inline H32FeatureDelta h32_feature_indices(
    Color perspective,
    const H32Perspective& transform,
    Piece pc,
    Square sq
) noexcept {
    if (pc == PIECE_NONE)
        return {};
    const PieceType type = type_of(pc);
    const int rel_color = color_of(pc) == perspective ? 0 : 1;
    const int canonical = static_cast<int>(sq) ^ transform.square_xor;
    if (type == KING) {
        return {
            .coarse = H32FeatureDelta::None,
            .fine = rel_color == 1
                ? static_cast<std::uint16_t>(
                    transform.fine_base + 10 * 64 + canonical
                )
                : H32FeatureDelta::None
        };
    }
    const int pt = nonking_piece_index(type);
    if (pt < 0)
        return {};
    return {
        .coarse = static_cast<std::uint16_t>(
            transform.coarse_base + (rel_color * 5 + pt) * 64 + canonical
        ),
        .fine = static_cast<std::uint16_t>(
            transform.fine_base + (rel_color * 5 + pt) * 64 + canonical
        )
    };
}

[[nodiscard]] inline int phase_bucket2(const Position& pos) noexcept {
    // Must match the Bullet trainer exactly:
    //   knight/bishop = 1, rook = 2, queen = 4, pawns ignored.
    //
    // Trainer:
    //   let mut npm = 0;
    //   N/B += 1, R += 2, Q += 4;
    //   phase = usize::from(npm < 10);
    //
    // Therefore:
    //   phase 0 = opening/middlegame
    //   phase 1 = simplified/endgame
    return mnue_phase_units(pos) < 10 ? 1 : 0;
}

[[nodiscard]] inline int game_ply(const Position& pos) noexcept {
    const int fullmove = std::max(1, pos.fullmove_number);
    return (fullmove - 1) * 2 + (pos.side_to_move == BLACK ? 1 : 0);
}

[[nodiscard]] inline int mnue_material_units(const Position& pos) noexcept {
    return std::clamp(non_king_material(pos), 0, kMnueMaterialMax);
}

[[nodiscard]] inline double mnue_material_ratio(const Position& pos) noexcept {
    return static_cast<double>(mnue_material_units(pos))
        / static_cast<double>(kMnueMaterialMax);
}

[[nodiscard]] inline double mnue_cp_denominator(const Position& pos) noexcept {
    const double material = mnue_material_ratio(pos);
    double denom = kMnueCpBase + kMnueCpMaterial * material;
    if (phase_bucket2(pos) != 0)
        denom -= kMnueCpEndgameDiscount;
    return std::clamp(
        denom,
        kMnueCpMinDenominator,
        kMnueCpMaxDenominator
    );
}

[[nodiscard]] inline int mnue_to_cp(int raw, const Position& pos) noexcept {
    const i64 abs_raw = raw >= 0 ? static_cast<i64>(raw) : -static_cast<i64>(raw);
    const double clipped = static_cast<double>(
        std::min<i64>(abs_raw, kMnueCpMaxRaw)
    );
    const int cp = static_cast<int>(
        std::round(clipped * 100.0 / mnue_cp_denominator(pos))
    );
    return raw >= 0 ? cp : -cp;
}

[[nodiscard]] inline int mnue_search_material_scale(
    const Position& pos
) noexcept {
    return kSearchMaterialScaleBase
        + mnue_material_units(pos) * kSearchMaterialScaleRange
            / kMnueMaterialMax;
}

[[nodiscard]] inline int mnue_search_score(
    int raw,
    const Position& pos
) noexcept {
    i64 scaled = static_cast<i64>(raw)
        * mnue_search_material_scale(pos)
        / kSearchMaterialScaleDenominator;

    const int fifty_move_clock = std::clamp(
        pos.halfmove_clock,
        0,
        kSearchFiftyMoveClockClamp
    );
    scaled = scaled
        * (kSearchFiftyMoveScaleBase - fifty_move_clock)
        / kSearchFiftyMoveScaleBase;

    return static_cast<int>(std::clamp<i64>(
        scaled,
        -kMnueCpMaxRaw,
        kMnueCpMaxRaw
    ));
}

[[nodiscard]] inline WinRateParams mnue_win_rate_params(const Position& pos) noexcept {
    const int material_units = mnue_material_units(pos);
    const double material = mnue_material_ratio(pos);
    const double ply =
        static_cast<double>(std::min(kWdlPlyClamp, game_ply(pos)))
        / kWdlPlyDivisor;
    const double low_material = 1.0 - material;

    double a = wdl_horner(kWdlA, ply);
    double b = wdl_horner(kWdlB, ply);

    if (material_units <= kWdlLowMaterialThreshold) {
        a += kWdlLowMaterialABonus * low_material;
        b += kWdlLowMaterialBBonus * low_material;
    }

    return {a, b};
}

[[nodiscard]] inline WdlTriplet mnue_wdl_from_cp(
    int cp,
    const Position& pos
) noexcept {
    const auto [a, b] = mnue_win_rate_params(pos);
    const double x = std::clamp(
        static_cast<double>(cp),
        -kWdlCpClamp,
        kWdlCpClamp
    );
    const double slope = std::max(1.0, b);

    const double win = 1.0 / (1.0 + std::exp((a - x) / slope));
    const double loss = 1.0 / (1.0 + std::exp((a + x) / slope));

    int win_i = std::clamp(static_cast<int>(std::round(1000.0 * win)), 0, 1000);
    int loss_i = std::clamp(static_cast<int>(std::round(1000.0 * loss)), 0, 1000);
    if (win_i + loss_i > 1000) {
        const int total = win_i + loss_i;
        win_i = (win_i * 1000) / total;
        loss_i = 1000 - win_i;
    }

    return {
        .win = win_i,
        .draw = 1000 - win_i - loss_i,
        .loss = loss_i
    };
}

template<class Layout>
[[nodiscard]] inline int king_bucket(Square relative_king_sq) noexcept {
    if constexpr (Layout::InputBuckets == 16) {
        return king_zone16(relative_king_sq);
    } else {
        static_assert(Layout::InputBuckets == 32);
        return king_zone32(relative_king_sq);
    }
}

template<class Layout>
[[nodiscard]] inline int input_bucket(const Position& pos, Color perspective) noexcept {
    const Square rel_ksq = relative_square(perspective, king_square_of(pos, perspective));
    return king_bucket<Layout>(rel_ksq);
}

template<class Layout>
[[nodiscard]] inline int output_bucket(const Position& pos, Color stm) noexcept {
    const Square rel_ksq = relative_square(stm, king_square_of(pos, stm));
    return phase_bucket2(pos) * Layout::InputBuckets + king_bucket<Layout>(rel_ksq);
}

template<class Layout>
[[nodiscard]] inline int feature_index(
    Color perspective,
    int bucket,
    Piece pc,
    Square sq
) noexcept {
    const PieceType pt = type_of(pc);
    const int pt_idx = nonking_piece_index(pt);
    if (pt_idx < 0)
        return -1;

    const int rel_color = color_of(pc) == perspective ? 0 : 1;
    const int rel_sq = static_cast<int>(relative_square(perspective, sq));

    return (((bucket * 2 + rel_color) * 5 + pt_idx) * 64 + rel_sq);
}

template<class Layout>
[[nodiscard]] inline int feature_index(
    const Position& pos,
    Color perspective,
    Piece pc,
    Square sq
) noexcept {
    return feature_index<Layout>(
        perspective,
        input_bucket<Layout>(pos, perspective),
        pc,
        sq
    );
}

template<class Layout>
using Accumulator = std::array<i16, Layout::HiddenSize>;

template<class Layout>
struct P2RefreshEntry {
    Accumulator<Layout> accumulation{};
    std::array<Piece, SQ_NB> board{};
    bool initialized = false;

    void reset_to_biases(const Network<Layout>& net) noexcept {
        std::copy(net.b0.begin(), net.b0.end(), accumulation.begin());
        board.fill(PIECE_NONE);
        initialized = true;
    }
};

template<class Layout>
struct P2RefreshCache {
    u32 generation = 0;
    std::array<std::array<P2RefreshEntry<Layout>, Layout::InputBuckets>, COLOR_NB> entries{};

    void sync(u32 current_generation) noexcept {
        if (generation == current_generation)
            return;

        for (auto& side_entries : entries)
            for (auto& entry : side_entries)
                entry.initialized = false;

        generation = current_generation;
    }
};

struct P2PerspectiveDiff {
    std::array<std::uint16_t, 2> removed{};
    std::array<std::uint16_t, 2> added{};
    std::uint8_t removed_count = 0;
    std::uint8_t added_count = 0;
    bool refresh = false;
};

#if defined(__AVX2__)
template<class Layout>
inline void add_feature_vector_avx2(
    std::array<i16, Layout::HiddenSize>& acc,
    const i16* w
) noexcept {
    i16* acc_ptr = acc.data();
    for (int i = 0; i < Layout::HiddenSize; i += 16) {
        const __m256i a = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc_ptr + i)
        );
        const __m256i b = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(w + i)
        );
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(acc_ptr + i),
            _mm256_add_epi16(a, b)
        );
    }
}

template<class Layout>
inline void sub_feature_vector_avx2(
    std::array<i16, Layout::HiddenSize>& acc,
    const i16* w
) noexcept {
    i16* acc_ptr = acc.data();
    for (int i = 0; i < Layout::HiddenSize; i += 16) {
        const __m256i a = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc_ptr + i)
        );
        const __m256i b = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(w + i)
        );
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(acc_ptr + i),
            _mm256_sub_epi16(a, b)
        );
    }
}

#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
template<class Layout>
inline void add_feature_vector_neon(
    std::array<i16, Layout::HiddenSize>& acc,
    const i16* weights
) noexcept {
    for (int i = 0; i < Layout::HiddenSize; i += 8) {
        const int16x8_t a = vld1q_s16(acc.data() + i);
        const int16x8_t b = vld1q_s16(weights + i);
        vst1q_s16(acc.data() + i, vaddq_s16(a, b));
    }
}

template<class Layout>
inline void sub_feature_vector_neon(
    std::array<i16, Layout::HiddenSize>& acc,
    const i16* weights
) noexcept {
    for (int i = 0; i < Layout::HiddenSize; i += 8) {
        const int16x8_t a = vld1q_s16(acc.data() + i);
        const int16x8_t b = vld1q_s16(weights + i);
        vst1q_s16(acc.data() + i, vsubq_s16(a, b));
    }
}
#endif

template<class Layout>
void add_feature(
    Accumulator<Layout>& acc,
    const Network<Layout>& net,
    int idx
) noexcept {
    const i16* w = net.w0.data() + static_cast<std::size_t>(idx) * Layout::HiddenSize;
#if defined(__AVX2__)
    add_feature_vector_avx2<Layout>(acc, w);
#elif defined(__aarch64__) && defined(__ARM_NEON)
    add_feature_vector_neon<Layout>(acc, w);
#else
    for (int i = 0; i < Layout::HiddenSize; ++i)
        acc[static_cast<std::size_t>(i)] = static_cast<i16>(
            static_cast<i32>(acc[static_cast<std::size_t>(i)]) + static_cast<i32>(w[i])
        );
#endif
}


template<class Layout>
void sub_feature(
    Accumulator<Layout>& acc,
    const Network<Layout>& net,
    int idx
) noexcept {
    const i16* w = net.w0.data() + static_cast<std::size_t>(idx) * Layout::HiddenSize;
#if defined(__AVX2__)
    sub_feature_vector_avx2<Layout>(acc, w);
#elif defined(__aarch64__) && defined(__ARM_NEON)
    sub_feature_vector_neon<Layout>(acc, w);
#else
    for (int i = 0; i < Layout::HiddenSize; ++i)
        acc[static_cast<std::size_t>(i)] = static_cast<i16>(
            static_cast<i32>(acc[static_cast<std::size_t>(i)]) - static_cast<i32>(w[i])
        );
#endif
}

template<class Layout>
void apply_p2_diff(
    Accumulator<Layout>& dst,
    const Accumulator<Layout>& src,
    const Network<Layout>& net,
    const P2PerspectiveDiff& diff,
    bool forward
) noexcept {
    const auto& added = forward ? diff.added : diff.removed;
    const auto& removed = forward ? diff.removed : diff.added;
    const std::uint8_t added_count = forward ? diff.added_count : diff.removed_count;
    const std::uint8_t removed_count = forward ? diff.removed_count : diff.added_count;

    const i16* add0 = added_count > 0
        ? net.w0.data() + static_cast<std::size_t>(added[0]) * Layout::HiddenSize
        : nullptr;
    const i16* add1 = added_count > 1
        ? net.w0.data() + static_cast<std::size_t>(added[1]) * Layout::HiddenSize
        : nullptr;
    const i16* sub0 = removed_count > 0
        ? net.w0.data() + static_cast<std::size_t>(removed[0]) * Layout::HiddenSize
        : nullptr;
    const i16* sub1 = removed_count > 1
        ? net.w0.data() + static_cast<std::size_t>(removed[1]) * Layout::HiddenSize
        : nullptr;

#if defined(__AVX2__)
    // P2 stack states explicitly align both perspective accumulators to a
    // cache line, so the source/destination traffic can use aligned moves.
    const auto apply = [&]<int AddCount, int SubCount>() noexcept {
        for (int i = 0; i < Layout::HiddenSize; i += 16) {
            __m256i value = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(src.data() + i)
            );
            if constexpr (AddCount >= 1)
                value = _mm256_add_epi16(
                    value,
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add0 + i))
                );
            if constexpr (AddCount >= 2)
                value = _mm256_add_epi16(
                    value,
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add1 + i))
                );
            if constexpr (SubCount >= 1)
                value = _mm256_sub_epi16(
                    value,
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sub0 + i))
                );
            if constexpr (SubCount >= 2)
                value = _mm256_sub_epi16(
                    value,
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sub1 + i))
                );
            _mm256_store_si256(reinterpret_cast<__m256i*>(dst.data() + i), value);
        }
    };

    switch ((static_cast<unsigned>(added_count) << 2) | removed_count) {
        case 0x0: apply.template operator()<0, 0>(); break;
        case 0x1: apply.template operator()<0, 1>(); break;
        case 0x2: apply.template operator()<0, 2>(); break;
        case 0x4: apply.template operator()<1, 0>(); break;
        case 0x5: apply.template operator()<1, 1>(); break;
        case 0x6: apply.template operator()<1, 2>(); break;
        case 0x8: apply.template operator()<2, 0>(); break;
        case 0x9: apply.template operator()<2, 1>(); break;
        case 0xA: apply.template operator()<2, 2>(); break;
        default:  assert(false); break;
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    for (int i = 0; i < Layout::HiddenSize; i += 8) {
        int16x8_t value = vld1q_s16(src.data() + i);
        if (add0 != nullptr)
            value = vaddq_s16(value, vld1q_s16(add0 + i));
        if (add1 != nullptr)
            value = vaddq_s16(value, vld1q_s16(add1 + i));
        if (sub0 != nullptr)
            value = vsubq_s16(value, vld1q_s16(sub0 + i));
        if (sub1 != nullptr)
            value = vsubq_s16(value, vld1q_s16(sub1 + i));
        vst1q_s16(dst.data() + i, value);
    }
#else
    for (int i = 0; i < Layout::HiddenSize; ++i) {
        i32 value = src[static_cast<std::size_t>(i)];
        if (add0 != nullptr) value += add0[i];
        if (add1 != nullptr) value += add1[i];
        if (sub0 != nullptr) value -= sub0[i];
        if (sub1 != nullptr) value -= sub1[i];
        dst[static_cast<std::size_t>(i)] = static_cast<i16>(value);
    }
#endif
}

using P2MoveDiff = std::array<P2PerspectiveDiff, COLOR_NB>;

template<class Layout>
void append_p2_feature(
    P2MoveDiff& move_diff,
    const std::array<int, COLOR_NB>& buckets,
    Piece pc,
    Square sq,
    bool added
) noexcept {
    if (pc == PIECE_NONE || type_of(pc) == KING)
        return;

    for (int persp = WHITE; persp <= BLACK; ++persp) {
        P2PerspectiveDiff& diff = move_diff[static_cast<std::size_t>(persp)];
        if (diff.refresh)
            continue;

        const int idx = feature_index<Layout>(
            static_cast<Color>(persp),
            buckets[static_cast<std::size_t>(persp)],
            pc,
            sq
        );
        if (idx < 0)
            continue;

        if (added) {
            assert(diff.added_count < diff.added.size());
            diff.added[diff.added_count++] = static_cast<std::uint16_t>(idx);
        } else {
            assert(diff.removed_count < diff.removed.size());
            diff.removed[diff.removed_count++] = static_cast<std::uint16_t>(idx);
        }
    }
}

template<class Layout>
[[nodiscard]] P2MoveDiff make_p2_move_diff(
    const Position& pos,
    Move move
) noexcept {
    P2MoveDiff move_diff{};
    const std::array<int, COLOR_NB> buckets{{
        input_bucket<Layout>(pos, WHITE),
        input_bucket<Layout>(pos, BLACK)
    }};
    const Square from = from_sq(move);
    const Square encoded_to = to_sq(move);
    const Piece moving = piece_on(pos, from);
    assert(moving != PIECE_NONE);

    const Color moving_color = color_of(moving);
    const PieceType moving_type = type_of(moving);
    const Square to = move_is_castle(move)
        ? castling_king_to(moving_color, move_flag(move))
        : encoded_to;
    if (moving_type == KING) {
        const int from_bucket = king_bucket<Layout>(relative_square(moving_color, from));
        const int to_bucket = king_bucket<Layout>(relative_square(moving_color, to));
        move_diff[static_cast<std::size_t>(moving_color)].refresh =
            from_bucket != to_bucket;
    }

    if (move_is_capture(move)) {
        const Square captured_sq = move_is_ep(move)
            ? static_cast<Square>(
                static_cast<int>(to) + (moving_color == WHITE ? -8 : 8)
            )
            : to;
        append_p2_feature<Layout>(
            move_diff,
            buckets,
            piece_on(pos, captured_sq),
            captured_sq,
            false
        );
    }

    append_p2_feature<Layout>(move_diff, buckets, moving, from, false);
    if (move_is_promotion(move)) {
        append_p2_feature<Layout>(
            move_diff,
            buckets,
            make_piece(moving_color, promo_piece(move)),
            to,
            true
        );
    } else {
        append_p2_feature<Layout>(move_diff, buckets, moving, to, true);
    }

    if (move_is_castle(move)) {
        const Square rook_from = encoded_to;
        const Square rook_to = castling_rook_to(moving_color, move_flag(move));
        const Piece rook = make_piece(moving_color, ROOK);
        append_p2_feature<Layout>(move_diff, buckets, rook, rook_from, false);
        append_p2_feature<Layout>(move_diff, buckets, rook, rook_to, true);
    }

    return move_diff;
}

template<class Layout>
void rebuild_accumulator(
    const Position& pos,
    Color perspective,
    Accumulator<Layout>& acc,
    const Network<Layout>& net
) noexcept {
    std::copy(net.b0.begin(), net.b0.end(), acc.begin());
    const int bucket = input_bucket<Layout>(pos, perspective);

    Bitboard bb = pieces(pos);
    while (bb) {
        const Square sq = static_cast<Square>(
            std::countr_zero(static_cast<std::uint64_t>(bb))
        );
        bb &= (bb - 1);

        const Piece pc = piece_on(pos, sq);
        if (pc == PIECE_NONE)
            continue;

        const int idx = feature_index<Layout>(perspective, bucket, pc, sq);
        if (idx >= 0)
            add_feature(acc, net, idx);
    }
}

using H32Accumulator = Accumulator<P2H32Layout>;

template<int Offset, int Width>
void h32_apply_segment(
    H32Accumulator& acc,
    const i16* weights,
    bool add
) noexcept {
#if defined(__AVX2__)
    static_assert(Width % 16 == 0);
    for (int i = 0; i < Width; i += 16) {
        i16* dst = acc.data() + Offset + i;
        const __m256i a = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(dst)
        );
        const __m256i w = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights + i)
        );
        const __m256i value = add
            ? _mm256_add_epi16(a, w)
            : _mm256_sub_epi16(a, w);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), value);
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    static_assert(Width % 8 == 0);
    for (int i = 0; i < Width; i += 8) {
        i16* dst = acc.data() + Offset + i;
        const int16x8_t a = vld1q_s16(dst);
        const int16x8_t w = vld1q_s16(weights + i);
        vst1q_s16(dst, add ? vaddq_s16(a, w) : vsubq_s16(a, w));
    }
#else
    for (int i = 0; i < Width; ++i) {
        const int index = Offset + i;
        const i32 value = static_cast<i32>(acc[static_cast<std::size_t>(index)])
            + (add ? static_cast<i32>(weights[i]) : -static_cast<i32>(weights[i]));
        acc[static_cast<std::size_t>(index)] = static_cast<i16>(value);
    }
#endif
}

void h32_apply_feature(
    H32Accumulator& acc,
    const H32Network& net,
    const H32FeatureDelta& feature,
    bool add
) noexcept {
    const int coarse = feature.coarse_index();
    const int fine = feature.fine_index();
    if (coarse >= 0) {
        const i16* weights = net.coarse_w.data()
            + static_cast<std::size_t>(coarse)
                * P2H32Layout::CoarseHiddenSize;
        h32_apply_segment<0, P2H32Layout::CoarseHiddenSize>(acc, weights, add);
    }
    if (fine >= 0) {
        const i16* weights = net.fine_w.data()
            + static_cast<std::size_t>(fine) * P2H32Layout::FineHiddenSize;
        h32_apply_segment<
            P2H32Layout::CoarseHiddenSize,
            P2H32Layout::FineHiddenSize
        >(acc, weights, add);
    }
}

void rebuild_h32_accumulator(
    const Position& pos,
    Color perspective,
    H32Accumulator& acc,
    const H32Network& net
) noexcept {
    std::copy(net.coarse_b.begin(), net.coarse_b.end(), acc.begin());
    std::copy(
        net.fine_b.begin(),
        net.fine_b.end(),
        acc.begin() + P2H32Layout::CoarseHiddenSize
    );
    const H32Perspective transform = h32_perspective(pos, perspective);
    Bitboard bb = pieces(pos);
    while (bb) {
        const Square sq = static_cast<Square>(
            std::countr_zero(static_cast<std::uint64_t>(bb))
        );
        bb &= bb - 1;
        const Piece pc = piece_on(pos, sq);
        const H32FeatureDelta feature = h32_feature_indices(
            perspective,
            transform,
            pc,
            sq
        );
        h32_apply_feature(acc, net, feature, true);
    }
}

struct H32PerspectiveDiff {
    std::array<H32FeatureDelta, 2> removed{};
    std::array<H32FeatureDelta, 2> added{};
    std::uint8_t removed_count = 0;
    std::uint8_t added_count = 0;
    bool refresh = false;
};

using H32MoveDiff = std::array<H32PerspectiveDiff, COLOR_NB>;

void append_h32_feature(
    H32MoveDiff& move_diff,
    const std::array<H32Perspective, COLOR_NB>& transforms,
    Piece pc,
    Square sq,
    bool added
) noexcept {
    if (pc == PIECE_NONE)
        return;
    for (int persp = WHITE; persp <= BLACK; ++persp) {
        H32PerspectiveDiff& diff = move_diff[static_cast<std::size_t>(persp)];
        if (diff.refresh)
            continue;
        const Color perspective = static_cast<Color>(persp);
        const H32FeatureDelta delta = h32_feature_indices(
            perspective,
            transforms[static_cast<std::size_t>(persp)],
            pc,
            sq
        );
        if (delta.coarse == H32FeatureDelta::None
            && delta.fine == H32FeatureDelta::None)
            continue;
        if (added) {
            assert(diff.added_count < diff.added.size());
            diff.added[diff.added_count++] = delta;
        } else {
            assert(diff.removed_count < diff.removed.size());
            diff.removed[diff.removed_count++] = delta;
        }
    }
}

[[nodiscard]] H32MoveDiff make_h32_move_diff(
    const Position& pos,
    Move move
) noexcept {
    H32MoveDiff move_diff{};
    const std::array<H32Perspective, COLOR_NB> transforms{{
        h32_perspective(pos, WHITE),
        h32_perspective(pos, BLACK)
    }};
    const Square from = from_sq(move);
    const Square encoded_to = to_sq(move);
    const Piece moving = piece_on(pos, from);
    assert(moving != PIECE_NONE);
    const Color moving_color = color_of(moving);
    const Square to = move_is_castle(move)
        ? castling_king_to(moving_color, move_flag(move))
        : encoded_to;

    if (type_of(moving) == KING)
        move_diff[static_cast<std::size_t>(moving_color)].refresh = true;

    if (move_is_capture(move)) {
        const Square captured_sq = move_is_ep(move)
            ? static_cast<Square>(
                static_cast<int>(to) + (moving_color == WHITE ? -8 : 8)
            )
            : to;
        append_h32_feature(
            move_diff,
            transforms,
            piece_on(pos, captured_sq),
            captured_sq,
            false
        );
    }

    append_h32_feature(move_diff, transforms, moving, from, false);
    const Piece placed = move_is_promotion(move)
        ? make_piece(moving_color, promo_piece(move))
        : moving;
    append_h32_feature(move_diff, transforms, placed, to, true);

    if (move_is_castle(move)) {
        const Square rook_from = encoded_to;
        const Square rook_to = castling_rook_to(moving_color, move_flag(move));
        const Piece rook = make_piece(moving_color, ROOK);
        append_h32_feature(move_diff, transforms, rook, rook_from, false);
        append_h32_feature(move_diff, transforms, rook, rook_to, true);
    }
    return move_diff;
}

template<bool Fine, int Offset, int Width, int AddCount, int SubCount>
void apply_h32_diff_segment(
    H32Accumulator& dst,
    const H32Accumulator& src,
    const H32Network& net,
    const std::array<H32FeatureDelta, 2>& added,
    const std::array<H32FeatureDelta, 2>& removed
) noexcept {
    const auto feature_index = [](const H32FeatureDelta& delta) noexcept {
        if constexpr (Fine)
            return delta.fine_index();
        else
            return delta.coarse_index();
    };
    const auto weight_ptr = [&](const H32FeatureDelta& delta) noexcept -> const i16* {
        const int feature = feature_index(delta);
        if (feature < 0)
            return nullptr;
        if constexpr (Fine) {
            return net.fine_w.data()
                + static_cast<std::size_t>(feature) * P2H32Layout::FineHiddenSize;
        } else {
            return net.coarse_w.data()
                + static_cast<std::size_t>(feature) * P2H32Layout::CoarseHiddenSize;
        }
    };
    const i16* add0 = AddCount >= 1 ? weight_ptr(added[0]) : nullptr;
    const i16* add1 = AddCount >= 2 ? weight_ptr(added[1]) : nullptr;
    const i16* sub0 = SubCount >= 1 ? weight_ptr(removed[0]) : nullptr;
    const i16* sub1 = SubCount >= 2 ? weight_ptr(removed[1]) : nullptr;

#if defined(__AVX2__)
    for (int i = 0; i < Width; i += 16) {
        __m256i value = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(src.data() + Offset + i)
        );
        if (add0 != nullptr)
            value = _mm256_add_epi16(
                value,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add0 + i))
            );
        if (add1 != nullptr)
            value = _mm256_add_epi16(
                value,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add1 + i))
            );
        if (sub0 != nullptr)
            value = _mm256_sub_epi16(
                value,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sub0 + i))
            );
        if (sub1 != nullptr)
            value = _mm256_sub_epi16(
                value,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sub1 + i))
            );
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(dst.data() + Offset + i),
            value
        );
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    static_assert(Width % 8 == 0);
    for (int i = 0; i < Width; i += 8) {
        int16x8_t value = vld1q_s16(src.data() + Offset + i);
        if (add0 != nullptr)
            value = vaddq_s16(value, vld1q_s16(add0 + i));
        if (add1 != nullptr)
            value = vaddq_s16(value, vld1q_s16(add1 + i));
        if (sub0 != nullptr)
            value = vsubq_s16(value, vld1q_s16(sub0 + i));
        if (sub1 != nullptr)
            value = vsubq_s16(value, vld1q_s16(sub1 + i));
        vst1q_s16(dst.data() + Offset + i, value);
    }
#else
    for (int i = 0; i < Width; ++i) {
        i32 value = src[static_cast<std::size_t>(Offset + i)];
        if (add0 != nullptr) value += add0[i];
        if (add1 != nullptr) value += add1[i];
        if (sub0 != nullptr) value -= sub0[i];
        if (sub1 != nullptr) value -= sub1[i];
        dst[static_cast<std::size_t>(Offset + i)] = static_cast<i16>(value);
    }
#endif
}

template<int AddCount, int SubCount>
void apply_h32_diff_counts(
    H32Accumulator& dst,
    const H32Accumulator& src,
    const H32Network& net,
    const std::array<H32FeatureDelta, 2>& added,
    const std::array<H32FeatureDelta, 2>& removed
) noexcept {
    apply_h32_diff_segment<
        false,
        0,
        P2H32Layout::CoarseHiddenSize,
        AddCount,
        SubCount
    >(dst, src, net, added, removed);
    apply_h32_diff_segment<
        true,
        P2H32Layout::CoarseHiddenSize,
        P2H32Layout::FineHiddenSize,
        AddCount,
        SubCount
    >(dst, src, net, added, removed);
}

void apply_h32_diff(
    H32Accumulator& dst,
    const H32Accumulator& src,
    const H32Network& net,
    const H32PerspectiveDiff& diff,
    bool forward
) noexcept {
    const auto& added = forward ? diff.added : diff.removed;
    const auto& removed = forward ? diff.removed : diff.added;
    const auto added_count = forward ? diff.added_count : diff.removed_count;
    const auto removed_count = forward ? diff.removed_count : diff.added_count;
    switch ((static_cast<unsigned>(added_count) << 2) | removed_count) {
        case 0x0: apply_h32_diff_counts<0, 0>(dst, src, net, added, removed); break;
        case 0x1: apply_h32_diff_counts<0, 1>(dst, src, net, added, removed); break;
        case 0x2: apply_h32_diff_counts<0, 2>(dst, src, net, added, removed); break;
        case 0x4: apply_h32_diff_counts<1, 0>(dst, src, net, added, removed); break;
        case 0x5: apply_h32_diff_counts<1, 1>(dst, src, net, added, removed); break;
        case 0x6: apply_h32_diff_counts<1, 2>(dst, src, net, added, removed); break;
        case 0x8: apply_h32_diff_counts<2, 0>(dst, src, net, added, removed); break;
        case 0x9: apply_h32_diff_counts<2, 1>(dst, src, net, added, removed); break;
        case 0xA: apply_h32_diff_counts<2, 2>(dst, src, net, added, removed); break;
        default: assert(false); break;
    }
}

struct H32RefreshEntry {
    H32Accumulator accumulation{};
    std::array<Piece, SQ_NB> canonical_board{};
    bool initialized = false;
};

struct H32RefreshCache {
    u32 generation = 0;
    std::array<std::array<H32RefreshEntry, P2H32Layout::FineInputBuckets>, COLOR_NB>
        entries{};

    void sync(u32 current_generation) noexcept {
        if (generation == current_generation)
            return;
        for (auto& side : entries)
            for (auto& entry : side)
                entry.initialized = false;
        generation = current_generation;
    }
};

[[nodiscard]] H32FeatureDelta h32_canonical_feature_indices(
    Color perspective,
    const H32Perspective& transform,
    Piece pc,
    int canonical_sq
) noexcept {
    if (pc == PIECE_NONE)
        return {};
    const int rel_color = color_of(pc) == perspective ? 0 : 1;
    if (type_of(pc) == KING) {
        if (rel_color == 0)
            return {};
        return {
            .coarse = H32FeatureDelta::None,
            .fine = static_cast<std::uint16_t>(
                transform.fine_base + 10 * 64 + canonical_sq
            )
        };
    }
    const int pt = nonking_piece_index(type_of(pc));
    if (pt < 0)
        return {};
    return {
        .coarse = static_cast<std::uint16_t>(
            transform.coarse_base + (rel_color * 5 + pt) * 64 + canonical_sq
        ),
        .fine = static_cast<std::uint16_t>(
            transform.fine_base + (rel_color * 5 + pt) * 64 + canonical_sq
        )
    };
}

void refresh_h32_accumulator_from_cache(
    H32RefreshCache& cache,
    const H32Network& net,
    const Position& pos,
    Color perspective,
    u32 current_generation,
    H32Accumulator& out
) noexcept {
    cache.sync(current_generation);
    const H32Perspective transform = h32_perspective(pos, perspective);
    H32RefreshEntry& entry =
        cache.entries[static_cast<std::size_t>(perspective)]
                     [static_cast<std::size_t>(transform.fine_bucket)];
    if (!entry.initialized) {
        std::copy(net.coarse_b.begin(), net.coarse_b.end(), entry.accumulation.begin());
        std::copy(
            net.fine_b.begin(),
            net.fine_b.end(),
            entry.accumulation.begin() + P2H32Layout::CoarseHiddenSize
        );
        entry.canonical_board.fill(PIECE_NONE);
        entry.initialized = true;
    }

    std::array<Piece, SQ_NB> current{};
    current.fill(PIECE_NONE);
    Bitboard bb = pieces(pos);
    while (bb) {
        const Square sq = static_cast<Square>(
            std::countr_zero(static_cast<std::uint64_t>(bb))
        );
        bb &= bb - 1;
        const int canonical = static_cast<int>(sq) ^ transform.square_xor;
        current[static_cast<std::size_t>(canonical)] = piece_on(pos, sq);
    }

    for (int sq = 0; sq < SQ_NB; ++sq) {
        const Piece old_pc = entry.canonical_board[static_cast<std::size_t>(sq)];
        const Piece new_pc = current[static_cast<std::size_t>(sq)];
        if (old_pc == new_pc)
            continue;
        h32_apply_feature(
            entry.accumulation,
            net,
            h32_canonical_feature_indices(perspective, transform, old_pc, sq),
            false
        );
        h32_apply_feature(
            entry.accumulation,
            net,
            h32_canonical_feature_indices(perspective, transform, new_pc, sq),
            true
        );
        entry.canonical_board[static_cast<std::size_t>(sq)] = new_pc;
    }
    out = entry.accumulation;
}

[[nodiscard]] inline i32 screlu(i32 x) noexcept {
    const i32 y = std::clamp(x, 0, kClip);
    return y * y;
}


#if defined(__AVX512F__) && defined(__AVX512BW__)
[[nodiscard]] inline i64 horizontal_sum_epi64_avx512(__m512i v) noexcept {
    return static_cast<i64>(_mm512_reduce_add_epi64(v));
}

[[nodiscard]] inline i64 horizontal_sum_epi32_to_i64_avx512(__m512i v) noexcept {
    const __m256i lo = _mm512_castsi512_si256(v);
    const __m256i hi = _mm512_extracti64x4_epi64(v, 1);
    return horizontal_sum_epi64_avx512(
        _mm512_add_epi64(
            _mm512_cvtepi32_epi64(lo),
            _mm512_cvtepi32_epi64(hi)
        )
    );
}

inline __m512i add_i32x16_to_i64_avx512(__m512i sum, __m512i v) noexcept {
    const __m256i lo = _mm512_castsi512_si256(v);
    const __m256i hi = _mm512_extracti64x4_epi64(v, 1);
    sum = _mm512_add_epi64(sum, _mm512_cvtepi32_epi64(lo));
    return _mm512_add_epi64(sum, _mm512_cvtepi32_epi64(hi));
}

[[nodiscard]] i64 dot_pair_screlu_i16_madd_avx512(
    const i16* acc0,
    const i16* weights0,
    const i16* acc1,
    const i16* weights1,
    int count
) noexcept {
    const __m512i zero16 = _mm512_setzero_si512();
    const __m512i clip16 = _mm512_set1_epi16(static_cast<i16>(kClip));
    __m512i sum0_32 = _mm512_setzero_si512();
    __m512i sum1_32 = _mm512_setzero_si512();

    int i = 0;
    for (; i + 31 < count; i += 32) {
        const __m512i acc0_16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(acc0 + i)
        );
        const __m512i w0_16 = _mm512_loadu_si512(
            reinterpret_cast<const void*>(weights0 + i)
        );
        const __m512i acc1_16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(acc1 + i)
        );
        const __m512i w1_16 = _mm512_loadu_si512(
            reinterpret_cast<const void*>(weights1 + i)
        );

        const __m512i clipped0_16 = _mm512_min_epi16(
            _mm512_max_epi16(acc0_16_raw, zero16),
            clip16
        );
        const __m512i clipped1_16 = _mm512_min_epi16(
            _mm512_max_epi16(acc1_16_raw, zero16),
            clip16
        );

        const __m512i aw0_16 = _mm512_mullo_epi16(clipped0_16, w0_16);
        const __m512i aw1_16 = _mm512_mullo_epi16(clipped1_16, w1_16);
        sum0_32 = _mm512_add_epi32(
            sum0_32,
            _mm512_madd_epi16(clipped0_16, aw0_16)
        );
        sum1_32 = _mm512_add_epi32(
            sum1_32,
            _mm512_madd_epi16(clipped1_16, aw1_16)
        );
    }

    i64 total = horizontal_sum_epi32_to_i64_avx512(sum0_32)
        + horizontal_sum_epi32_to_i64_avx512(sum1_32);
    for (; i < count; ++i) {
        total += static_cast<i64>(screlu(acc0[i])) * static_cast<i64>(weights0[i]);
        total += static_cast<i64>(screlu(acc1[i])) * static_cast<i64>(weights1[i]);
    }

    return total;
}

[[nodiscard]] i64 dot_pair_screlu_i16_i32_avx512(
    const i16* acc0,
    const i16* weights0,
    const i16* acc1,
    const i16* weights1,
    int count
) noexcept {
    const __m512i zero16 = _mm512_setzero_si512();
    const __m512i clip16 = _mm512_set1_epi16(static_cast<i16>(kClip));
    __m512i sum0_32 = _mm512_setzero_si512();
    __m512i sum1_32 = _mm512_setzero_si512();

    int i = 0;
    for (; i + 31 < count; i += 32) {
        const __m512i acc0_16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(acc0 + i)
        );
        const __m512i w0_16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(weights0 + i)
        );
        const __m512i acc1_16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(acc1 + i)
        );
        const __m512i w1_16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(weights1 + i)
        );

        const __m512i clipped0_16 = _mm512_min_epi16(
            _mm512_max_epi16(acc0_16_raw, zero16),
            clip16
        );
        const __m512i clipped1_16 = _mm512_min_epi16(
            _mm512_max_epi16(acc1_16_raw, zero16),
            clip16
        );

        const __m256i acc0_lo16 = _mm512_castsi512_si256(clipped0_16);
        const __m256i acc0_hi16 = _mm512_extracti64x4_epi64(clipped0_16, 1);
        const __m256i w0_lo16 = _mm512_castsi512_si256(w0_16_raw);
        const __m256i w0_hi16 = _mm512_extracti64x4_epi64(w0_16_raw, 1);
        const __m256i acc1_lo16 = _mm512_castsi512_si256(clipped1_16);
        const __m256i acc1_hi16 = _mm512_extracti64x4_epi64(clipped1_16, 1);
        const __m256i w1_lo16 = _mm512_castsi512_si256(w1_16_raw);
        const __m256i w1_hi16 = _mm512_extracti64x4_epi64(w1_16_raw, 1);

        const __m512i acc0_lo32 = _mm512_cvtepi16_epi32(acc0_lo16);
        const __m512i acc0_hi32 = _mm512_cvtepi16_epi32(acc0_hi16);
        const __m512i w0_lo32 = _mm512_cvtepi16_epi32(w0_lo16);
        const __m512i w0_hi32 = _mm512_cvtepi16_epi32(w0_hi16);
        const __m512i acc1_lo32 = _mm512_cvtepi16_epi32(acc1_lo16);
        const __m512i acc1_hi32 = _mm512_cvtepi16_epi32(acc1_hi16);
        const __m512i w1_lo32 = _mm512_cvtepi16_epi32(w1_lo16);
        const __m512i w1_hi32 = _mm512_cvtepi16_epi32(w1_hi16);

        const __m512i sq0_lo32 = _mm512_mullo_epi32(acc0_lo32, acc0_lo32);
        const __m512i sq0_hi32 = _mm512_mullo_epi32(acc0_hi32, acc0_hi32);
        const __m512i sq1_lo32 = _mm512_mullo_epi32(acc1_lo32, acc1_lo32);
        const __m512i sq1_hi32 = _mm512_mullo_epi32(acc1_hi32, acc1_hi32);

        sum0_32 = _mm512_add_epi32(
            sum0_32,
            _mm512_mullo_epi32(sq0_lo32, w0_lo32)
        );
        sum0_32 = _mm512_add_epi32(
            sum0_32,
            _mm512_mullo_epi32(sq0_hi32, w0_hi32)
        );
        sum1_32 = _mm512_add_epi32(
            sum1_32,
            _mm512_mullo_epi32(sq1_lo32, w1_lo32)
        );
        sum1_32 = _mm512_add_epi32(
            sum1_32,
            _mm512_mullo_epi32(sq1_hi32, w1_hi32)
        );
    }

    i64 total = horizontal_sum_epi32_to_i64_avx512(sum0_32)
        + horizontal_sum_epi32_to_i64_avx512(sum1_32);
    for (; i < count; ++i) {
        total += static_cast<i64>(screlu(acc0[i])) * static_cast<i64>(weights0[i]);
        total += static_cast<i64>(screlu(acc1[i])) * static_cast<i64>(weights1[i]);
    }

    return total;
}

[[nodiscard]] i64 dot_screlu_i16_avx512(
    const i16* acc,
    const i16* weights,
    int count
) noexcept {
    const __m512i zero16 = _mm512_setzero_si512();
    const __m512i clip16 = _mm512_set1_epi16(static_cast<i16>(kClip));
    __m512i sum64 = _mm512_setzero_si512();

    int i = 0;
    for (; i + 31 < count; i += 32) {
        const __m512i acc16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(acc + i)
        );
        const __m512i w16_raw = _mm512_loadu_si512(
            reinterpret_cast<const void*>(weights + i)
        );
        const __m512i clipped16 = _mm512_min_epi16(
            _mm512_max_epi16(acc16_raw, zero16),
            clip16
        );

        const __m256i acc_lo16 = _mm512_castsi512_si256(clipped16);
        const __m256i acc_hi16 = _mm512_extracti64x4_epi64(clipped16, 1);
        const __m256i w_lo16 = _mm512_castsi512_si256(w16_raw);
        const __m256i w_hi16 = _mm512_extracti64x4_epi64(w16_raw, 1);

        const __m512i acc_lo32 = _mm512_cvtepi16_epi32(acc_lo16);
        const __m512i acc_hi32 = _mm512_cvtepi16_epi32(acc_hi16);
        const __m512i w_lo32 = _mm512_cvtepi16_epi32(w_lo16);
        const __m512i w_hi32 = _mm512_cvtepi16_epi32(w_hi16);
        const __m512i prod_lo32 = _mm512_mullo_epi32(
            _mm512_mullo_epi32(acc_lo32, acc_lo32),
            w_lo32
        );
        const __m512i prod_hi32 = _mm512_mullo_epi32(
            _mm512_mullo_epi32(acc_hi32, acc_hi32),
            w_hi32
        );

        sum64 = add_i32x16_to_i64_avx512(sum64, prod_lo32);
        sum64 = add_i32x16_to_i64_avx512(sum64, prod_hi32);
    }

    i64 total = horizontal_sum_epi64_avx512(sum64);
    for (; i < count; ++i)
        total += static_cast<i64>(screlu(acc[i])) * static_cast<i64>(weights[i]);

    return total;
}
#endif

#if defined(__AVX2__) \
    && !(defined(__AVX512F__) && defined(__AVX512BW__))
[[nodiscard]] inline i64 horizontal_sum_epi64_avx2(__m256i v) noexcept {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    const __m128i sum = _mm_add_epi64(lo, hi);
    return static_cast<i64>(_mm_cvtsi128_si64(sum))
        + static_cast<i64>(_mm_extract_epi64(sum, 1));
}

[[nodiscard]] inline i64 horizontal_sum_epi32_to_i64_avx2(__m256i v) noexcept {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    const __m256i lo64 = _mm256_cvtepi32_epi64(lo);
    const __m256i hi64 = _mm256_cvtepi32_epi64(hi);
    return horizontal_sum_epi64_avx2(_mm256_add_epi64(lo64, hi64));
}

[[nodiscard]] inline __m256i cvt_i32x4_to_i64x4_avx2(__m128i v) noexcept {
    return _mm256_cvtepi32_epi64(v);
}

#if defined(__AVXVNNI__) \
    && !(defined(__AVX512F__) && defined(__AVX512BW__))
inline void screlu_square_bytes_avx2(
    __m256i raw0,
    __m256i raw1,
    __m256i& lo8,
    __m256i& hi8
) noexcept {
    const __m256i zero16 = _mm256_setzero_si256();
    const __m256i clip16 = _mm256_set1_epi16(static_cast<i16>(kClip));
    const __m256i low_byte_mask = _mm256_set1_epi16(static_cast<i16>(0x00FF));

    const __m256i clipped0 = _mm256_min_epi16(
        _mm256_max_epi16(raw0, zero16),
        clip16
    );
    const __m256i clipped1 = _mm256_min_epi16(
        _mm256_max_epi16(raw1, zero16),
        clip16
    );

    const __m256i sq0 = _mm256_mullo_epi16(clipped0, clipped0);
    const __m256i sq1 = _mm256_mullo_epi16(clipped1, clipped1);

    lo8 = _mm256_packus_epi16(
        _mm256_and_si256(sq0, low_byte_mask),
        _mm256_and_si256(sq1, low_byte_mask)
    );
    hi8 = _mm256_packus_epi16(
        _mm256_srli_epi16(sq0, 8),
        _mm256_srli_epi16(sq1, 8)
    );

    lo8 = _mm256_permute4x64_epi64(lo8, 0xD8);
    hi8 = _mm256_permute4x64_epi64(hi8, 0xD8);
}

[[nodiscard]] i64 dot_pair_screlu_i8_vnni_avx2(
    const i16* acc0,
    const std::int8_t* weights0,
    const i16* acc1,
    const std::int8_t* weights1,
    int count
) noexcept {
    __m256i sum_lo_32 = _mm256_setzero_si256();
    __m256i sum_hi_32 = _mm256_setzero_si256();

    int i = 0;
    for (; i + 31 < count; i += 32) {
        const __m256i acc0_raw0 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc0 + i)
        );
        const __m256i acc0_raw1 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc0 + i + 16)
        );
        const __m256i acc1_raw0 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc1 + i)
        );
        const __m256i acc1_raw1 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc1 + i + 16)
        );

        __m256i lo8 = _mm256_setzero_si256();
        __m256i hi8 = _mm256_setzero_si256();
        screlu_square_bytes_avx2(acc0_raw0, acc0_raw1, lo8, hi8);

        const __m256i w0 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights0 + i)
        );
        sum_lo_32 = _mm256_dpbusd_epi32(sum_lo_32, lo8, w0);
        sum_hi_32 = _mm256_dpbusd_epi32(sum_hi_32, hi8, w0);

        screlu_square_bytes_avx2(acc1_raw0, acc1_raw1, lo8, hi8);

        const __m256i w1 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights1 + i)
        );
        sum_lo_32 = _mm256_dpbusd_epi32(sum_lo_32, lo8, w1);
        sum_hi_32 = _mm256_dpbusd_epi32(sum_hi_32, hi8, w1);
    }

    i64 total = horizontal_sum_epi32_to_i64_avx2(sum_lo_32)
        + 256LL * horizontal_sum_epi32_to_i64_avx2(sum_hi_32);
    for (; i < count; ++i) {
        total += static_cast<i64>(screlu(acc0[i]))
            * static_cast<i64>(weights0[i]);
        total += static_cast<i64>(screlu(acc1[i]))
            * static_cast<i64>(weights1[i]);
    }

    return total;
}
#endif

[[nodiscard]] i64 dot_pair_screlu_i16_madd_avx2(
    const i16* acc0,
    const i16* weights0,
    const i16* acc1,
    const i16* weights1,
    int count
) noexcept {
    const __m256i zero16 = _mm256_setzero_si256();
    const __m256i clip16 = _mm256_set1_epi16(static_cast<i16>(kClip));
    __m256i sum0_32 = _mm256_setzero_si256();
    __m256i sum1_32 = _mm256_setzero_si256();

    int i = 0;
    for (; i + 15 < count; i += 16) {
        const __m256i acc0_16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc0 + i)
        );
        const __m256i w0_16 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights0 + i)
        );
        const __m256i acc1_16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc1 + i)
        );
        const __m256i w1_16 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights1 + i)
        );

        const __m256i clipped0_16 = _mm256_min_epi16(
            _mm256_max_epi16(acc0_16_raw, zero16),
            clip16
        );
        const __m256i clipped1_16 = _mm256_min_epi16(
            _mm256_max_epi16(acc1_16_raw, zero16),
            clip16
        );

        const __m256i aw0_16 = _mm256_mullo_epi16(clipped0_16, w0_16);
        const __m256i aw1_16 = _mm256_mullo_epi16(clipped1_16, w1_16);

        sum0_32 = _mm256_add_epi32(
            sum0_32,
            _mm256_madd_epi16(clipped0_16, aw0_16)
        );
        sum1_32 = _mm256_add_epi32(
            sum1_32,
            _mm256_madd_epi16(clipped1_16, aw1_16)
        );
    }

    i64 total = horizontal_sum_epi32_to_i64_avx2(sum0_32)
        + horizontal_sum_epi32_to_i64_avx2(sum1_32);
    for (; i < count; ++i) {
        total += static_cast<i64>(screlu(acc0[i])) * static_cast<i64>(weights0[i]);
        total += static_cast<i64>(screlu(acc1[i])) * static_cast<i64>(weights1[i]);
    }

    return total;
}

[[nodiscard]] i64 dot_pair_screlu_i16_i32_avx2(
    const i16* acc0,
    const i16* weights0,
    const i16* acc1,
    const i16* weights1,
    int count
) noexcept {
    const __m256i zero16 = _mm256_setzero_si256();
    const __m256i clip16 = _mm256_set1_epi16(static_cast<i16>(kClip));
    __m256i sum0_32 = _mm256_setzero_si256();
    __m256i sum1_32 = _mm256_setzero_si256();

    int i = 0;
    for (; i + 15 < count; i += 16) {
        const __m256i acc0_16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc0 + i)
        );
        const __m256i w0_16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights0 + i)
        );
        const __m256i acc1_16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc1 + i)
        );
        const __m256i w1_16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights1 + i)
        );

        const __m256i clipped0_16 = _mm256_min_epi16(
            _mm256_max_epi16(acc0_16_raw, zero16),
            clip16
        );
        const __m256i clipped1_16 = _mm256_min_epi16(
            _mm256_max_epi16(acc1_16_raw, zero16),
            clip16
        );

        const __m128i acc0_lo16 = _mm256_castsi256_si128(clipped0_16);
        const __m128i acc0_hi16 = _mm256_extracti128_si256(clipped0_16, 1);
        const __m128i w0_lo16 = _mm256_castsi256_si128(w0_16_raw);
        const __m128i w0_hi16 = _mm256_extracti128_si256(w0_16_raw, 1);
        const __m128i acc1_lo16 = _mm256_castsi256_si128(clipped1_16);
        const __m128i acc1_hi16 = _mm256_extracti128_si256(clipped1_16, 1);
        const __m128i w1_lo16 = _mm256_castsi256_si128(w1_16_raw);
        const __m128i w1_hi16 = _mm256_extracti128_si256(w1_16_raw, 1);

        const __m256i acc0_lo32 = _mm256_cvtepi16_epi32(acc0_lo16);
        const __m256i acc0_hi32 = _mm256_cvtepi16_epi32(acc0_hi16);
        const __m256i w0_lo32 = _mm256_cvtepi16_epi32(w0_lo16);
        const __m256i w0_hi32 = _mm256_cvtepi16_epi32(w0_hi16);
        const __m256i acc1_lo32 = _mm256_cvtepi16_epi32(acc1_lo16);
        const __m256i acc1_hi32 = _mm256_cvtepi16_epi32(acc1_hi16);
        const __m256i w1_lo32 = _mm256_cvtepi16_epi32(w1_lo16);
        const __m256i w1_hi32 = _mm256_cvtepi16_epi32(w1_hi16);

        const __m256i sq0_lo32 = _mm256_mullo_epi32(acc0_lo32, acc0_lo32);
        const __m256i sq0_hi32 = _mm256_mullo_epi32(acc0_hi32, acc0_hi32);
        const __m256i sq1_lo32 = _mm256_mullo_epi32(acc1_lo32, acc1_lo32);
        const __m256i sq1_hi32 = _mm256_mullo_epi32(acc1_hi32, acc1_hi32);

        sum0_32 = _mm256_add_epi32(
            sum0_32,
            _mm256_mullo_epi32(sq0_lo32, w0_lo32)
        );
        sum0_32 = _mm256_add_epi32(
            sum0_32,
            _mm256_mullo_epi32(sq0_hi32, w0_hi32)
        );
        sum1_32 = _mm256_add_epi32(
            sum1_32,
            _mm256_mullo_epi32(sq1_lo32, w1_lo32)
        );
        sum1_32 = _mm256_add_epi32(
            sum1_32,
            _mm256_mullo_epi32(sq1_hi32, w1_hi32)
        );
    }

    i64 total = horizontal_sum_epi32_to_i64_avx2(sum0_32)
        + horizontal_sum_epi32_to_i64_avx2(sum1_32);
    for (; i < count; ++i) {
        total += static_cast<i64>(screlu(acc0[i])) * static_cast<i64>(weights0[i]);
        total += static_cast<i64>(screlu(acc1[i])) * static_cast<i64>(weights1[i]);
    }

    return total;
}

[[nodiscard]] i64 dot_screlu_i16_avx2(
    const i16* acc,
    const i16* weights,
    int count
) noexcept {
    const __m256i zero16 = _mm256_setzero_si256();
    const __m256i clip16 = _mm256_set1_epi16(static_cast<i16>(kClip));
    __m256i sum64 = _mm256_setzero_si256();

    int i = 0;
    for (; i + 15 < count; i += 16) {
        const __m256i acc16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(acc + i)
        );
        const __m256i w16_raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights + i)
        );

        const __m256i clipped16 = _mm256_min_epi16(
            _mm256_max_epi16(acc16_raw, zero16),
            clip16
        );

        const __m128i acc_lo16 = _mm256_castsi256_si128(clipped16);
        const __m128i acc_hi16 = _mm256_extracti128_si256(clipped16, 1);
        const __m128i w_lo16 = _mm256_castsi256_si128(w16_raw);
        const __m128i w_hi16 = _mm256_extracti128_si256(w16_raw, 1);

        const __m256i acc_lo32 = _mm256_cvtepi16_epi32(acc_lo16);
        const __m256i acc_hi32 = _mm256_cvtepi16_epi32(acc_hi16);
        const __m256i w_lo32 = _mm256_cvtepi16_epi32(w_lo16);
        const __m256i w_hi32 = _mm256_cvtepi16_epi32(w_hi16);

        const __m256i sq_lo32 = _mm256_mullo_epi32(acc_lo32, acc_lo32);
        const __m256i sq_hi32 = _mm256_mullo_epi32(acc_hi32, acc_hi32);
        const __m256i prod_lo32 = _mm256_mullo_epi32(sq_lo32, w_lo32);
        const __m256i prod_hi32 = _mm256_mullo_epi32(sq_hi32, w_hi32);

        sum64 = _mm256_add_epi64(
            sum64,
            cvt_i32x4_to_i64x4_avx2(_mm256_castsi256_si128(prod_lo32))
        );
        sum64 = _mm256_add_epi64(
            sum64,
            cvt_i32x4_to_i64x4_avx2(_mm256_extracti128_si256(prod_lo32, 1))
        );
        sum64 = _mm256_add_epi64(
            sum64,
            cvt_i32x4_to_i64x4_avx2(_mm256_castsi256_si128(prod_hi32))
        );
        sum64 = _mm256_add_epi64(
            sum64,
            cvt_i32x4_to_i64x4_avx2(_mm256_extracti128_si256(prod_hi32, 1))
        );
    }

    i64 total = horizontal_sum_epi64_avx2(sum64);
    for (; i < count; ++i)
        total += static_cast<i64>(screlu(acc[i])) * static_cast<i64>(weights[i]);

    return total;
}

#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
[[nodiscard]] inline i64 horizontal_sum_i64x4_neon(int64x2_t low, int64x2_t high) noexcept {
    return vaddvq_s64(vaddq_s64(low, high));
}

// Exact signed 16-bit output-weight path. Activations are clipped to [0,255],
// squared in unsigned 16-bit lanes, then widened before multiplication so no
// intermediate 16-bit or 32-bit overflow can change the NNUE result.
[[nodiscard]] i64 dot_screlu_i16_neon(
    const i16* acc,
    const i16* weights,
    int count
) noexcept {
    const int16x4_t zero = vdup_n_s16(0);
    const int16x4_t clip = vdup_n_s16(static_cast<i16>(kClip));
    int64x2_t sum_low = vdupq_n_s64(0);
    int64x2_t sum_high = vdupq_n_s64(0);

    for (int i = 0; i < count; i += 4) {
        const int16x4_t raw = vld1_s16(acc + i);
        const int16x4_t clipped = vmin_s16(vmax_s16(raw, zero), clip);
        const uint16x4_t clipped_u = vreinterpret_u16_s16(clipped);
        const uint16x4_t square = vmul_u16(clipped_u, clipped_u);
        const int32x4_t square32 = vreinterpretq_s32_u32(vmovl_u16(square));
        const int32x4_t weight32 = vmovl_s16(vld1_s16(weights + i));
        const int32x4_t product = vmulq_s32(square32, weight32);

        sum_low = vaddq_s64(sum_low, vmovl_s32(vget_low_s32(product)));
        sum_high = vaddq_s64(sum_high, vmovl_s32(vget_high_s32(product)));
    }

    return horizontal_sum_i64x4_neon(sum_low, sum_high);
}

[[nodiscard]] i64 dot_pair_screlu_i16_neon(
    const i16* acc0,
    const i16* weights0,
    const i16* acc1,
    const i16* weights1,
    int count
) noexcept {
    return dot_screlu_i16_neon(acc0, weights0, count)
        + dot_screlu_i16_neon(acc1, weights1, count);
}
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
// Dot-product path for networks whose complete output matrix is signed-i8 safe.
// Each squared activation is uint16. Split it into low/high bytes, center each
// byte by 128 to make it signed, and restore the offset using a dot of ones.
// This is exact: x = (lo - 128 + 128) + 256 * (hi - 128 + 128).
[[nodiscard]] i64 dot_screlu_i8_dotprod(
    const i16* acc,
    const std::int8_t* weights,
    int count
) noexcept {
    const int16x8_t zero16 = vdupq_n_s16(0);
    const int16x8_t clip16 = vdupq_n_s16(static_cast<i16>(kClip));
    const int8x16_t ones = vdupq_n_s8(1);
    const int32x4_t zero32 = vdupq_n_s32(0);
    const uint8x16_t byte_bias = vdupq_n_u8(128);

    int64x2_t total_low = vdupq_n_s64(0);
    int64x2_t total_high = vdupq_n_s64(0);

    for (int i = 0; i < count; i += 16) {
        const int16x8_t raw0 = vld1q_s16(acc + i);
        const int16x8_t raw1 = vld1q_s16(acc + i + 8);
        const int16x8_t clipped0 = vminq_s16(vmaxq_s16(raw0, zero16), clip16);
        const int16x8_t clipped1 = vminq_s16(vmaxq_s16(raw1, zero16), clip16);
        const uint16x8_t c0 = vreinterpretq_u16_s16(clipped0);
        const uint16x8_t c1 = vreinterpretq_u16_s16(clipped1);
        const uint16x8_t square0 = vmulq_u16(c0, c0);
        const uint16x8_t square1 = vmulq_u16(c1, c1);

        const uint8x16_t low_bytes = vcombine_u8(
            vmovn_u16(square0),
            vmovn_u16(square1)
        );
        const uint8x16_t high_bytes = vcombine_u8(
            vshrn_n_u16(square0, 8),
            vshrn_n_u16(square1, 8)
        );
        const int8x16_t low_centered =
            vreinterpretq_s8_u8(vsubq_u8(low_bytes, byte_bias));
        const int8x16_t high_centered =
            vreinterpretq_s8_u8(vsubq_u8(high_bytes, byte_bias));
        const int8x16_t weight_vec = vld1q_s8(weights + i);

        const int32x4_t weight_sum = vdotq_s32(zero32, ones, weight_vec);
        int32x4_t low_dot = vdotq_s32(zero32, low_centered, weight_vec);
        int32x4_t high_dot = vdotq_s32(zero32, high_centered, weight_vec);

        low_dot = vmlaq_n_s32(low_dot, weight_sum, 128);
        high_dot = vmlaq_n_s32(high_dot, weight_sum, 128);
        const int32x4_t contribution =
            vmlaq_n_s32(low_dot, high_dot, 256);

        // Widen each 16-element block immediately. This avoids accumulating
        // more than 32-bit range even for the wider P2-Pro layout.
        total_low = vaddq_s64(
            total_low,
            vmovl_s32(vget_low_s32(contribution))
        );
        total_high = vaddq_s64(
            total_high,
            vmovl_s32(vget_high_s32(contribution))
        );
    }

    return horizontal_sum_i64x4_neon(total_low, total_high);
}

[[nodiscard]] i64 dot_pair_screlu_i8_dotprod(
    const i16* acc0,
    const std::int8_t* weights0,
    const i16* acc1,
    const std::int8_t* weights1,
    int count
) noexcept {
    return dot_screlu_i8_dotprod(acc0, weights0, count)
        + dot_screlu_i8_dotprod(acc1, weights1, count);
}
#endif

template<class Layout, class Net>
[[nodiscard]] int forward_selected(
    int bucket,
    const Net& net,
    const Accumulator<Layout>& stm_acc,
    const Accumulator<Layout>& nstm_acc
) noexcept {
    const std::size_t base = static_cast<std::size_t>(bucket) * 2 * Layout::HiddenSize;
    const i16* w_stm = net.w1.data() + base;
    const i16* w_nstm = w_stm + Layout::HiddenSize;

    i64 output = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (net.forward_madd_safe) {
        output += dot_pair_screlu_i16_madd_avx512(
            stm_acc.data(),
            w_stm,
            nstm_acc.data(),
            w_nstm,
            Layout::HiddenSize
        );
    } else if (net.forward_i32_safe) {
        output += dot_pair_screlu_i16_i32_avx512(
            stm_acc.data(),
            w_stm,
            nstm_acc.data(),
            w_nstm,
            Layout::HiddenSize
        );
    } else {
        output += dot_screlu_i16_avx512(stm_acc.data(), w_stm, Layout::HiddenSize);
        output += dot_screlu_i16_avx512(nstm_acc.data(), w_nstm, Layout::HiddenSize);
    }
#elif defined(__AVX2__)
#if defined(__AVXVNNI__) \
    && !(defined(__AVX512F__) && defined(__AVX512BW__))
    if (net.forward_vnni_safe && net.w1_i8.size() == net.w1.size()) {
        const std::int8_t* w_stm_i8 = net.w1_i8.data() + base;
        const std::int8_t* w_nstm_i8 = w_stm_i8 + Layout::HiddenSize;
        output += dot_pair_screlu_i8_vnni_avx2(
            stm_acc.data(),
            w_stm_i8,
            nstm_acc.data(),
            w_nstm_i8,
            Layout::HiddenSize
        );
    } else if (net.forward_madd_safe) {
#else
    if (net.forward_madd_safe) {
#endif
        output += dot_pair_screlu_i16_madd_avx2(
            stm_acc.data(),
            w_stm,
            nstm_acc.data(),
            w_nstm,
            Layout::HiddenSize
        );
    } else if (net.forward_i32_safe) {
        output += dot_pair_screlu_i16_i32_avx2(
            stm_acc.data(),
            w_stm,
            nstm_acc.data(),
            w_nstm,
            Layout::HiddenSize
        );
    } else {
        output += dot_screlu_i16_avx2(stm_acc.data(), w_stm, Layout::HiddenSize);
        output += dot_screlu_i16_avx2(nstm_acc.data(), w_nstm, Layout::HiddenSize);
    }
#elif defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    if (net.forward_vnni_safe && net.w1_i8.size() == net.w1.size()) {
        const std::int8_t* w_stm_i8 = net.w1_i8.data() + base;
        const std::int8_t* w_nstm_i8 = w_stm_i8 + Layout::HiddenSize;
        output += dot_pair_screlu_i8_dotprod(
            stm_acc.data(), w_stm_i8,
            nstm_acc.data(), w_nstm_i8,
            Layout::HiddenSize
        );
    } else {
        output += dot_pair_screlu_i16_neon(
            stm_acc.data(), w_stm,
            nstm_acc.data(), w_nstm,
            Layout::HiddenSize
        );
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    output += dot_pair_screlu_i16_neon(
        stm_acc.data(), w_stm,
        nstm_acc.data(), w_nstm,
        Layout::HiddenSize
    );
#else
    for (int i = 0; i < Layout::HiddenSize; ++i)
        output += static_cast<i64>(screlu(stm_acc[static_cast<std::size_t>(i)]))
            * static_cast<i64>(w_stm[i]);

    for (int i = 0; i < Layout::HiddenSize; ++i)
        output += static_cast<i64>(screlu(nstm_acc[static_cast<std::size_t>(i)]))
            * static_cast<i64>(w_nstm[i]);
#endif

    output /= kQa;
    output += static_cast<i64>(net.b1[static_cast<std::size_t>(bucket)]);
    output *= net.scale;
    output /= static_cast<i64>(kQa) * kQb;

    return static_cast<int>(std::clamp<i64>(output, -32000, 32000));
}

template<class Layout>
[[nodiscard]] int forward(
    const Position& pos,
    const Network<Layout>& net,
    const Accumulator<Layout>& stm_acc,
    const Accumulator<Layout>& nstm_acc
) noexcept {
    const Color stm = static_cast<Color>(pos.side_to_move);
    return forward_selected<Layout>(
        output_bucket<Layout>(pos, stm),
        net,
        stm_acc,
        nstm_acc
    );
}

[[nodiscard]] int forward_h32(
    const Position& pos,
    const H32Network& net,
    const H32Accumulator& stm_acc,
    const H32Accumulator& nstm_acc
) noexcept {
    return forward_selected<P2H32Layout>(
        h32_output_bucket(pos),
        net,
        stm_acc,
        nstm_acc
    );
}

template<class Layout>
[[nodiscard]] int forward_reference(
    const Position& pos,
    const Network<Layout>& net,
    const Accumulator<Layout>& stm_acc,
    const Accumulator<Layout>& nstm_acc
) noexcept {
    const Color stm = static_cast<Color>(pos.side_to_move);
    const int bucket = output_bucket<Layout>(pos, stm);
    const std::size_t base = static_cast<std::size_t>(bucket) * 2 * Layout::HiddenSize;
    const i16* w_stm = net.w1.data() + base;
    const i16* w_nstm = w_stm + Layout::HiddenSize;

    i64 output = 0;
    for (int i = 0; i < Layout::HiddenSize; ++i)
        output += static_cast<i64>(screlu(stm_acc[static_cast<std::size_t>(i)]))
            * static_cast<i64>(w_stm[i]);

    for (int i = 0; i < Layout::HiddenSize; ++i)
        output += static_cast<i64>(screlu(nstm_acc[static_cast<std::size_t>(i)]))
            * static_cast<i64>(w_nstm[i]);

    output /= kQa;
    output += static_cast<i64>(net.b1[static_cast<std::size_t>(bucket)]);
    output *= net.scale;
    output /= static_cast<i64>(kQa) * kQb;

    return static_cast<int>(std::clamp<i64>(output, -32000, 32000));
}

[[nodiscard]] int forward_h32_reference(
    const Position& pos,
    const H32Network& net,
    const H32Accumulator& stm_acc,
    const H32Accumulator& nstm_acc
) noexcept {
    const int bucket = h32_output_bucket(pos);
    const std::size_t base = static_cast<std::size_t>(bucket)
        * 2 * P2H32Layout::HiddenSize;
    const i16* w_stm = net.w1.data() + base;
    const i16* w_nstm = w_stm + P2H32Layout::HiddenSize;
    i64 output = 0;
    for (int i = 0; i < P2H32Layout::HiddenSize; ++i) {
        output += static_cast<i64>(screlu(stm_acc[static_cast<std::size_t>(i)]))
            * static_cast<i64>(w_stm[i]);
        output += static_cast<i64>(screlu(nstm_acc[static_cast<std::size_t>(i)]))
            * static_cast<i64>(w_nstm[i]);
    }
    output /= kQa;
    output += static_cast<i64>(net.b1[static_cast<std::size_t>(bucket)]);
    output *= net.scale;
    output /= static_cast<i64>(kQa) * kQb;
    return static_cast<int>(std::clamp<i64>(output, -32000, 32000));
}

template<class Layout>
[[nodiscard]] int eval_lazy(
    const Position& pos,
    const Network<Layout>& net
) noexcept {
    if (!net.loaded || !net.valid())
        return 0;

    thread_local Accumulator<Layout> white_acc{};
    thread_local Accumulator<Layout> black_acc{};

    rebuild_accumulator<Layout>(pos, WHITE, white_acc, net);
    rebuild_accumulator<Layout>(pos, BLACK, black_acc, net);

    const Color stm = static_cast<Color>(pos.side_to_move);
    const auto& stm_acc = stm == WHITE ? white_acc : black_acc;
    const auto& nstm_acc = stm == WHITE ? black_acc : white_acc;
    return forward<Layout>(pos, net, stm_acc, nstm_acc);
}

template<class Layout>
void refresh_p2_accumulator_from_cache(
    P2RefreshCache<Layout>& cache,
    const Network<Layout>& net,
    const Position& pos,
    Color perspective,
    u32 current_generation,
    Accumulator<Layout>& out
) noexcept {
    if (!net.loaded || !net.valid())
        return;

    cache.sync(current_generation);
    const int bucket = input_bucket<Layout>(pos, perspective);
    P2RefreshEntry<Layout>& entry =
        cache.entries[static_cast<std::size_t>(perspective)]
                     [static_cast<std::size_t>(bucket)];
    if (!entry.initialized)
        entry.reset_to_biases(net);

    for (int sq_idx = 0; sq_idx < SQ_NB; ++sq_idx) {
        const Square sq = static_cast<Square>(sq_idx);
        const Piece old_pc = entry.board[static_cast<std::size_t>(sq_idx)];
        const Piece new_pc = piece_on(pos, sq);
        if (old_pc == new_pc)
            continue;

        const int old_idx = feature_index<Layout>(
            perspective,
            bucket,
            old_pc,
            sq
        );
        if (old_idx >= 0)
            sub_feature<Layout>(entry.accumulation, net, old_idx);

        const int new_idx = feature_index<Layout>(
            perspective,
            bucket,
            new_pc,
            sq
        );
        if (new_idx >= 0)
            add_feature<Layout>(entry.accumulation, net, new_idx);

        entry.board[static_cast<std::size_t>(sq_idx)] = new_pc;
    }

    out = entry.accumulation;
}

} // namespace

template<class Layout>
struct P2StackStateSet {
    static constexpr std::size_t Capacity = P2AccumulatorStack::Capacity;

    struct alignas(64) State {
        alignas(64) std::array<Accumulator<Layout>, COLOR_NB> accumulation{};
        P2MoveDiff diff{};
        std::uint8_t computed_mask = 0;
    };

    std::array<State, Capacity> states{};
    P2RefreshCache<Layout> refresh_cache{};
    std::size_t state_count = 1;
    u32 generation = 0;

    void sync_generation(u32 current) noexcept {
        if (generation == current)
            return;

        for (std::size_t i = 0; i < state_count; ++i)
            states[i].computed_mask = 0;
        refresh_cache.sync(current);
        generation = current;
    }

    void reset(u32 current_generation) noexcept {
        sync_generation(current_generation);
        state_count = 1;
        states[0].diff = {};
        states[0].computed_mask = 0;
    }

    void push(const Position& pos, Move move, u32 current_generation) noexcept {
        sync_generation(current_generation);
        if (state_count >= states.size()) {
            assert(false && "P2 accumulator stack overflow");
            std::abort();
        }

        State& next = states[state_count++];
        next.diff = make_p2_move_diff<Layout>(pos, move);
        next.computed_mask = 0;
    }

    void pop() noexcept {
        if (state_count <= 1) {
            assert(false && "P2 accumulator stack underflow");
            std::abort();
        }
        --state_count;
    }

    [[nodiscard]] const Accumulator<Layout>& ensure(
        const Position& pos,
        Color perspective,
        const Network<Layout>& net,
        u32 current_generation
    ) noexcept {
        sync_generation(current_generation);

        const std::uint8_t mask = p2_perspective_mask(perspective);
        const std::size_t current = state_count - 1;
        if ((states[current].computed_mask & mask) != 0)
            return states[current].accumulation[static_cast<std::size_t>(perspective)];

        std::size_t boundary = current;
        bool found_computed = false;
        for (std::size_t i = current;; --i) {
            if ((states[i].computed_mask & mask) != 0) {
                boundary = i;
                found_computed = true;
                break;
            }
            if (i > 0
                && states[i].diff[static_cast<std::size_t>(perspective)].refresh) {
                boundary = i;
                break;
            }
            if (i == 0) {
                boundary = 0;
                break;
            }
        }

        if (found_computed) {
            for (std::size_t i = boundary + 1; i <= current; ++i) {
                apply_p2_diff<Layout>(
                    states[i].accumulation[static_cast<std::size_t>(perspective)],
                    states[i - 1].accumulation[static_cast<std::size_t>(perspective)],
                    net,
                    states[i].diff[static_cast<std::size_t>(perspective)],
                    true
                );
                states[i].computed_mask = static_cast<std::uint8_t>(
                    states[i].computed_mask | mask
                );
            }
        } else {
            refresh_p2_accumulator_from_cache(
                refresh_cache,
                net,
                pos,
                perspective,
                generation,
                states[current].accumulation[static_cast<std::size_t>(perspective)]
            );
            states[current].computed_mask = static_cast<std::uint8_t>(
                states[current].computed_mask | mask
            );

            for (std::size_t i = current; i > boundary; --i) {
                apply_p2_diff<Layout>(
                    states[i - 1].accumulation[static_cast<std::size_t>(perspective)],
                    states[i].accumulation[static_cast<std::size_t>(perspective)],
                    net,
                    states[i].diff[static_cast<std::size_t>(perspective)],
                    false
                );
                states[i - 1].computed_mask = static_cast<std::uint8_t>(
                    states[i - 1].computed_mask | mask
                );
            }
        }

        return states[current].accumulation[static_cast<std::size_t>(perspective)];
    }
};

struct H32StackStateSet {
    static constexpr std::size_t Capacity = P2AccumulatorStack::Capacity;

    struct alignas(64) State {
        alignas(64) std::array<H32Accumulator, COLOR_NB> accumulation{};
        H32MoveDiff diff{};
        std::uint8_t computed_mask = 0;
    };

    std::array<State, Capacity> states{};
    H32RefreshCache refresh_cache{};
    std::size_t state_count = 1;
    u32 generation = 0;

    void sync_generation(u32 current) noexcept {
        if (generation == current)
            return;
        for (std::size_t i = 0; i < state_count; ++i)
            states[i].computed_mask = 0;
        refresh_cache.sync(current);
        generation = current;
    }

    void reset(u32 current) noexcept {
        sync_generation(current);
        state_count = 1;
        states[0].diff = {};
        states[0].computed_mask = 0;
    }

    void push(const Position& pos, Move move, u32 current) noexcept {
        sync_generation(current);
        if (state_count >= states.size()) {
            assert(false && "H32 accumulator stack overflow");
            std::abort();
        }
        State& next = states[state_count++];
        next.diff = make_h32_move_diff(pos, move);
        next.computed_mask = 0;
    }

    void pop() noexcept {
        if (state_count <= 1) {
            assert(false && "H32 accumulator stack underflow");
            std::abort();
        }
        --state_count;
    }

    [[nodiscard]] const H32Accumulator& ensure(
        const Position& pos,
        Color perspective,
        const H32Network& net,
        u32 current
    ) noexcept {
        sync_generation(current);
        const std::uint8_t mask = p2_perspective_mask(perspective);
        const std::size_t index = state_count - 1;
        if ((states[index].computed_mask & mask) != 0)
            return states[index].accumulation[static_cast<std::size_t>(perspective)];

        std::size_t boundary = index;
        bool found = false;
        for (std::size_t i = index;; --i) {
            if ((states[i].computed_mask & mask) != 0) {
                boundary = i;
                found = true;
                break;
            }
            if (i > 0
                && states[i].diff[static_cast<std::size_t>(perspective)].refresh) {
                boundary = i;
                break;
            }
            if (i == 0) {
                boundary = 0;
                break;
            }
        }

        if (found) {
            for (std::size_t i = boundary + 1; i <= index; ++i) {
                apply_h32_diff(
                    states[i].accumulation[static_cast<std::size_t>(perspective)],
                    states[i - 1].accumulation[static_cast<std::size_t>(perspective)],
                    net,
                    states[i].diff[static_cast<std::size_t>(perspective)],
                    true
                );
                states[i].computed_mask = static_cast<std::uint8_t>(
                    states[i].computed_mask | mask
                );
            }
        } else {
            refresh_h32_accumulator_from_cache(
                refresh_cache,
                net,
                pos,
                perspective,
                generation,
                states[index].accumulation[static_cast<std::size_t>(perspective)]
            );
            states[index].computed_mask = static_cast<std::uint8_t>(
                states[index].computed_mask | mask
            );
            for (std::size_t i = index; i > boundary; --i) {
                apply_h32_diff(
                    states[i - 1].accumulation[static_cast<std::size_t>(perspective)],
                    states[i].accumulation[static_cast<std::size_t>(perspective)],
                    net,
                    states[i].diff[static_cast<std::size_t>(perspective)],
                    false
                );
                states[i - 1].computed_mask = static_cast<std::uint8_t>(
                    states[i - 1].computed_mask | mask
                );
            }
        }
        return states[index].accumulation[static_cast<std::size_t>(perspective)];
    }
};

struct P2AccumulatorStack::Impl {
    std::unique_ptr<P2StackStateSet<P2Layout>> p2;
    std::unique_ptr<P2StackStateSet<P2K32Layout>> p2k32;
    std::unique_ptr<H32StackStateSet> p2h32;
    std::unique_ptr<P2StackStateSet<P2ProLayout>> p2pro;

    [[nodiscard]] P2StackStateSet<P2Layout>& p2_state() {
        if (!p2)
            p2 = std::make_unique<P2StackStateSet<P2Layout>>();
        return *p2;
    }

    [[nodiscard]] P2StackStateSet<P2K32Layout>& p2k32_state() {
        if (!p2k32)
            p2k32 = std::make_unique<P2StackStateSet<P2K32Layout>>();
        return *p2k32;
    }

    [[nodiscard]] H32StackStateSet& p2h32_state() {
        if (!p2h32)
            p2h32 = std::make_unique<H32StackStateSet>();
        return *p2h32;
    }

    [[nodiscard]] P2StackStateSet<P2ProLayout>& p2pro_state() {
        if (!p2pro)
            p2pro = std::make_unique<P2StackStateSet<P2ProLayout>>();
        return *p2pro;
    }

    [[nodiscard]] static u32 current_generation() noexcept {
        return g_p2_generation.load(std::memory_order_acquire);
    }

    void reset() noexcept {
        const u32 current = current_generation();
        switch (g_p2_active) {
            case P2ActiveKind::P2:
                p2_state().reset(current);
                break;
            case P2ActiveKind::P2K32:
                p2k32_state().reset(current);
                break;
            case P2ActiveKind::P2H32:
                p2h32_state().reset(current);
                break;
            case P2ActiveKind::P2Pro:
                p2pro_state().reset(current);
                break;
            case P2ActiveKind::None:
                if (p2)
                    p2->reset(current);
                if (p2k32)
                    p2k32->reset(current);
                if (p2h32)
                    p2h32->reset(current);
                if (p2pro)
                    p2pro->reset(current);
                break;
        }
    }

    void push(const Position& pos, Move move) noexcept {
        const u32 current = current_generation();
        switch (g_p2_active) {
            case P2ActiveKind::P2:
                p2_state().push(pos, move, current);
                break;
            case P2ActiveKind::P2K32:
                p2k32_state().push(pos, move, current);
                break;
            case P2ActiveKind::P2H32:
                p2h32_state().push(pos, move, current);
                break;
            case P2ActiveKind::P2Pro:
                p2pro_state().push(pos, move, current);
                break;
            case P2ActiveKind::None:
                break;
        }
    }

    void pop() noexcept {
        switch (g_p2_active) {
            case P2ActiveKind::P2:
                if (p2)
                    p2->pop();
                break;
            case P2ActiveKind::P2K32:
                if (p2k32)
                    p2k32->pop();
                break;
            case P2ActiveKind::P2H32:
                if (p2h32)
                    p2h32->pop();
                break;
            case P2ActiveKind::P2Pro:
                if (p2pro)
                    p2pro->pop();
                break;
            case P2ActiveKind::None:
                break;
        }
    }

    [[nodiscard]] std::size_t size() const noexcept {
        switch (g_p2_active) {
            case P2ActiveKind::P2:
                return p2 ? p2->state_count : 1;
            case P2ActiveKind::P2K32:
                return p2k32 ? p2k32->state_count : 1;
            case P2ActiveKind::P2H32:
                return p2h32 ? p2h32->state_count : 1;
            case P2ActiveKind::P2Pro:
                return p2pro ? p2pro->state_count : 1;
            case P2ActiveKind::None:
                return 1;
        }
        return 1;
    }

    [[nodiscard]] const Accumulator<P2Layout>& ensure_p2(
        const Position& pos,
        Color perspective
    ) noexcept {
        return p2_state().ensure(
            pos,
            perspective,
            g_p2,
            current_generation()
        );
    }

    [[nodiscard]] const Accumulator<P2K32Layout>& ensure_p2k32(
        const Position& pos,
        Color perspective
    ) noexcept {
        return p2k32_state().ensure(
            pos,
            perspective,
            g_p2k32,
            current_generation()
        );
    }

    [[nodiscard]] const H32Accumulator& ensure_p2h32(
        const Position& pos,
        Color perspective
    ) noexcept {
        return p2h32_state().ensure(
            pos,
            perspective,
            g_p2h32,
            current_generation()
        );
    }

    [[nodiscard]] const Accumulator<P2ProLayout>& ensure_p2pro(
        const Position& pos,
        Color perspective
    ) noexcept {
        return p2pro_state().ensure(
            pos,
            perspective,
            g_p2pro,
            current_generation()
        );
    }
};

P2AccumulatorStack::P2AccumulatorStack() noexcept
    : impl_(std::make_unique<Impl>()) {
    impl_->reset();
}

P2AccumulatorStack::~P2AccumulatorStack() = default;
P2AccumulatorStack::P2AccumulatorStack(P2AccumulatorStack&&) noexcept = default;
P2AccumulatorStack& P2AccumulatorStack::operator=(P2AccumulatorStack&&) noexcept = default;

void P2AccumulatorStack::reset() noexcept {
    impl_->reset();
}

void P2AccumulatorStack::push(const Position& pos, Move move) noexcept {
    impl_->push(pos, move);
}

void P2AccumulatorStack::pop() noexcept {
    impl_->pop();
}

std::size_t P2AccumulatorStack::size() const noexcept {
    return impl_->size();
}

bool load_p2(const std::string& path) {
    Network<P2Layout> p2_candidate{};
    if (load_network(p2_candidate, path)) {
        g_p2 = std::move(p2_candidate);
        clear_network(g_p2k32);
        clear_network(g_p2h32);
        clear_network(g_p2pro);
        g_p2_active = P2ActiveKind::P2;
        (void)next_p2_generation();
        return true;
    }

    const std::string p2_error = g_error;
    Network<P2K32Layout> p2k32_candidate{};
    if (load_network(p2k32_candidate, path)) {
        clear_network(g_p2);
        g_p2k32 = std::move(p2k32_candidate);
        clear_network(g_p2h32);
        clear_network(g_p2pro);
        g_p2_active = P2ActiveKind::P2K32;
        (void)next_p2_generation();
        return true;
    }

    const std::string p2k32_error = g_error;
    H32Network p2h32_candidate{};
    if (load_network(p2h32_candidate, path)) {
        clear_network(g_p2);
        clear_network(g_p2k32);
        g_p2h32 = std::move(p2h32_candidate);
        clear_network(g_p2pro);
        g_p2_active = P2ActiveKind::P2H32;
        (void)next_p2_generation();
        return true;
    }

    const std::string p2h32_error = g_error;
    Network<P2ProLayout> p2pro_candidate{};
    if (load_network(p2pro_candidate, path)) {
        clear_network(g_p2);
        clear_network(g_p2k32);
        clear_network(g_p2h32);
        g_p2pro = std::move(p2pro_candidate);
        g_p2_active = P2ActiveKind::P2Pro;
        (void)next_p2_generation();
        return true;
    }

    g_p2_active = P2ActiveKind::None;
    clear_network(g_p2);
    clear_network(g_p2k32);
    clear_network(g_p2h32);
    clear_network(g_p2pro);
    g_error = "failed to load MNUE P2/P2-K32/P2-H32/P2Pro: P2: " + p2_error
        + "; P2-K32: " + p2k32_error
        + "; P2-H32: " + p2h32_error
        + "; P2Pro: " + g_error;
    return false;
}

bool load_p4(const std::string& path) {
    return load_network(g_p4, path);
}

class MemoryInputBuffer final : public std::streambuf {
public:
    MemoryInputBuffer(const char* begin, const char* end) {
        char* first = const_cast<char*>(begin);
        setg(first, first, const_cast<char*>(end));
    }

protected:
    pos_type seekoff(
        off_type offset,
        std::ios_base::seekdir direction,
        std::ios_base::openmode mode
    ) override {
        if ((mode & std::ios_base::in) == 0)
            return pos_type(off_type(-1));

        off_type base = 0;
        if (direction == std::ios_base::cur)
            base = static_cast<off_type>(gptr() - eback());
        else if (direction == std::ios_base::end)
            base = static_cast<off_type>(egptr() - eback());
        else if (direction != std::ios_base::beg)
            return pos_type(off_type(-1));

        const off_type next = base + offset;
        const off_type size = static_cast<off_type>(egptr() - eback());
        if (next < 0 || next > size)
            return pos_type(off_type(-1));

        setg(eback(), eback() + next, egptr());
        return pos_type(next);
    }

    pos_type seekpos(pos_type position, std::ios_base::openmode mode) override {
        return seekoff(off_type(position), std::ios_base::beg, mode);
    }
};

template<class Layout>
[[nodiscard]] bool load_p2_family_from_memory(
    Network<Layout>& target,
    const char* data,
    std::size_t size,
    const std::string& path
) {
    MemoryInputBuffer buffer(data, data + size);
    std::istream stream(&buffer);
    Network<Layout> candidate{};
    if (!load_network_from_stream(
            stream,
            candidate,
            path,
            static_cast<std::uintmax_t>(size)
        )) {
        return false;
    }

    target = std::move(candidate);
    return true;
}

[[nodiscard]] bool load_p2_family_from_memory(
    H32Network& target,
    const char* data,
    std::size_t size,
    const std::string& path
) {
    MemoryInputBuffer buffer(data, data + size);
    std::istream stream(&buffer);
    H32Network candidate{};
    if (!load_h32_from_stream(stream, candidate, path, size))
        return false;
    target = std::move(candidate);
    return true;
}

// Load the compile-time embedded P2/P2-K32/P2Pro network from .rodata.
// Uses a zero-copy memory streambuf to avoid an extra 20-28 MiB string allocation.
bool load_p2_embedded() {
    if (!mnue_p2_embedded_data || !mnue_p2_embedded_end) {
        g_error = "embedded MNUE P2/P2-K32/P2Pro not linked (missing MnueEmbedded.o)";
        return false;
    }

    const auto begin =
        reinterpret_cast<std::uintptr_t>(mnue_p2_embedded_data);
    const auto end =
        reinterpret_cast<std::uintptr_t>(mnue_p2_embedded_end);

    if (end <= begin) {
        g_error = "invalid embedded MNUE range";
        return false;
    }

    const std::size_t size = static_cast<std::size_t>(end - begin);

    if (load_p2_family_from_memory(
            g_p2,
            mnue_p2_embedded_data,
            size,
            kEmbeddedP2Filename
        )) {
        clear_network(g_p2k32);
        clear_network(g_p2h32);
        clear_network(g_p2pro);
        g_p2_active = P2ActiveKind::P2;
        (void)next_p2_generation();
        return true;
    }

    const std::string p2_error = g_error;
    if (load_p2_family_from_memory(
            g_p2k32,
            mnue_p2_embedded_data,
            size,
            kEmbeddedP2Filename
        )) {
        clear_network(g_p2);
        clear_network(g_p2h32);
        clear_network(g_p2pro);
        g_p2_active = P2ActiveKind::P2K32;
        (void)next_p2_generation();
        return true;
    }

    const std::string p2k32_error = g_error;
    if (load_p2_family_from_memory(
            g_p2h32,
            mnue_p2_embedded_data,
            size,
            kEmbeddedP2Filename
        )) {
        clear_network(g_p2);
        clear_network(g_p2k32);
        clear_network(g_p2pro);
        g_p2_active = P2ActiveKind::P2H32;
        (void)next_p2_generation();
        return true;
    }

    const std::string p2h32_error = g_error;
    if (load_p2_family_from_memory(
            g_p2pro,
            mnue_p2_embedded_data,
            size,
            kEmbeddedP2Filename
        )) {
        clear_network(g_p2);
        clear_network(g_p2k32);
        clear_network(g_p2h32);
        g_p2_active = P2ActiveKind::P2Pro;
        (void)next_p2_generation();
        return true;
    }

    g_p2_active = P2ActiveKind::None;
    clear_network(g_p2);
    clear_network(g_p2k32);
    clear_network(g_p2h32);
    clear_network(g_p2pro);
    g_error = "failed to load embedded MNUE P2/P2-K32/P2-H32/P2Pro: P2: " + p2_error
        + "; P2-K32: " + p2k32_error
        + "; P2-H32: " + p2h32_error
        + "; P2Pro: " + g_error;
    return false;
}

bool p2_embedded_available() noexcept {
    if (!mnue_p2_embedded_data || !mnue_p2_embedded_end)
        return false;
    const auto begin =
        reinterpret_cast<std::uintptr_t>(mnue_p2_embedded_data);
    const auto end =
        reinterpret_cast<std::uintptr_t>(mnue_p2_embedded_end);
    return end > begin;
}

void unload_p2() noexcept {
    clear_network(g_p2);
    clear_network(g_p2k32);
    clear_network(g_p2h32);
    clear_network(g_p2pro);
    g_p2_active = P2ActiveKind::None;
    (void)next_p2_generation();
}

void unload_p4() noexcept {
    clear_network(g_p4);
}

void unload_all() noexcept {
    unload_p2();
    unload_p4();
}

bool p2_loaded() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:    return g_p2.loaded && g_p2.valid();
        case P2ActiveKind::P2K32: return g_p2k32.loaded && g_p2k32.valid();
        case P2ActiveKind::P2H32: return g_p2h32.loaded && g_p2h32.valid();
        case P2ActiveKind::P2Pro: return g_p2pro.loaded && g_p2pro.valid();
        case P2ActiveKind::None:  return false;
    }
    return false;
}

bool p4_loaded() noexcept {
    return g_p4.loaded;
}

const std::string& p2_path() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:    return g_p2.path;
        case P2ActiveKind::P2K32: return g_p2k32.path;
        case P2ActiveKind::P2H32: return g_p2h32.path;
        case P2ActiveKind::P2Pro: return g_p2pro.path;
        case P2ActiveKind::None:  return kEmptyPath;
    }
    return kEmptyPath;
}

const std::string& p4_path() noexcept {
    return g_p4.path;
}

const std::string& last_error() noexcept {
    return g_error;
}

[[nodiscard]] bool p2_vnni_forward_enabled() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:
            return g_p2.loaded && g_p2.valid() && g_p2.forward_vnni_safe
                && g_p2.w1_i8.size() == g_p2.w1.size();
        case P2ActiveKind::P2K32:
            return g_p2k32.loaded && g_p2k32.valid() && g_p2k32.forward_vnni_safe
                && g_p2k32.w1_i8.size() == g_p2k32.w1.size();
        case P2ActiveKind::P2H32:
            return g_p2h32.loaded && g_p2h32.valid() && g_p2h32.forward_vnni_safe
                && g_p2h32.w1_i8.size() == g_p2h32.w1.size();
        case P2ActiveKind::P2Pro:
            return g_p2pro.loaded && g_p2pro.valid() && g_p2pro.forward_vnni_safe
                && g_p2pro.w1_i8.size() == g_p2pro.w1.size();
        case P2ActiveKind::None:
            return false;
    }
    return false;
}

const char* eval_simd_name() noexcept {
#if defined(__AVXVNNI__) \
    && !(defined(__AVX512F__) && defined(__AVX512BW__))
    if (p2_vnni_forward_enabled())
        return "AVX-VNNI-SCRELU-exact";
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
    return "AVX-512BW";
#elif defined(__AVX2__)
    return "AVX2";
#elif defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    return p2_vnni_forward_enabled() ? "ARM-Dotprod-SCRELU-exact" : "ARM-NEON-fallback";
#elif defined(__aarch64__) && defined(__ARM_NEON)
    return "ARM-NEON";
#else
    return "scalar";
#endif
}

const char* p2_eval_name() noexcept {
    return p2_kind_eval_name(g_p2_active);
}

const char* p2_arch_name() noexcept {
    return p2_kind_arch_name(g_p2_active);
}

const char* p2_short_name() noexcept {
    return p2_kind_short_name(g_p2_active);
}

int p2_input_size() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:    return P2Layout::InputSize;
        case P2ActiveKind::P2K32: return P2K32Layout::InputSize;
        case P2ActiveKind::P2H32: return P2H32Layout::InputSize;
        case P2ActiveKind::P2Pro: return P2ProLayout::InputSize;
        case P2ActiveKind::None:  return 0;
    }
    return 0;
}

int p2_hidden_size() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:    return P2Layout::HiddenSize;
        case P2ActiveKind::P2K32: return P2K32Layout::HiddenSize;
        case P2ActiveKind::P2H32: return P2H32Layout::HiddenSize;
        case P2ActiveKind::P2Pro: return P2ProLayout::HiddenSize;
        case P2ActiveKind::None:  return 0;
    }
    return 0;
}

int p2_input_buckets() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:    return P2Layout::InputBuckets;
        case P2ActiveKind::P2K32: return P2K32Layout::InputBuckets;
        case P2ActiveKind::P2H32: return P2H32Layout::InputBuckets;
        case P2ActiveKind::P2Pro: return P2ProLayout::InputBuckets;
        case P2ActiveKind::None:  return 0;
    }
    return 0;
}

int p2_output_buckets() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:    return P2Layout::OutputBuckets;
        case P2ActiveKind::P2K32: return P2K32Layout::OutputBuckets;
        case P2ActiveKind::P2H32: return P2H32Layout::OutputBuckets;
        case P2ActiveKind::P2Pro: return P2ProLayout::OutputBuckets;
        case P2ActiveKind::None:  return 0;
    }
    return 0;
}

std::size_t p2_file_bytes() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:
            return static_cast<std::size_t>(file_bytes<P2Layout>());
        case P2ActiveKind::P2K32:
            return static_cast<std::size_t>(file_bytes<P2K32Layout>());
        case P2ActiveKind::P2H32:
            return static_cast<std::size_t>(h32_file_bytes());
        case P2ActiveKind::P2Pro:
            return static_cast<std::size_t>(file_bytes<P2ProLayout>());
        case P2ActiveKind::None:
            return 0;
    }
    return 0;
}

int eval_p2(const Position& pos) noexcept {
    struct StandaloneContext {
        P2AccumulatorStack stack{};
        Key key = 0;
        bool initialized = false;
    };

    thread_local StandaloneContext context{};
    if (!context.initialized || context.key != pos.key) {
        context.stack.reset();
        context.key = pos.key;
        context.initialized = true;
    }

    return eval_p2(pos, context.stack);
}

int eval_p2(const Position& pos, P2AccumulatorStack& stack) noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2: {
            if (!g_p2.loaded || !g_p2.valid())
                return 0;

            const auto& white_acc = stack.impl_->ensure_p2(pos, WHITE);
            const auto& black_acc = stack.impl_->ensure_p2(pos, BLACK);
            const Color stm = static_cast<Color>(pos.side_to_move);
            const auto& stm_acc = stm == WHITE ? white_acc : black_acc;
            const auto& nstm_acc = stm == WHITE ? black_acc : white_acc;
            return forward<P2Layout>(pos, g_p2, stm_acc, nstm_acc);
        }
        case P2ActiveKind::P2K32: {
            if (!g_p2k32.loaded || !g_p2k32.valid())
                return 0;

            const auto& white_acc = stack.impl_->ensure_p2k32(pos, WHITE);
            const auto& black_acc = stack.impl_->ensure_p2k32(pos, BLACK);
            const Color stm = static_cast<Color>(pos.side_to_move);
            const auto& stm_acc = stm == WHITE ? white_acc : black_acc;
            const auto& nstm_acc = stm == WHITE ? black_acc : white_acc;
            return forward<P2K32Layout>(pos, g_p2k32, stm_acc, nstm_acc);
        }
        case P2ActiveKind::P2H32: {
            if (!g_p2h32.loaded || !g_p2h32.valid())
                return 0;
            const auto& white_acc = stack.impl_->ensure_p2h32(pos, WHITE);
            const auto& black_acc = stack.impl_->ensure_p2h32(pos, BLACK);
            const Color stm = static_cast<Color>(pos.side_to_move);
            return forward_h32(
                pos,
                g_p2h32,
                stm == WHITE ? white_acc : black_acc,
                stm == WHITE ? black_acc : white_acc
            );
        }
        case P2ActiveKind::P2Pro: {
            if (!g_p2pro.loaded || !g_p2pro.valid())
                return 0;

            const auto& white_acc = stack.impl_->ensure_p2pro(pos, WHITE);
            const auto& black_acc = stack.impl_->ensure_p2pro(pos, BLACK);
            const Color stm = static_cast<Color>(pos.side_to_move);
            const auto& stm_acc = stm == WHITE ? white_acc : black_acc;
            const auto& nstm_acc = stm == WHITE ? black_acc : white_acc;
            return forward<P2ProLayout>(pos, g_p2pro, stm_acc, nstm_acc);
        }
        case P2ActiveKind::None:
            return 0;
    }
    return 0;
}

int debug_eval_p2_reference(const Position& pos) noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2: {
            if (!g_p2.loaded || !g_p2.valid())
                return 0;

            Accumulator<P2Layout> white_acc{};
            Accumulator<P2Layout> black_acc{};
            rebuild_accumulator<P2Layout>(pos, WHITE, white_acc, g_p2);
            rebuild_accumulator<P2Layout>(pos, BLACK, black_acc, g_p2);
            const Color stm = static_cast<Color>(pos.side_to_move);
            const auto& stm_acc = stm == WHITE ? white_acc : black_acc;
            const auto& nstm_acc = stm == WHITE ? black_acc : white_acc;
            return forward_reference<P2Layout>(pos, g_p2, stm_acc, nstm_acc);
        }
        case P2ActiveKind::P2K32: {
            if (!g_p2k32.loaded || !g_p2k32.valid())
                return 0;

            Accumulator<P2K32Layout> white_acc{};
            Accumulator<P2K32Layout> black_acc{};
            rebuild_accumulator<P2K32Layout>(pos, WHITE, white_acc, g_p2k32);
            rebuild_accumulator<P2K32Layout>(pos, BLACK, black_acc, g_p2k32);
            const Color stm = static_cast<Color>(pos.side_to_move);
            const auto& stm_acc = stm == WHITE ? white_acc : black_acc;
            const auto& nstm_acc = stm == WHITE ? black_acc : white_acc;
            return forward_reference<P2K32Layout>(pos, g_p2k32, stm_acc, nstm_acc);
        }
        case P2ActiveKind::P2H32: {
            if (!g_p2h32.loaded || !g_p2h32.valid())
                return 0;
            H32Accumulator white_acc{};
            H32Accumulator black_acc{};
            rebuild_h32_accumulator(pos, WHITE, white_acc, g_p2h32);
            rebuild_h32_accumulator(pos, BLACK, black_acc, g_p2h32);
            const Color stm = static_cast<Color>(pos.side_to_move);
            return forward_h32_reference(
                pos,
                g_p2h32,
                stm == WHITE ? white_acc : black_acc,
                stm == WHITE ? black_acc : white_acc
            );
        }
        case P2ActiveKind::P2Pro: {
            if (!g_p2pro.loaded || !g_p2pro.valid())
                return 0;

            Accumulator<P2ProLayout> white_acc{};
            Accumulator<P2ProLayout> black_acc{};
            rebuild_accumulator<P2ProLayout>(pos, WHITE, white_acc, g_p2pro);
            rebuild_accumulator<P2ProLayout>(pos, BLACK, black_acc, g_p2pro);
            const Color stm = static_cast<Color>(pos.side_to_move);
            const auto& stm_acc = stm == WHITE ? white_acc : black_acc;
            const auto& nstm_acc = stm == WHITE ? black_acc : white_acc;
            return forward_reference<P2ProLayout>(pos, g_p2pro, stm_acc, nstm_acc);
        }
        case P2ActiveKind::None:
            return 0;
    }
    return 0;
}

bool p2_i32_forward_enabled() noexcept {
#if (defined(__AVX512F__) && defined(__AVX512BW__)) || defined(__AVX2__)
    switch (g_p2_active) {
        case P2ActiveKind::P2:
            return g_p2.loaded && g_p2.valid() && g_p2.forward_i32_safe;
        case P2ActiveKind::P2K32:
            return g_p2k32.loaded && g_p2k32.valid() && g_p2k32.forward_i32_safe;
        case P2ActiveKind::P2H32:
            return g_p2h32.loaded && g_p2h32.valid() && g_p2h32.forward_i32_safe;
        case P2ActiveKind::P2Pro:
            return g_p2pro.loaded && g_p2pro.valid() && g_p2pro.forward_i32_safe;
        case P2ActiveKind::None:
            return false;
    }
    return false;
#else
    return false;
#endif
}

int p2_w1_max_abs() noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:
            return g_p2.loaded && g_p2.valid() ? g_p2.w1_max_abs : 0;
        case P2ActiveKind::P2K32:
            return g_p2k32.loaded && g_p2k32.valid() ? g_p2k32.w1_max_abs : 0;
        case P2ActiveKind::P2H32:
            return g_p2h32.loaded && g_p2h32.valid() ? g_p2h32.w1_max_abs : 0;
        case P2ActiveKind::P2Pro:
            return g_p2pro.loaded && g_p2pro.valid() ? g_p2pro.w1_max_abs : 0;
        case P2ActiveKind::None:
            return 0;
    }
    return 0;
}

int eval_p4_lazy(const Position& pos) noexcept {
    return eval_lazy(pos, g_p4);
}

template<class Layout>
void debug_dump_p2_features_impl(
    const Position& pos,
    const Network<Layout>& net,
    const char* label,
    std::ostream& out
) {
    out << "mnue debug " << label << '\n';
    out << "side_to_move " << pos.side_to_move << '\n';
    out << "hidden_size " << Layout::HiddenSize << '\n';

    const int out_bucket = output_bucket<Layout>(
        pos,
        static_cast<Color>(pos.side_to_move)
    );
    out << "output_bucket " << out_bucket << '\n';

    if (net.loaded && net.valid()) {
        out << "p2_loaded 1\n";
        out << "p2_eval_raw " << eval_p2(pos) << '\n';
    } else {
        out << "p2_loaded 0\n";
    }

    for (int persp = WHITE; persp <= BLACK; ++persp) {
        const Color perspective = static_cast<Color>(persp);
        const Square ksq = king_square_of(pos, perspective);
        const Square rksq = relative_square(perspective, ksq);
        const int bucket = input_bucket<Layout>(pos, perspective);

        out << "perspective " << persp
            << " king_sq " << static_cast<int>(ksq)
            << " rel_king_sq " << static_cast<int>(rksq)
            << " input_bucket " << bucket
            << '\n';

        Bitboard bb = pieces(pos);
        while (bb) {
            const Square sq = static_cast<Square>(
                std::countr_zero(static_cast<std::uint64_t>(bb))
            );
            bb &= (bb - 1);

            const Piece pc = piece_on(pos, sq);
            if (pc == PIECE_NONE)
                continue;

            const int idx = feature_index<Layout>(perspective, bucket, pc, sq);
            if (idx < 0)
                continue;

            out << "  feature"
                << " sq " << static_cast<int>(sq)
                << " rel_sq " << static_cast<int>(relative_square(perspective, sq))
                << " piece " << static_cast<int>(pc)
                << " color " << static_cast<int>(color_of(pc))
                << " type " << static_cast<int>(type_of(pc))
                << " idx " << idx
                << '\n';
        }
    }
}

void debug_dump_h32_features(const Position& pos, std::ostream& out) {
    out << "mnue debug p2-h32\n";
    out << "side_to_move " << pos.side_to_move << '\n';
    out << "hidden_size " << P2H32Layout::HiddenSize << '\n';
    out << "output_bucket " << h32_output_bucket(pos) << '\n';
    out << "p2_loaded " << (g_p2h32.loaded && g_p2h32.valid() ? 1 : 0) << '\n';
    if (g_p2h32.loaded && g_p2h32.valid())
        out << "p2_eval_raw " << eval_p2(pos) << '\n';

    for (int persp = WHITE; persp <= BLACK; ++persp) {
        const Color perspective = static_cast<Color>(persp);
        const H32Perspective transform = h32_perspective(pos, perspective);
        out << "perspective " << persp
            << " coarse_bucket " << transform.coarse_bucket
            << " fine_bucket " << transform.fine_bucket
            << " mirror " << (transform.mirror ? 1 : 0)
            << '\n';
        Bitboard bb = pieces(pos);
        while (bb) {
            const Square sq = static_cast<Square>(
                std::countr_zero(static_cast<std::uint64_t>(bb))
            );
            bb &= bb - 1;
            const Piece pc = piece_on(pos, sq);
            const int coarse = h32_coarse_feature_index(
                perspective,
                transform,
                pc,
                sq
            );
            const int fine = h32_fine_feature_index(
                perspective,
                transform,
                pc,
                sq
            );
            if (coarse < 0 && fine < 0)
                continue;
            out << "  feature sq " << static_cast<int>(sq)
                << " piece " << static_cast<int>(pc)
                << " coarse " << coarse
                << " fine " << fine
                << '\n';
        }
    }
}

void debug_dump_p2_features(const Position& pos, std::ostream& out) {
    switch (g_p2_active) {
        case P2ActiveKind::P2:
            debug_dump_p2_features_impl<P2Layout>(pos, g_p2, "p2", out);
            return;
        case P2ActiveKind::P2K32:
            debug_dump_p2_features_impl<P2K32Layout>(pos, g_p2k32, "p2-k32", out);
            return;
        case P2ActiveKind::P2H32:
            debug_dump_h32_features(pos, out);
            return;
        case P2ActiveKind::P2Pro:
            debug_dump_p2_features_impl<P2ProLayout>(pos, g_p2pro, "p2pro", out);
            return;
        case P2ActiveKind::None:
            out << "mnue debug p2\n";
            out << "p2_loaded 0\n";
            return;
    }
}

template<class Layout>
bool debug_check_p2_incremental_impl(
    const Position& pos,
    P2StackStateSet<Layout>& set,
    const Network<Layout>& net,
    const char* label,
    std::ostream& out
) noexcept {
    if (!net.loaded || !net.valid()) {
        out << "mnuecheck " << label << " loaded 0\n";
        return false;
    }

    const u32 current_generation = g_p2_generation.load(std::memory_order_acquire);
    set.sync_generation(current_generation);
    const std::size_t current = set.state_count - 1;
    const std::uint8_t valid_mask_before = set.states[current].computed_mask;
    const bool valid_before =
        valid_mask_before
        == static_cast<std::uint8_t>(
            p2_perspective_mask(WHITE) | p2_perspective_mask(BLACK)
        );
    const u32 generation_before = set.generation;

    const auto& white_acc = set.ensure(pos, WHITE, net, current_generation);
    const auto& black_acc = set.ensure(pos, BLACK, net, current_generation);
    const Color stm = static_cast<Color>(pos.side_to_move);
    const auto& stm_acc = stm == WHITE ? white_acc : black_acc;
    const auto& nstm_acc = stm == WHITE ? black_acc : white_acc;
    const int incremental = forward<Layout>(pos, net, stm_acc, nstm_acc);

    Accumulator<Layout> white_rebuild{};
    Accumulator<Layout> black_rebuild{};
    rebuild_accumulator<Layout>(pos, WHITE, white_rebuild, net);
    rebuild_accumulator<Layout>(pos, BLACK, black_rebuild, net);

    const auto& rebuilt_stm = stm == WHITE ? white_rebuild : black_rebuild;
    const auto& rebuilt_nstm = stm == WHITE ? black_rebuild : white_rebuild;
    const int rebuilt = forward<Layout>(pos, net, rebuilt_stm, rebuilt_nstm);

    const bool white_acc_equal = std::equal(
        set.states[current].accumulation[WHITE].begin(),
        set.states[current].accumulation[WHITE].end(),
        white_rebuild.begin()
    );
    const bool black_acc_equal = std::equal(
        set.states[current].accumulation[BLACK].begin(),
        set.states[current].accumulation[BLACK].end(),
        black_rebuild.begin()
    );
    const bool score_equal = incremental == rebuilt;
    const bool ok = score_equal && white_acc_equal && black_acc_equal;

    out << "mnuecheck " << label << " loaded 1"
        << " valid_before " << (valid_before ? 1 : 0)
        << " valid_mask_before " << static_cast<int>(valid_mask_before)
        << " generation_before " << generation_before
        << " generation_current " << g_p2_generation.load(std::memory_order_relaxed)
        << " stack_size " << set.state_count
        << " incremental " << incremental
        << " rebuild " << rebuilt
        << " score_equal " << (score_equal ? 1 : 0)
        << " white_acc_equal " << (white_acc_equal ? 1 : 0)
        << " black_acc_equal " << (black_acc_equal ? 1 : 0)
        << " ok " << (ok ? 1 : 0)
        << '\n';

    return ok;
}

bool debug_check_h32_incremental_impl(
    const Position& pos,
    H32StackStateSet& set,
    std::ostream& out
) noexcept {
    if (!g_p2h32.loaded || !g_p2h32.valid()) {
        out << "mnuecheck p2-h32 loaded 0\n";
        return false;
    }
    const u32 generation = g_p2_generation.load(std::memory_order_acquire);
    set.sync_generation(generation);
    const std::size_t current = set.state_count - 1;
    const auto& white = set.ensure(pos, WHITE, g_p2h32, generation);
    const auto& black = set.ensure(pos, BLACK, g_p2h32, generation);
    const Color stm = static_cast<Color>(pos.side_to_move);
    const int incremental = forward_h32(
        pos,
        g_p2h32,
        stm == WHITE ? white : black,
        stm == WHITE ? black : white
    );

    H32Accumulator white_rebuild{};
    H32Accumulator black_rebuild{};
    rebuild_h32_accumulator(pos, WHITE, white_rebuild, g_p2h32);
    rebuild_h32_accumulator(pos, BLACK, black_rebuild, g_p2h32);
    const int rebuilt = forward_h32(
        pos,
        g_p2h32,
        stm == WHITE ? white_rebuild : black_rebuild,
        stm == WHITE ? black_rebuild : white_rebuild
    );
    const bool white_equal = std::equal(
        set.states[current].accumulation[WHITE].begin(),
        set.states[current].accumulation[WHITE].end(),
        white_rebuild.begin()
    );
    const bool black_equal = std::equal(
        set.states[current].accumulation[BLACK].begin(),
        set.states[current].accumulation[BLACK].end(),
        black_rebuild.begin()
    );
    const bool ok = incremental == rebuilt && white_equal && black_equal;
    out << "mnuecheck p2-h32 loaded 1"
        << " stack_size " << set.state_count
        << " incremental " << incremental
        << " rebuild " << rebuilt
        << " score_equal " << (incremental == rebuilt ? 1 : 0)
        << " white_acc_equal " << (white_equal ? 1 : 0)
        << " black_acc_equal " << (black_equal ? 1 : 0)
        << " ok " << (ok ? 1 : 0)
        << '\n';
    return ok;
}

bool debug_check_p2_incremental(
    const Position& pos,
    P2AccumulatorStack& stack,
    std::ostream& out
) noexcept {
    switch (g_p2_active) {
        case P2ActiveKind::P2:
            return debug_check_p2_incremental_impl<P2Layout>(
                pos,
                stack.impl_->p2_state(),
                g_p2,
                "p2",
                out
            );
        case P2ActiveKind::P2K32:
            return debug_check_p2_incremental_impl<P2K32Layout>(
                pos,
                stack.impl_->p2k32_state(),
                g_p2k32,
                "p2-k32",
                out
            );
        case P2ActiveKind::P2H32:
            return debug_check_h32_incremental_impl(
                pos,
                stack.impl_->p2h32_state(),
                out
            );
        case P2ActiveKind::P2Pro:
            return debug_check_p2_incremental_impl<P2ProLayout>(
                pos,
                stack.impl_->p2pro_state(),
                g_p2pro,
                "p2pro",
                out
            );
        case P2ActiveKind::None:
            out << "mnuecheck p2 loaded 0\n";
            return false;
    }
    return false;
}

bool debug_check_p2_incremental(
    const Position& pos,
    std::ostream& out
) noexcept {
    P2AccumulatorStack stack{};
    stack.reset();
    return debug_check_p2_incremental(pos, stack, out);
}

int search_score(int v, const Position& pos) noexcept {
    return mnue_search_score(v, pos);
}

int material_units(const Position& pos) noexcept {
    return mnue_material_units(pos);
}

WinRateParams win_rate_params(const Position& pos) noexcept {
    return mnue_win_rate_params(pos);
}

int to_cp(int v, const Position& pos) noexcept {
    return mnue_to_cp(v, pos);
}

int win_rate_model(int v, const Position& pos) noexcept {
    return search_score_to_wdl(search_score(v, pos), pos).win;
}

int search_score_to_cp(int score, const Position& pos) noexcept {
    return to_cp(score, pos);
}

int search_score_to_winrate(int score, const Position& pos) noexcept {
    return search_score_to_wdl(score, pos).win;
}

WdlTriplet search_score_to_wdl(int score, const Position& pos) noexcept {
    return mnue_wdl_from_cp(search_score_to_cp(score, pos), pos);
}

} // namespace magnus::mnue
