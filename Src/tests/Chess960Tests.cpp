#include "Common.h"
#include "board/Attack.h"
#include "board/MoveGen.h"
#include "board/Position.h"
#include "mnue/Mnue.h"
#include "search/Search.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string_view>

namespace {

using namespace magnus;

[[noreturn]] void fail(const char* label) {
    std::cerr << "chess960 test failed: " << label << '\n';
    std::exit(1);
}

void expect(bool condition, const char* label) {
    if (!condition)
        fail(label);
}

[[nodiscard]] bool positions_equal(
    const Position& lhs,
    const Position& rhs
) noexcept {
    return lhs.side_to_move == rhs.side_to_move &&
        lhs.ep_sq == rhs.ep_sq &&
        lhs.castling_rights == rhs.castling_rights &&
        lhs.chess960 == rhs.chess960 &&
        lhs.halfmove_clock == rhs.halfmove_clock &&
        lhs.fullmove_number == rhs.fullmove_number &&
        lhs.plies_from_null == rhs.plies_from_null &&
        std::equal(std::begin(lhs.king_sq), std::end(lhs.king_sq), std::begin(rhs.king_sq)) &&
        lhs.castling_rights_mask == rhs.castling_rights_mask &&
        lhs.castling_rook_square == rhs.castling_rook_square &&
        lhs.castling_path == rhs.castling_path &&
        std::equal(std::begin(lhs.color_bb), std::end(lhs.color_bb), std::begin(rhs.color_bb)) &&
        std::equal(std::begin(lhs.piece_bb), std::end(lhs.piece_bb), std::begin(rhs.piece_bb)) &&
        lhs.occupied == rhs.occupied &&
        lhs.piece_counts == rhs.piece_counts &&
        lhs.non_king_material == rhs.non_king_material &&
        lhs.mnue_phase_units == rhs.mnue_phase_units &&
        lhs.material_signature == rhs.material_signature &&
        lhs.key == rhs.key &&
        std::equal(std::begin(lhs.board), std::end(lhs.board), std::begin(rhs.board));
}

[[nodiscard]] u64 perft(
    Position& pos,
    const memory::Memory& mem,
    int depth
) noexcept {
    if (depth == 0)
        return 1;

    MoveList legal{};
    generate_legal(pos, mem, legal);
    if (depth == 1)
        return static_cast<u64>(legal.size);

    u64 nodes = 0;
    for (int i = 0; i < legal.size; ++i) {
        StateInfo state{};
        make_move(pos, legal.moves[i], mem.tables, state);
        nodes += perft(pos, mem, depth - 1);
        unmake_move(pos, legal.moves[i], mem.tables, state);
    }
    return nodes;
}

[[nodiscard]] Position parsed(
    const memory::Memory& mem,
    std::string_view fen,
    bool chess960 = true
) {
    Position pos{};
    expect(parse_fen(pos, mem, fen, chess960), "parse fixture");
    return pos;
}

[[nodiscard]] bool has_uci_move(
    const Position& pos,
    const memory::Memory& mem,
    std::string_view token
) noexcept {
    Move move = 0;
    return find_uci_move(pos, mem, token, move);
}

void play_uci_move(
    Position& pos,
    const memory::Memory& mem,
    std::string_view token
) {
    Move move = 0;
    expect(find_uci_move(pos, mem, token, move), "benchmark move parse");
    StateInfo state{};
    make_move(pos, move, mem.tables, state);
}

void test_fen_castling_metadata(const memory::Memory& mem) {
    Position shredder = parsed(
        mem,
        "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w HFhf - 0 1"
    );
    expect(shredder.chess960, "Shredder-FEN variant flag");
    expect(shredder.castling_rights == ANY_CASTLING, "Shredder-FEN rights");
    expect(shredder.castling_rook_square[WHITE_OO] == 7, "white H-file right");
    expect(shredder.castling_rook_square[WHITE_OOO] == 5, "white F-file right");
    expect(shredder.castling_rook_square[BLACK_OO] == 63, "black h-file right");
    expect(shredder.castling_rook_square[BLACK_OOO] == 61, "black f-file right");

    Position xfen = parsed(
        mem,
        "nqbnrkrb/pppppppp/8/8/8/8/PPPPPPPP/NQBNRKRB w KQkq - 0 1"
    );
    expect(xfen.castling_rook_square[WHITE_OO] == 6, "X-FEN white kingside");
    expect(xfen.castling_rook_square[WHITE_OOO] == 4, "X-FEN white queenside");
    expect(xfen.castling_rook_square[BLACK_OO] == 62, "X-FEN black kingside");
    expect(xfen.castling_rook_square[BLACK_OOO] == 60, "X-FEN black queenside");

    Position invalid{};
    expect(
        !parse_fen(invalid, mem, "4k3/8/8/8/8/8/8/4K3 w H - 0 1", true),
        "reject missing castling rook"
    );
    expect(
        !parse_fen(invalid, mem, "4k3/8/8/8/8/8/8/4K2R w HH - 0 1", true),
        "reject duplicate castling side"
    );
    expect(
        !parse_fen(invalid, mem, "4k3/8/8/8/8/8/8/R3K2R w -K - 0 1", true),
        "reject mixed dash castling field"
    );
}

void test_uci_notation(const memory::Memory& mem) {
    Position classic = parsed(
        mem,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        false
    );
    Move move = 0;
    expect(find_uci_move(classic, mem, "e1g1", move), "classic UCI castle input");
    expect(move_is_castle(move) && to_sq(move) == 7, "classic internal rook encoding");
    expect(search::move_to_uci(move, false) == "e1g1", "classic UCI castle output");
    expect(search::move_to_uci(move, true) == "e1h1", "Chess960 UCI castle output");

    Position chess960 = parsed(mem, "4k3/8/8/8/8/8/8/R5KR w AH - 0 1");
    expect(find_uci_move(chess960, mem, "g1h1", move), "stationary king castle input");
    expect(move_flag(move) == MOVE_OO && to_sq(move) == 7, "stationary king encoding");
    expect(!has_uci_move(chess960, mem, "g1g1"), "reject king-destination 960 notation");
    expect(has_uci_move(chess960, mem, "g1a1"), "Chess960 queenside notation");
}

struct CastleCase {
    const char* fen;
    const char* move;
    Square king_to;
    Square rook_to;
};

void test_castling_make_unmake(const memory::Memory& mem) {
    constexpr std::array<CastleCase, 6> cases{{
        {"4k3/8/8/8/8/8/8/R5KR w AH - 0 1", "g1h1", 6, 5},
        {"4k3/8/8/8/8/8/8/4KR2 w F - 0 1", "e1f1", 6, 5},
        {"4k3/8/8/8/8/8/8/4K1R1 w G - 0 1", "e1g1", 6, 5},
        {"4k3/8/8/8/8/8/8/5K1R w H - 0 1", "f1h1", 6, 5},
        {"4k3/8/8/8/8/8/8/2RK4 w C - 0 1", "d1c1", 2, 3},
        {"4k3/8/8/8/8/8/8/3RK3 w D - 0 1", "e1d1", 2, 3}
    }};

    const bool p2_ready = mnue::p2_loaded() ||
        (mnue::p2_embedded_available() && mnue::load_p2_embedded());
    expect(p2_ready, "embedded P2 available");

    for (const CastleCase& fixture : cases) {
        Position pos = parsed(mem, fixture.fen);
        const Position before = pos;
        Move move = 0;
        expect(find_uci_move(pos, mem, fixture.move, move), "edge castle generated");

        mnue::P2AccumulatorStack stack{};
        stack.reset();
        (void)mnue::eval_p2(pos, stack);
        stack.push(pos, move);

        StateInfo state{};
        make_move(pos, move, mem.tables, state);
        expect(piece_on(pos, fixture.king_to) == W_KING, "castle king destination");
        expect(piece_on(pos, fixture.rook_to) == W_ROOK, "castle rook destination");
        expect(pos.castling_rights == NO_CASTLING, "castle clears rights");
        expect(position_board_matches_bitboards(pos), "castle board consistency");
        expect(pos.key == position_compute_key(pos, mem.tables), "castle incremental key");
        (void)mnue::eval_p2(pos, stack);
        std::ostringstream diagnostic;
        expect(
            mnue::debug_check_p2_incremental(pos, stack, diagnostic),
            "castle P2 incremental rebuild"
        );
        stack.pop();

        unmake_move(pos, move, mem.tables, state);
        expect(positions_equal(pos, before), "castle make/unmake round trip");

        Position copied = before;
        do_move_copy(copied, move, mem.tables);
        StateInfo repeated_state{};
        Position repeated = before;
        make_move(repeated, move, mem.tables, repeated_state);
        expect(positions_equal(copied, repeated), "castle copy-make parity");
    }
}

void test_castling_legality(const memory::Memory& mem) {
    Position blocked = parsed(mem, "4k3/8/8/8/8/8/8/R1B1K2R w AH - 0 1");
    expect(has_uci_move(blocked, mem, "e1h1"), "unblocked kingside castle");
    expect(!has_uci_move(blocked, mem, "e1a1"), "blocked queenside castle");

    Position transit_attacked = parsed(mem, "4kr2/8/8/8/8/8/8/R3K2R w AH - 0 1");
    expect(!has_uci_move(transit_attacked, mem, "e1h1"), "attacked transit square");
    expect(has_uci_move(transit_attacked, mem, "e1a1"), "safe opposite castle");

    Position destination_attacked = parsed(mem, "4k1r1/8/8/8/8/8/8/R3K2R w AH - 0 1");
    expect(
        !has_uci_move(destination_attacked, mem, "e1h1"),
        "attacked destination square"
    );

    Position in_check = parsed(mem, "k3r3/8/8/8/8/8/8/R3K2R w AH - 0 1");
    expect(!has_uci_move(in_check, mem, "e1h1"), "no castle out of check kingside");
    expect(!has_uci_move(in_check, mem, "e1a1"), "no castle out of check queenside");

    Position rook_blocker = parsed(mem, "4k3/8/8/8/8/8/8/rR4K1 w B - 0 1");
    expect(!has_uci_move(rook_blocker, mem, "g1b1"), "rook move exposes slider check");
}

void test_castling_gives_check(const memory::Memory& mem) {
    Position king_side = parsed(mem, "5k2/8/8/8/8/8/8/4K2R w H - 0 1");
    Move move = 0;
    expect(find_uci_move(king_side, mem, "e1h1", move), "checking kingside castle");
    expect(move_gives_check(king_side, mem, move), "dynamic kingside castle gives check");

    Position queen_side = parsed(mem, "3k4/8/8/8/8/8/8/R5K1 w A - 0 1");
    expect(find_uci_move(queen_side, mem, "g1a1", move), "checking queenside castle");
    expect(move_gives_check(queen_side, mem, move), "dynamic queenside castle gives check");
}

void test_dynamic_rights_clearing(const memory::Memory& mem) {
    Position rook_move = parsed(mem, "4k3/8/8/8/8/8/8/1R2K1R1 w BG - 0 1");
    const Position before = rook_move;
    const Move g1g2 = make_move(6, 14, MOVE_QUIET);
    StateInfo state{};
    make_move(rook_move, g1g2, mem.tables, state);
    expect(rook_move.castling_rights == WHITE_OOO, "moving dynamic rook clears one right");
    unmake_move(rook_move, g1g2, mem.tables, state);
    expect(positions_equal(rook_move, before), "dynamic rook right undo");

    const Move king_move = make_move(4, 12, MOVE_QUIET);
    make_move(rook_move, king_move, mem.tables, state);
    expect(rook_move.castling_rights == NO_CASTLING, "king move clears both rights");
    unmake_move(rook_move, king_move, mem.tables, state);

    Position capture = parsed(mem, "1r2k3/8/8/8/8/8/8/1R2K1R1 b BG - 0 1");
    const Position capture_before = capture;
    const Move b8b1 = make_move(57, 1, MOVE_CAPTURE);
    make_move(capture, b8b1, mem.tables, state);
    expect(capture.castling_rights == WHITE_OO, "capturing dynamic rook clears right");
    expect(capture.key == position_compute_key(capture, mem.tables), "capture rights key");
    unmake_move(capture, b8b1, mem.tables, state);
    expect(positions_equal(capture, capture_before), "captured dynamic rook right undo");
}

void test_stockfish_reference_perft(const memory::Memory& mem) {
    Position start{};
    set_start_position(start);
    position_refresh_key(start, mem.tables);
    expect(perft(start, mem, 4) == 197281ULL, "standard start position perft d4");

    Position classic = parsed(
        mem,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        false
    );
    expect(perft(classic, mem, 4) == 314346ULL, "classic castling perft d4");

    Position stationary = parsed(mem, "4k3/8/8/8/8/8/8/R5KR w AH - 0 1");
    expect(perft(stationary, mem, 4) == 16219ULL, "stationary king perft d4");

    Position swap = parsed(mem, "4k3/8/8/8/8/8/8/2RK4 w C - 0 1");
    expect(perft(swap, mem, 4) == 6173ULL, "king rook swap perft d4");

    Position benchmark = parsed(
        mem,
        "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w HFhf - 0 1"
    );
    for (std::string_view move : {
            "g2g3", "d7d5", "d2d4", "c8h3",
            "c1g5", "e8d6", "g5e7", "f7f6"}) {
        play_uci_move(benchmark, mem, move);
    }
    expect(perft(benchmark, mem, 3) == 24386ULL, "Stockfish HFhf benchmark perft d3");

    Position xfen = parsed(
        mem,
        "nqbnrkrb/pppppppp/8/8/8/8/PPPPPPPP/NQBNRKRB w KQkq - 0 1"
    );
    expect(perft(xfen, mem, 3) == 8934ULL, "Stockfish X-FEN benchmark perft d3");
}

} // namespace

int main() {
    memory::Memory mem{};
    memory::memory_init(mem, 1, 1, 1);
    attack_init_backend(mem);

    test_fen_castling_metadata(mem);
    test_uci_notation(mem);
    test_castling_make_unmake(mem);
    test_castling_legality(mem);
    test_castling_gives_check(mem);
    test_dynamic_rights_clearing(mem);
    test_stockfish_reference_perft(mem);

    memory::memory_free(mem);
    std::cout << "chess960 compatibility tests ok\n";
    return 0;
}
