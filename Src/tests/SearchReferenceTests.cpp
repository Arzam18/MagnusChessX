#include "Common.h"
#include "board/MoveGen.h"
#include "board/Position.h"
#include "search/History.h"
#include "search/Search.h"
#include "search/See.h"

#include <cstdlib>
#include <iostream>

namespace {

using namespace magnus;

[[noreturn]] void fail(const char* label) {
    std::cerr << "search reference test failed: " << label << '\n';
    std::exit(1);
}

void expect(bool condition, const char* label) {
    if (!condition)
        fail(label);
}

void test_plies_from_null() {
    memory::Memory mem{};
    memory::memory_init(mem, 1, 1, 1);

    Position pos{};
    set_start_position(pos);
    position_refresh_key(pos, mem.tables);
    expect(pos.plies_from_null == 0, "start/FEN initialization");
    const Key start_key = pos.key;

    const Move e2e4 = magnus::make_move(12, 28, MOVE_DOUBLE_PUSH);
    StateInfo e2e4_state{};
    make_move(pos, e2e4, mem.tables, e2e4_state);
    expect(pos.plies_from_null == 1, "pawn move does not reset");

    Position copied{};
    position_copy_without_accumulators(copied, pos);
    expect(copied.plies_from_null == 1, "position copy");

    StateInfo null_state{};
    make_null_move(pos, mem.tables, null_state);
    expect(pos.plies_from_null == 0, "null resets");
    unmake_null_move(pos, null_state);
    expect(pos.plies_from_null == 1, "null undo restores");

    const Move d7d5 = magnus::make_move(51, 35, MOVE_DOUBLE_PUSH);
    StateInfo d7d5_state{};
    make_move(pos, d7d5, mem.tables, d7d5_state);
    const Move e4d5 = magnus::make_move(28, 35, MOVE_CAPTURE);
    StateInfo capture_state{};
    make_move(pos, e4d5, mem.tables, capture_state);
    expect(pos.plies_from_null == 3, "capture does not reset");
    unmake_move(pos, e4d5, mem.tables, capture_state);
    expect(pos.plies_from_null == 2, "capture undo");
    unmake_move(pos, d7d5, mem.tables, d7d5_state);
    unmake_move(pos, e2e4, mem.tables, e2e4_state);
    expect(pos.plies_from_null == 0 && pos.key == start_key, "normal undo round trip");

    Position copied_move{};
    set_start_position(copied_move);
    position_refresh_key(copied_move, mem.tables);
    do_move_copy(copied_move, e2e4, mem.tables);
    expect(copied_move.plies_from_null == 1, "do_move_copy");
    position_clear(copied_move);
    expect(copied_move.plies_from_null == 0, "FEN clear path");

    memory::memory_free(mem);
}

void test_shuffling_truth_table() {
    Position pos{};
    pos.halfmove_clock = 10;
    pos.plies_from_null = 7;

    const Move move4 = magnus::make_move(0, 1, MOVE_QUIET);
    const Move move2 = magnus::make_move(1, 2, MOVE_QUIET);
    const Move move = magnus::make_move(2, 3, MOVE_QUIET);
    const auto shuffling = [&](Move candidate, int ply = 20) {
        return search::is_shuffling(candidate, pos, move2, move4, ply);
    };

    expect(shuffling(move), "two-leg return");
    expect(!shuffling(magnus::make_move(2, 3, MOVE_CAPTURE)), "capture stage");
    expect(!shuffling(magnus::make_move(2, 3, MOVE_PROMO_Q)), "queen promotion stage");
    expect(shuffling(magnus::make_move(2, 3, MOVE_PROMO_N)), "underpromotion not capture stage");

    pos.halfmove_clock = 9;
    expect(!shuffling(move), "rule50 threshold");
    pos.halfmove_clock = 10;
    pos.plies_from_null = 6;
    expect(!shuffling(move), "null distance threshold");
    pos.plies_from_null = 7;
    expect(!shuffling(move, 19), "ply threshold");
    expect(!shuffling(magnus::make_move(3, 4, MOVE_QUIET)), "return squares");
}

void test_search_move_classification() {
    const Move quiet = magnus::make_move(1, 18, MOVE_QUIET);
    const Move capture = magnus::make_move(1, 18, MOVE_CAPTURE);
    const Move castle_king_side = magnus::make_move(4, 7, MOVE_OO);
    const Move castle_queen_side = magnus::make_move(4, 0, MOVE_OOO);
    const Move promotion = magnus::make_move(48, 56, MOVE_PROMO_Q);

    expect(search::is_search_quiet(quiet), "ordinary quiet classification");
    expect(search::is_search_quiet(castle_king_side), "king-side castle is quiet");
    expect(search::is_search_quiet(castle_queen_side), "queen-side castle is quiet");
    expect(!search::is_search_quiet(capture), "capture is not quiet");
    expect(!search::is_search_quiet(promotion), "promotion is not quiet");
    expect(!search::is_search_quiet(Move(0)), "null move is not quiet");
}

void test_depth_transitions() {
    const int original_node_depth = 10;
    const int move_base_depth = original_node_depth - 1;
    const int extended_node_depth = original_node_depth + 1;

    expect(
        search::singular_child_depth(move_base_depth, 1) == 10,
        "current TT child depth is frozen"
    );
    expect(extended_node_depth - 1 == 10, "later move inherits +1");
    expect(
        search::depth_after_alpha_improvement(extended_node_depth, false) == 9,
        "alpha improvement applies -2"
    );
    expect(
        search::depth_after_alpha_improvement(14, false) == 14,
        "alpha depth upper boundary"
    );
    expect(
        search::fail_high_softbound(150, 100, 9) == 145,
        "fail-high softbound uses final depth"
    );
}

void test_tt_trust_required_gap() {
    constexpr int base_gap = 53;

    expect(search::tt_trust_required_gap(base_gap, 511) == 53, "+511 truncates to zero");
    expect(search::tt_trust_required_gap(base_gap, -511) == 53, "-511 truncates to zero");
    expect(search::tt_trust_required_gap(base_gap, 512) == 52, "+512 lowers gap by one");
    expect(search::tt_trust_required_gap(base_gap, -512) == 54, "-512 raises gap by one");
    expect(search::tt_trust_required_gap(base_gap, 4096) == 45, "+4096 reaches clamp");
    expect(search::tt_trust_required_gap(base_gap, -4096) == 61, "-4096 reaches clamp");
    expect(search::tt_trust_required_gap(base_gap, 8192) == 45, "+8192 stays clamped");
    expect(search::tt_trust_required_gap(base_gap, -8192) == 61, "-8192 stays clamped");
    expect(search::tt_trust_required_gap(1, 8192) == 1, "required gap floor");
}

void test_capture_threat_buckets() {
    const Move capture = magnus::make_move(18, 35, MOVE_CAPTURE); // c3d5

    expect(search::capture_threat_bucket(0ULL, capture) == 0, "capture threat bucket 0");
    if constexpr (search::SearchComponents::threat_capture_history) {
        expect(
            search::capture_threat_bucket(bb_of(18), capture) == 1,
            "capture threat bucket 1"
        );
        expect(
            search::capture_threat_bucket(bb_of(35), capture) == 2,
            "capture threat bucket 2"
        );
        expect(
            search::capture_threat_bucket(bb_of(18) | bb_of(35), capture) == 3,
            "capture threat bucket 3"
        );
    } else {
        expect(
            search::capture_threat_bucket(bb_of(18) | bb_of(35), capture) == 0,
            "disabled capture threat bucket"
        );
    }
}

void test_capture_threat_attack_map() {
    memory::Memory mem{};
    memory::memory_init(mem, 1, 1, 1);

    Position pos{};
    position_clear(pos);
    position_put_piece(pos, WHITE, KING, 4);    // e1
    position_put_piece(pos, WHITE, ROOK, 12);   // e2 pins the e7 knight
    position_put_piece(pos, BLACK, ROOK, 56);   // a8
    position_put_piece(pos, BLACK, PAWN, 40);   // a6 blocks the rook
    position_put_piece(pos, BLACK, PAWN, 43);   // d6
    position_put_piece(pos, BLACK, KNIGHT, 52); // e7, pinned to e8
    position_put_piece(pos, BLACK, KING, 60);   // e8
    pos.side_to_move = WHITE;

    GenInfo info{};
    init_gen_info(info, pos, mem);
    const Bitboard threats = capture_history_threats(info, pos, mem);

    expect((threats & bb_of(36)) != 0ULL, "capture threats include pawn attack");   // e5
    expect((threats & bb_of(37)) != 0ULL, "capture threats include pinned knight"); // f5
    expect((threats & bb_of(48)) != 0ULL, "capture threats include slider attack"); // a7
    expect((threats & bb_of(53)) != 0ULL, "capture threats include king attack");   // f7
    expect((threats & bb_of(32)) == 0ULL, "capture threats stop behind blocker");   // a5
    expect(info.capture_threats_valid, "capture threats cache becomes valid");

    memory::memory_free(mem);
}

void test_capture_history_bucket_updates() {
    Position pos{};
    position_clear(pos);
    position_put_piece(pos, WHITE, KING, 4);     // e1
    position_put_piece(pos, WHITE, KNIGHT, 18); // c3
    position_put_piece(pos, BLACK, BISHOP, 35); // d5
    position_put_piece(pos, BLACK, KING, 60);   // e8
    pos.side_to_move = WHITE;

    const Move capture = magnus::make_move(18, 35, MOVE_CAPTURE);
    search::HistoryTables history{};
    const Bitboard both_threatened = bb_of(18) | bb_of(35);
    history.bonus_capture_fast(pos, capture, 4, both_threatened);

    const int updated_bucket = search::capture_threat_bucket(both_threatened, capture);
    expect(
        history.capture.value[WHITE][KNIGHT][35][BISHOP][updated_bucket]
            == search::history_bonus(4),
        "capture history bonus updates selected bucket"
    );
    if constexpr (search::SearchComponents::threat_capture_history) {
        for (int bucket = 0; bucket < 4; ++bucket) {
            if (bucket != updated_bucket)
                expect(
                    history.capture.value[WHITE][KNIGHT][35][BISHOP][bucket] == 0,
                    "capture history leaves other buckets untouched"
                );
        }
    }

    const Bitboard from_threatened = bb_of(18);
    history.penalty_capture_fast(pos, capture, 2, from_threatened);
    const int penalized_bucket = search::capture_threat_bucket(from_threatened, capture);
    if constexpr (search::SearchComponents::threat_capture_history) {
        expect(
            history.capture.value[WHITE][KNIGHT][35][BISHOP][penalized_bucket]
                == -search::history_penalty(2),
            "capture history penalty updates selected bucket"
        );
    } else {
        expect(
            history.capture.value[WHITE][KNIGHT][35][BISHOP][0]
                < search::history_bonus(4),
            "single capture history bucket accepts penalty"
        );
    }

    history.clear();
    expect(
        history.capture.value[WHITE][KNIGHT][35][BISHOP][0] == 0,
        "capture history clear resets buckets"
    );
}

void test_special_capture_history_indices() {
    Position ep_pos{};
    position_clear(ep_pos);
    position_put_piece(ep_pos, WHITE, KING, 4);   // e1
    position_put_piece(ep_pos, WHITE, PAWN, 36); // e5
    position_put_piece(ep_pos, BLACK, PAWN, 35); // d5
    position_put_piece(ep_pos, BLACK, KING, 60); // e8
    ep_pos.side_to_move = WHITE;
    ep_pos.ep_sq = 43; // d6

    search::HistoryTables history{};
    const Move ep = magnus::make_move(36, 43, MOVE_EP);
    const Bitboard ep_target_threatened = bb_of(43);
    history.bonus_capture_fast(ep_pos, ep, 3, ep_target_threatened);
    const int ep_bucket = search::capture_threat_bucket(ep_target_threatened, ep);
    expect(
        history.capture.value[WHITE][PAWN][43][PAWN][ep_bucket]
            == search::history_bonus(3),
        "en passant uses target threat and captured pawn index"
    );

    Position promotion_pos{};
    position_clear(promotion_pos);
    position_put_piece(promotion_pos, WHITE, KING, 4);   // e1
    position_put_piece(promotion_pos, WHITE, PAWN, 54); // g7
    position_put_piece(promotion_pos, BLACK, KING, 56); // a8
    position_put_piece(promotion_pos, BLACK, ROOK, 63); // h8
    promotion_pos.side_to_move = WHITE;

    const Move promotion = magnus::make_move(54, 63, MOVE_CAP_PROMO_Q);
    const Bitboard promotion_from_threatened = bb_of(54);
    history.bonus_capture_fast(
        promotion_pos,
        promotion,
        5,
        promotion_from_threatened
    );
    const int promotion_bucket =
        search::capture_threat_bucket(promotion_from_threatened, promotion);
    expect(
        history.capture.value[WHITE][PAWN][63][ROOK][promotion_bucket]
            == search::history_bonus(5),
        "capture promotion uses pawn mover and captured piece index"
    );
}

void test_capture_threat_cache_survives_make_unmake() {
    memory::Memory mem{};
    memory::memory_init(mem, 1, 1, 1);

    Position pos{};
    position_clear(pos);
    position_put_piece(pos, WHITE, KING, 7);     // h1
    position_put_piece(pos, WHITE, BISHOP, 40); // a6 blocks the rook
    position_put_piece(pos, BLACK, KING, 63);   // h8
    position_put_piece(pos, BLACK, ROOK, 56);   // a8
    pos.side_to_move = WHITE;
    position_refresh_key(pos, mem.tables);

    GenInfo info{};
    init_gen_info(info, pos, mem);
    const Bitboard parent_threats = capture_history_threats(info, pos, mem);
    expect((parent_threats & bb_of(32)) == 0ULL, "parent threat map sees blocker"); // a5

    const Move unblock = magnus::make_move(40, 33, MOVE_QUIET); // a6b5
    StateInfo state{};
    make_move(pos, unblock, mem.tables, state);
    expect(
        (danger_bb(pos, mem, BLACK) & bb_of(32)) != 0ULL,
        "child board exposes rook attack"
    );
    expect(
        capture_history_threats(info, pos, mem) == parent_threats,
        "capture threat cache remains parent map after make"
    );

    unmake_move(pos, unblock, mem.tables, state);
    expect(
        capture_history_threats(info, pos, mem) == parent_threats,
        "capture threat cache remains parent map after unmake"
    );

    memory::memory_free(mem);
}

void test_see_ignores_pinned_recapture() {
    memory::Memory mem{};
    memory::memory_init(mem, 1, 1, 1);

    Position pos{};
    set_start_position(pos);
    position_refresh_key(pos, mem.tables);

    constexpr Move moves[] = {
        magnus::make_move(12, 28, MOVE_DOUBLE_PUSH), // e2e4
        magnus::make_move(51, 35, MOVE_DOUBLE_PUSH), // d7d5
        magnus::make_move(1, 18, MOVE_QUIET),        // b1c3
        magnus::make_move(57, 42, MOVE_QUIET),       // b8c6
        magnus::make_move(18, 12, MOVE_QUIET),       // c3e2
        magnus::make_move(35, 27, MOVE_QUIET),       // d5d4
        magnus::make_move(6, 21, MOVE_QUIET),        // g1f3
        magnus::make_move(52, 36, MOVE_DOUBLE_PUSH), // e7e5
        magnus::make_move(10, 18, MOVE_QUIET),       // c2c3
        magnus::make_move(54, 46, MOVE_QUIET),       // g7g6
        magnus::make_move(12, 29, MOVE_QUIET),       // e2f4
        magnus::make_move(58, 44, MOVE_QUIET),       // c8e6
        magnus::make_move(5, 33, MOVE_QUIET),        // f1b5
        magnus::make_move(27, 19, MOVE_QUIET)        // d4d3
    };
    for (Move move : moves) {
        StateInfo state{};
        make_move(pos, move, mem.tables, state);
    }

    const Move capture = magnus::make_move(21, 36, MOVE_CAPTURE); // f3e5
    expect(legal(pos, mem, capture), "SEE regression capture is legal");
    expect(
        search::see_value_fast(pos, mem, capture) == 100,
        "SEE excludes pinned c6 knight recapture"
    );
    expect(
        search::see_ge_fast(pos, mem, capture, 0),
        "SEE threshold excludes pinned c6 knight recapture"
    );

    memory::memory_free(mem);
}

void test_see_pinned_piece_still_guards_against_king() {
    memory::Memory mem{};
    memory::memory_init(mem, 1, 1, 1);

    Position pos{};
    position_clear(pos);
    position_put_piece(pos, WHITE, KING, 4);   // e1
    position_put_piece(pos, WHITE, ROOK, 5);   // f1, pinned by h1 queen
    position_put_piece(pos, WHITE, BISHOP, 46); // g6
    position_put_piece(pos, BLACK, QUEEN, 7);  // h1
    position_put_piece(pos, BLACK, PAWN, 53);  // f7
    position_put_piece(pos, BLACK, KING, 60);  // e8
    pos.side_to_move = WHITE;
    position_refresh_key(pos, mem.tables);

    const Move capture = magnus::make_move(46, 53, MOVE_CAPTURE); // g6f7+
    expect(legal(pos, mem, capture), "SEE guarded-king regression capture is legal");
    expect(
        search::see_value_fast(pos, mem, capture) == 100,
        "pinned rook still guards f7 from black king"
    );
    expect(
        search::see_ge_fast(pos, mem, capture, 0),
        "SEE threshold honors pinned attack for king legality"
    );

    memory::memory_free(mem);
}

void test_persistent_worker_slots() {
    Position pos{};
    pos.side_to_move = WHITE;
    search::SearchSessionState session{};
    session.ensure_workers(3);
    session.worker(2).update_trust(pos, 256);
    const int learned = session.worker(2).trust(pos);
    expect(learned > 0, "worker trust learns");

    const u64 first_sequence = session.begin_search(3);
    expect(first_sequence == 1, "search sequence starts at one");
    expect(session.worker(2).search_sequence == 1, "worker search sequence");
    session.ensure_workers(1);
    expect(session.worker(2).trust(pos) == learned, "disabled worker slot retained");

    session.new_game();
    expect(session.worker(2).trust(pos) == 0, "new game clears trust");
    expect(session.worker(2).game_epoch == 1, "new game advances epoch");
}

} // namespace

int main() {
    test_plies_from_null();
    test_shuffling_truth_table();
    test_search_move_classification();
    test_depth_transitions();
    test_tt_trust_required_gap();
    test_capture_threat_buckets();
    test_capture_threat_attack_map();
    test_capture_history_bucket_updates();
    test_special_capture_history_indices();
    test_capture_threat_cache_survives_make_unmake();
    test_see_ignores_pinned_recapture();
    test_see_pinned_piece_still_guards_against_king();
    test_persistent_worker_slots();
    std::cout << "search reference state and depth transitions ok\n";
    return 0;
}
