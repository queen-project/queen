"""Parse model analyses and compare sanitized critical lines."""

from __future__ import annotations

import math
import re

import chess


ANALYSIS_PROMPT = (
    "Analyze the given chess position, identify the best move, give the critical "
    "line assuming best play, list promising moves for deeper investigation, and "
    "evaluate the position."
)
FIELDS = re.compile(
    r"(?ms)^(?P<name>ANALYSIS|BEST_MOVE|CRITICAL_LINE|PROMISING_MOVES|EVALUATION):"
    r"\s*(?P<body>.*?)(?=^(?:ANALYSIS|BEST_MOVE|CRITICAL_LINE|PROMISING_MOVES|EVALUATION):|\Z)"
)
MOVE = re.compile(
    r"<PIECE_(?P<mover_side>[MO])(?P<mover_piece>[PNBRQK])>\s*"
    r"<SQUARE_(?P<source>\d+)>\s*"
    r"(?:<PIECE_(?P<captured_side>[MO])(?P<captured_piece>[PNBRQK])>\s*)?"
    r"<SQUARE_(?P<target>\d+)>"
    r"(?:<PIECE_(?P<promotion_side>[MO])(?P<promotion_piece>[QRBN])>)?"
)
PIECES = {
    "P": chess.PAWN,
    "N": chess.KNIGHT,
    "B": chess.BISHOP,
    "R": chess.ROOK,
    "Q": chess.QUEEN,
    "K": chess.KING,
}
PROMOTIONS = {
    "Q": chess.QUEEN,
    "R": chess.ROOK,
    "B": chess.BISHOP,
    "N": chess.KNIGHT,
}
EVALUATION = re.compile(r"(?im)^EVALUATION:\s*(.+)$")
EXPLICIT_POV = re.compile(r"(?i)(?:from\s+)?<(WHITE|BLACK)>[’']?s?\s+perspective")
SIDE_FAVORED = (
    re.compile(r"(?i)<(WHITE|BLACK)>\s+(?:has|retains|holds)\b"),
    re.compile(r"(?i)<(WHITE|BLACK)>\s+is\s+(?:clearly\s+|much\s+)?better\b"),
    re.compile(r"(?i)(?:better|winning|decisive|advantage|favors?)\s+(?:for\s+)?<(WHITE|BLACK)>"),
)
# "<BLACK> is up approximately 2 pawns": the verb, not the number, carries the sign.
SIDE_VERB = re.compile(
    r"(?i)<(?P<side>WHITE|BLACK)>\s+is\s+(?:\w+\s+)?(?P<verb>up|ahead|better|winning|"
    r"down|behind|worse|losing)\b[^.]*?(?P<num>\d+(?:\.\d+)?)\s*pawns?"
)
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
MATE = re.compile(r"(?i)([-+])\s*M(?:ATE)?\s*(\d+)")


def field_map(text: str) -> dict[str, str]:
    return {
        match.group("name"): match.group("body").strip()
        for match in FIELDS.finditer(text)
    }


def expected_winrate(cp: int) -> float:
    exponent = max(-60.0, min(60.0, -0.00368208 * cp))
    return 1.0 / (1.0 + math.exp(exponent))


def pov_move(
    match: re.Match,
    board: chess.Board,
    pov: chess.Color,
) -> tuple[chess.Move | None, str | None]:
    """Decode one POV-token move and check it against the current board."""
    source_number = int(match.group("source")) - 1
    target_number = int(match.group("target")) - 1
    if not (0 <= source_number < 64 and 0 <= target_number < 64):
        return None, "square token is outside 1..64"
    source = source_number if pov == chess.WHITE else source_number ^ 56
    target = target_number if pov == chess.WHITE else target_number ^ 56
    mover_colour = pov if match.group("mover_side") == "M" else not pov
    piece = board.piece_at(source)
    if (
        piece is None
        or piece.color != mover_colour
        or piece.piece_type != PIECES[match.group("mover_piece")]
    ):
        return None, "mover token does not match the board"
    promotion = PROMOTIONS.get(match.group("promotion_piece"))
    candidate = chess.Move(source, target, promotion=promotion)
    if (
        promotion is None
        and piece.piece_type == chess.PAWN
        and chess.square_rank(target) in (0, 7)
    ):
        candidate = chess.Move(source, target, promotion=chess.QUEEN)
    if candidate not in board.legal_moves:
        return None, "move is illegal"

    actual = None
    if board.is_en_passant(candidate):
        actual = chess.Piece(chess.PAWN, not board.turn)
    elif board.is_capture(candidate):
        actual = board.piece_at(target)
    captured_piece = match.group("captured_piece")
    if actual is None and captured_piece:
        return None, "capture token on a non-capture"
    if actual is not None:
        if captured_piece is None:
            return None, "capture omitted its captured-piece token"
        captured_colour = pov if match.group("captured_side") == "M" else not pov
        if (
            actual.color != captured_colour
            or actual.piece_type != PIECES[captured_piece]
        ):
            return None, "captured-piece token does not match the board"
    return candidate, None


def legal_root_candidates(fen: str, generation: str) -> list[chess.Move]:
    """Read up to three distinct legal candidates in model preference order."""
    fields = field_map(generation)
    board = chess.Board(fen)
    pov = board.turn
    result = []
    for section in (fields.get("BEST_MOVE", ""), fields.get("PROMISING_MOVES", "")):
        for atom in MOVE.finditer(section):
            move, _ = pov_move(atom, board.copy(stack=False), pov)
            if move is not None and move not in result:
                result.append(move)
    for atom in MOVE.finditer(fields.get("ANALYSIS", "")):
        move, _ = pov_move(atom, board.copy(stack=False), pov)
        if move is not None and move not in result:
            result.append(move)
        if len(result) == 3:
            break
    return result[:3]


def legal_structured_best(fen: str, generation: str) -> chess.Move | None:
    board = chess.Board(fen)
    for atom in MOVE.finditer(field_map(generation).get("BEST_MOVE", "")):
        move, _ = pov_move(atom, board, board.turn)
        return move
    return None


def parse_model_evaluation(
    text: str,
    side: chess.Color,
) -> tuple[dict | None, str | None]:
    """Normalize a model evaluation to root-side win rate without exceptions."""
    match = EVALUATION.search(text)
    if match is None:
        return None, "missing EVALUATION field"
    value = match.group(1).strip()
    value = value.replace("<PLAYER>", "<WHITE>" if side else "<BLACK>")
    value = value.replace("<OPPONENT>", "<BLACK>" if side else "<WHITE>")
    value = re.sub(
        r"\b(?:the\s+)?player\b",
        "<WHITE>" if side else "<BLACK>",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\b(?:the\s+)?opponent\b",
        "<BLACK>" if side else "<WHITE>",
        value,
        flags=re.IGNORECASE,
    )

    # POV heuristic: explicit perspective wins, then "<side> is up/down N pawns",
    # then a named favored side, then White POV.
    perspective_match = EXPLICIT_POV.search(value)
    perspective = perspective_match.group(1).lower() if perspective_match else None
    equal = bool(re.search(r"(?i)\b(?:equal|balanced|drawish)\b", value))
    favored = None
    if not equal:
        for pattern in SIDE_FAVORED:
            favored_match = pattern.search(value)
            if favored_match:
                favored = favored_match.group(1).lower()
                break
    mate = MATE.search(value)
    number = NUMBER.search(value)
    verb = None if perspective or mate else SIDE_VERB.search(value)
    if verb:
        up = verb.group("verb").lower() in ("up", "ahead", "better", "winning")
        raw_sign = 1 if up else -1
        magnitude = float(verb.group("num"))
        kind = "pawn"
    elif mate:
        raw_sign = 1 if mate.group(1) == "+" else -1
        magnitude = float(int(mate.group(2)))
        kind = "mate"
    elif number:
        raw = float(number.group())
        raw_sign = 0 if raw == 0 else (1 if raw > 0 else -1)
        magnitude = abs(raw)
        kind = "pawn"
    else:
        return None, "evaluation contains no number"

    if perspective:
        white_sign = raw_sign if perspective == "white" else -raw_sign
        method = "explicit_perspective"
    elif verb:
        white_sign = raw_sign if verb.group("side").lower() == "white" else -raw_sign
        method = "side_verb"
    elif favored:
        numeric_pov = favored if raw_sign >= 0 else (
            "black" if favored == "white" else "white"
        )
        white_sign = raw_sign if numeric_pov == "white" else -raw_sign
        method = "named_favored_side"
    else:
        white_sign = raw_sign
        method = "white_pov_fallback"
    root_sign = white_sign if side == chess.WHITE else -white_sign
    if kind == "mate":
        winrate = 1.0 if root_sign > 0 else 0.0 if root_sign < 0 else 0.5
        display = f"{'+' if root_sign > 0 else '-'}M{int(magnitude)}"
        order = (root_sign, -root_sign * magnitude)
    else:
        cp = round(root_sign * magnitude * 100)
        winrate = expected_winrate(cp)
        display = f"{root_sign * magnitude:+.2f}"
        order = (0, root_sign * magnitude)
    return {
        "raw": value,
        "kind": kind,
        "method": method,
        "root_pov": display,
        "winrate": winrate,
        "order": order,
    }, None


def parse_critical(fen: str, generation: str) -> dict:
    """Replay the structured critical line until its first illegal move or terminal."""
    line = field_map(generation).get("CRITICAL_LINE", "")
    atoms = list(MOVE.finditer(line))
    board = chess.Board(fen)
    pov = board.turn
    steps = []
    illegal = None
    for index, atom in enumerate(atoms, 1):
        move, error = pov_move(atom, board, pov)
        if move is None:
            illegal = {
                "ply": index,
                "reason": "illegal",
                "detail": error,
                "token_text": atom.group(0),
            }
            break
        steps.append(
            {
                "ply": index,
                "fen": board.fen(),
                "uci": move.uci(),
                "token_text": atom.group(0),
            }
        )
        board.push(move)
        if board.is_game_over(claim_draw=False):
            break
    terminal = board.is_game_over(claim_draw=False)
    if not atoms and not terminal:
        illegal = {"ply": 1, "reason": "unparseable_or_empty"}
    return {
        "line": line,
        "parsed_move_count": len(atoms),
        "legal_steps": steps,
        "illegal_at": illegal,
        "terminal": terminal,
    }


def finish_critical(
    parsed: dict,
    audits: dict[tuple[str, str], dict],
    mistake_drop: float,
    min_clean_ply: int,
) -> dict:
    """Attach oracle scores and retain the clean prefix before a mistake or illegality."""
    plies = []
    first_mistake = None
    for step in parsed["legal_steps"]:
        row = {**step, **audits[(step["fen"], step["uci"])]}
        plies.append(row)
        if first_mistake is None and row["winrate_drop"] >= mistake_drop:
            first_mistake = {**row, "reason": "mistake"}
    cut = len(plies)
    illegal = parsed["illegal_at"]
    if illegal is not None:
        cut = min(cut, illegal["ply"] - 1)
    if first_mistake is not None:
        cut = min(cut, first_mistake["ply"] - 1)
    emitted = parsed["parsed_move_count"]
    return {
        "parsed_move_count": emitted,
        "accepted_ply": cut,
        "accepted_uci": [row["uci"] for row in plies[:cut]],
        "ply_audit": plies,
        "illegal_at": illegal,
        "first_mistake": first_mistake,
        "full_line_legal": illegal is None and (emitted > 0 or parsed["terminal"]),
        "valid_checked_prefix": (
            (emitted >= min_clean_ply and cut >= min_clean_ply)
            or (parsed["terminal"] and cut == len(plies))
        ),
    }


def parsed_sequence(parsed: dict) -> list[dict]:
    result = [{"uci": step["uci"], "legal": True} for step in parsed["legal_steps"]]
    if parsed["illegal_at"] is not None:
        result.append(
            {
                "uci": None,
                "legal": False,
                "token_text": parsed["illegal_at"].get("token_text"),
            }
        )
    return result


def first_line_divergence(fen: str, old: list[dict], new: list[dict]) -> dict:
    """Return the first distinct move while both lines share one position."""
    board = chess.Board(fen)
    for ply, (old_move, new_move) in enumerate(zip(old, new), 1):
        same = (
            old_move.get("uci") == new_move.get("uci")
            and old_move["legal"] == new_move["legal"]
        )
        if not same:
            return {
                "kind": "divergence",
                "ply": ply,
                "fen": board.fen(),
                "old": old_move,
                "new": new_move,
            }
        if not old_move["legal"]:
            return {
                "kind": "identical_illegal",
                "ply": ply,
                "old": old_move,
                "new": new_move,
            }
        board.push_uci(old_move["uci"])
    return {
        "kind": "no_divergence",
        "shared_ply": min(len(old), len(new)),
        "old_length": len(old),
        "new_length": len(new),
    }


def comparison_boundary(sanitation: dict) -> dict | None:
    """Represent the first removed move so truncation cannot hide a regression."""
    candidates = [
        row
        for row in (sanitation.get("illegal_at"), sanitation.get("first_mistake"))
        if row is not None
    ]
    if not candidates:
        return None
    boundary = min(candidates, key=lambda row: row["ply"])
    if boundary.get("reason") == "mistake":
        return {
            "uci": boundary["uci"],
            "legal": True,
            "rejected": "mistake",
            "winrate_drop": boundary.get("winrate_drop"),
        }
    return {
        "uci": None,
        "legal": False,
        "rejected": "illegal",
        "token_text": boundary.get("token_text"),
        "detail": boundary.get("detail"),
    }


def truncate_critical_line(generation: str, retained_ply: int) -> str:
    """Drop structured critical-line moves at and beyond the first bad ply."""
    critical = next(
        (match for match in FIELDS.finditer(generation) if match.group("name") == "CRITICAL_LINE"),
        None,
    )
    if critical is None:
        return generation
    atoms = list(MOVE.finditer(critical.group("body")))
    if retained_ply >= len(atoms):
        return generation
    retained = " ".join(atom.group(0) for atom in atoms[:retained_ply])
    start, end = critical.span("body")
    return generation[:start] + retained + "\n" + generation[end:]


def terminal_analysis(board: chess.Board) -> str:
    if board.is_checkmate():
        return "This position is checkmate."
    if board.is_stalemate():
        return "This position is stalemate."
    return "This position is drawn."


def terminal_evaluation(board: chess.Board) -> dict:
    if board.is_checkmate():
        return {
            "raw": "-M0 from the player's perspective.",
            "kind": "mate",
            "method": "terminal",
            "root_pov": "-M0",
            "winrate": 0.0,
            "order": (-1, 0),
        }
    return {
        "raw": "+0.00 from the player's perspective.",
        "kind": "pawn",
        "method": "terminal",
        "root_pov": "+0.00",
        "winrate": 0.5,
        "order": (0, 0.0),
    }


def terminal_attempt(board: chess.Board) -> dict:
    return {
        "attempt": 1,
        "generation": terminal_analysis(board),
        "token_count": 0,
        "finish_reason": "terminal",
        "evaluation": terminal_evaluation(board),
        "evaluation_error": None,
        "sanitation": {
            "parsed_move_count": 0,
            "accepted_ply": 0,
            "accepted_uci": [],
            "ply_audit": [],
            "illegal_at": None,
            "first_mistake": None,
            "full_line_legal": True,
            "valid_checked_prefix": True,
            "terminal": True,
        },
    }
