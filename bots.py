import random

from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move import Move

# Regulation F (gen9vgc2024regf) team in Showdown paste format
REGULATION_I_TEAM = """
Brute Bonnet @ Sitrus Berry
Ability: Protosynthesis
Level: 50
Tera Type: Water
EVs: 252 HP / 84 Def / 172 SpD
Sassy Nature
IVs: 0 Spe
- Seed Bomb
- Sucker Punch
- Spore
- Rage Powder

Lunala @ Power Herb
Ability: Shadow Shield
Level: 50
Tera Type: Water
EVs: 228 HP / 4 Def / 20 SpA / 4 SpD / 252 Spe
Timid Nature
- Moongeist Beam
- Meteor Beam
- Wide Guard
- Trick Room

Ursaluna @ Flame Orb
Ability: Guts
Level: 50
Tera Type: Ghost
EVs: 252 HP / 196 Atk / 60 SpD
Brave Nature
IVs: 0 Spe
- Headlong Rush
- Facade
- Earthquake
- Protect

Chi-Yu @ Choice Scarf
Ability: Beads of Ruin
Level: 50
Tera Type: Ghost
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Dark Pulse
- Heat Wave
- Overheat
- Snarl

Koraidon @ Life Orb
Ability: Orichalcum Pulse
Level: 50
Tera Type: Fire
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Close Combat
- Flare Blitz
- Flame Charge
- Protect

Flutter Mane @ Focus Sash
Ability: Protosynthesis
Level: 50
Tera Type: Stellar
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Moonblast
- Icy Wind
- Protect
"""

SPECIES_THREAT_MOVES: dict[str, list[str]] = {
    "brutebonnet": ["seedbomb", "suckerpunch", "spore", "ragepowder"],
    "lunala": ["moongeistbeam", "meteorbeam", "wideguard", "trickroom"],
    "ursaluna": ["headlongrush", "facade", "earthquake", "protect"],
    "chiyu": ["darkpulse", "heatwave", "overheat", "snarl"],
    "koraidon": ["closecombat", "flareblitz", "flamecharge", "protect"],
    "fluttermane": ["shadowball", "moonblast", "icywind", "protect"],
}


def get_board_snapshot(battle: AbstractBattle) -> dict:
    """
    Return a simple snapshot of the current board state from a poke-env DoubleBattle.
    Structure:
      - my_active: list of up to 2 dicts (one per active slot on our side)
      - opp_active: same for opponent side
      - field: weather / terrain / trick_room flags
      - my_bench: list of our non-fainted, non-active mons
    """

    def _pokemon_summary(p) -> dict | None:
        if p is None:
            return None
        if getattr(p, "fainted", False):
            # Still return something, but mark as fainted so callers can see it.
            fainted = True
        else:
            fainted = False

        # Types as simple lowercase strings
        types = []
        for t in getattr(p, "types", []) or []:
            try:
                types.append(t.name.lower() if hasattr(t, "name") else str(t).lower())
            except Exception:
                types.append(None)

        # Moves as simple dicts
        moves_list = []
        moves = getattr(p, "moves", {}) or {}
        for m in moves.values():
            try:
                move_type = getattr(m, "type", None)
                type_str = None
                if move_type is not None:
                    type_str = (
                        move_type.name.lower()
                        if hasattr(move_type, "name")
                        else str(move_type).lower()
                    )
                moves_list.append(
                    {
                        "id": getattr(m, "id", None),
                        "name": getattr(m, "name", None),
                        "base_power": getattr(m, "base_power", None),
                        "type": type_str,
                        "category": getattr(m, "category", None),
                        "target": getattr(m, "target", None),
                    }
                )
            except Exception:
                # If anything is off, skip this move rather than crashing
                continue

        return {
            "name": getattr(p, "species", None) or getattr(p, "name", None),
            "ident": getattr(p, "ident", None),
            "current_hp_fraction": getattr(p, "current_hp_fraction", None),
            "fainted": fainted,
            "types": types,
            "base_stats": getattr(p, "base_stats", None),
            "item": getattr(p, "item", None),
            "ability": getattr(p, "ability", None),
            "status": getattr(p, "status", None),
            "moves": moves_list,
        }

    # Active mons (our side and opponent side)
    try:
        my_active_raw = list(getattr(battle, "active_pokemon", []) or [])
    except Exception:
        my_active_raw = []

    try:
        opp_active_raw = list(getattr(battle, "opponent_active_pokemon", []) or [])
    except Exception:
        opp_active_raw = []

    my_active = [_pokemon_summary(p) for p in my_active_raw]
    # Ensure exactly 2 slots (pad with None)
    while len(my_active) < 2:
        my_active.append(None)

    opp_active = [_pokemon_summary(p) for p in opp_active_raw]
    while len(opp_active) < 2:
        opp_active.append(None)

    # Field info: weather, terrain, trick room
    weather = None
    try:
        if getattr(battle, "weather", None):
            weather = [str(w) for w in battle.weather]
    except Exception:
        weather = None

    terrain = None
    trick_room = False
    try:
        fields = getattr(battle, "fields", {}) or {}
        # poke-env exposes fields as a dict-like; stringify keys
        terrain = [str(k) for k in fields.keys()] if hasattr(fields, "keys") else None
        # Trick Room usually appears as a field/side condition; do a simple name check
        trick_room = any("trick" in str(k).lower() for k in (fields.keys() if hasattr(fields, "keys") else []))
    except Exception:
        terrain = None
        trick_room = False

    # Bench: our non-active, non-fainted mons
    my_bench = []
    try:
        # battle.team is our side's full roster
        for p in getattr(battle, "team", {}).values():
            if p is None:
                continue
            if getattr(p, "fainted", False):
                continue
            if getattr(p, "active", False):
                continue
            summary = _pokemon_summary(p)
            if summary is not None:
                my_bench.append(summary)
    except Exception:
        my_bench = []

    return {
        "my_active": my_active,
        "opp_active": opp_active,
        "field": {
            "weather": weather,
            "terrain": terrain,
            "trick_room": trick_room,
        },
        "my_bench": my_bench,
    }

class BaseDoubleBot(Player):
    """
    Shared doubles logic: greedy ``choose_move`` → ``_choose_doubles_move`` (damage-max heuristic),
    HP debug, no ally-targeting attacks. Subclass to override ``choose_move`` for other policies.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.won: bool | None = None
        # Stores previous opponent active HP fractions so we can print turn-to-turn deltas.
        self._last_opp_active_hp: dict[str, float] = {}

    def teampreview(self, battle: AbstractBattle) -> str:
        members = random.sample(range(1, 7), 4)
        return "/team " + "".join(str(m) for m in members)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        self._log_opponent_hp_delta(battle)
        order = self._choose_doubles_move(battle)
        self._last_opp_active_hp = self._opponent_active_hp_snapshot(battle)
        return order

    def _opponent_active_hp_snapshot(self, battle: AbstractBattle) -> dict[str, float]:
        snap: dict[str, float] = {}
        for i, opp in enumerate(getattr(battle, "opponent_active_pokemon", []) or []):
            if opp is None:
                continue
            ident = str(
                getattr(opp, "ident", None)
                or getattr(opp, "species", None)
                or getattr(opp, "name", None)
                or f"opp_slot_{i}"
            )
            hp_frac = float(getattr(opp, "current_hp_fraction", 0.0) or 0.0)
            snap[ident] = hp_frac
        return snap

    def _log_opponent_hp_delta(self, battle: AbstractBattle) -> None:
        # Debug: "[POST CMD HP]" = opponent active HP before vs after our last order resolved
        # (turn-to-turn delta from self._last_opp_active_hp snapshot).
        current = self._opponent_active_hp_snapshot(battle)
        if not self._last_opp_active_hp:
            return
        printed = False
        for ident, now_hp in current.items():
            if ident not in self._last_opp_active_hp:
                continue
            prev_hp = self._last_opp_active_hp[ident]
            if abs(prev_hp - now_hp) < 1e-9:
                continue
            delta_pct = (prev_hp - now_hp) * 100.0
            # print(
            #     f"[POST CMD HP] opponent={ident} "
            #     f"prev={prev_hp:.4f} now={now_hp:.4f} delta={delta_pct:+.2f}%"
            # )
            printed = True
        # if printed:
        #     print("[POST CMD HP] ---")

    def _is_immune(self, move: Move, defender) -> bool:
        """Hardcoded immunity check based on defender types."""
        move_type = str(getattr(move, "type", "") or "").lower()
        defender_types = {str(t).lower() for t in (getattr(defender, "types", []) or []) if t}

        immunities = {
            "normal": {"ghost"},
            "fighting": {"ghost"},
            "ghost": {"normal"},
            "electric": {"ground"},
            "ground": {"flying"},
            "poison": {"steel"},
            "dragon": {"fairy"},
        }

        blocked_by = immunities.get(move_type, set())
        return bool(blocked_by & defender_types)

    def _estimate_damage(self, move: Move, attacker, defender, battle: AbstractBattle) -> float:
        """Estimate damage as fraction of defender's max HP. Returns -1.0 if immune."""
        if move.base_power == 0:
            return -1.0
        try:
            if self._is_immune(move, defender):
                return -1.0
            eff = defender.damage_multiplier(move)
            if eff == 0:
                return -1.0

            # STAB
            move_type = getattr(move, "type", None)
            stab = 1.5 if any(str(move_type).lower() == str(t).lower() for t in attacker.types if t) else 1.0

            def _approx_non_hp_stat(base: int, level: int = 50, iv: int = 31, ev: int = 0) -> float:
                # Neutral-nature approximation when live calculated stats are unavailable.
                return float(((2 * base + iv + ev // 4) * level) // 100 + 5)

            def _resolved_stat(pokemon, key: str) -> float:
                # Prefer live stats from poke-env if available.
                for attr in ("stats", "_stats", "base_stats"):
                    stats_obj = getattr(pokemon, attr, None)
                    if isinstance(stats_obj, dict):
                        v = stats_obj.get(key, None)
                        if isinstance(v, (int, float)) and v > 0:
                            # base_stats are not battle stats; only use as fallback approximation.
                            if attr == "base_stats":
                                return _approx_non_hp_stat(int(v))
                            return float(v)
                return 1.0

            cat_s = str(getattr(move, "category", "")).lower()
            is_physical = "physical" in cat_s
            if is_physical:
                atk_key, def_key = "atk", "def"
            else:
                atk_key, def_key = "spa", "spd"

            atk = _resolved_stat(attacker, atk_key)
            def_ = max(1.0, _resolved_stat(defender, def_key))

            # --- STAT STAGES + BURN + SCREENS ---
            # Stage multipliers
            stage_mult = {
                -6: 0.25,
                -5: 0.2857,
                -4: 0.3333,
                -3: 0.4,
                -2: 0.5,
                -1: 0.6667,
                 0: 1.0,
                 1: 1.5,
                 2: 2.0,
                 3: 2.5,
                 4: 3.0,
                 5: 3.5,
                 6: 4.0,
            }

            # Attacker / defender boosts (clamped to [-6, 6])
            atk_stage = int((getattr(attacker, "boosts", {}) or {}).get(atk_key, 0))
            def_stage = int((getattr(defender, "boosts", {}) or {}).get(def_key, 0))
            atk_stage = max(-6, min(6, atk_stage))
            def_stage = max(-6, min(6, def_stage))

            atk *= stage_mult.get(atk_stage, 1.0)

            # Burn: physical moves only
            is_burned = False
            if is_physical:
                status = getattr(attacker, "status", None)
                if status is not None and str(status).upper().endswith("BRN"):
                    atk *= 0.5
                    is_burned = True

            def_ *= stage_mult.get(def_stage, 1.0)

            # Screens: based on which side defender is on
            reflect_on = False
            lightscreen_on = False
            try:
                from poke_env.battle.side_condition import SideCondition

                # Determine defender side
                if defender in (getattr(battle, "opponent_active_pokemon", []) or []):
                    side_conds = getattr(battle, "opponent_side_conditions", {}) or {}
                else:
                    side_conds = getattr(battle, "side_conditions", {}) or {}

                if is_physical and SideCondition.REFLECT in side_conds:
                    reflect_on = True
                if not is_physical and SideCondition.LIGHT_SCREEN in side_conds:
                    lightscreen_on = True
            except Exception:
                reflect_on = False
                lightscreen_on = False

            # Item modifiers
            item = (getattr(attacker, "item", "") or "").lower().replace(" ", "")
            if item == "lifeorb":
                atk *= 1.3
            elif item in {"choiceband", "choicespecs"}:
                atk *= 1.5

            # Gen 9-style base damage at level 50 using float division.
            base = (((2 * 50 / 5 + 2) * move.base_power * atk / def_) / 50) + 2
            damage = base * stab * eff

            # Spread move penalty in doubles.
            target_s = str(getattr(move, "target", "")).lower().replace("_", "")
            is_spread = ("alladjacentfoes" in target_s) or ("alladjacent" in target_s)
            if is_spread:
                damage *= 0.75

            # Apply screens at the very end (doubles: 0.5)
            if reflect_on or lightscreen_on:
                damage *= 0.5

            max_hp = getattr(defender, "max_hp", None) or 1
            return damage / max_hp

        except Exception:
            return -1.0

    def _score_move(self, damage_fraction: float, defender_hp_fraction: float) -> float:
        """
        Base scoring — raw damage fraction.
        Subclasses override this to add KO priority, utility weighting, etc.
        """
        if damage_fraction <= 0:
            return 0.0
        return damage_fraction

    def _score_utility_move(self, move, attacker, battle: AbstractBattle) -> float:
        """Base implementation — utility moves ignored. Subclasses override."""
        return -1.0

    def _score_spore_for_targeted_foe(
        self,
        move: Move,
        attacker,
        battle: AbstractBattle,
        defender,
        foe_slot: int,
    ) -> float:
        """
        Spore targets a specific foe slot (0 or 1); score that order using the intended target.
        Base: never rank Spore (subclasses may override).
        """
        return -1.0

    def _order_target_index(self, order: SingleBattleOrder) -> int | None:
        """Best-effort target parse: foes are 1/2, allies are -1/-2."""
        import re

        mt = getattr(order, "move_target", None)
        if isinstance(mt, int):
            return mt

        msg = getattr(order, "message", None)
        if callable(getattr(order, "message", None)):
            try:
                msg = order.message()
            except Exception:
                msg = None
        if not isinstance(msg, str) or not msg:
            to_msg = getattr(order, "to_showdown_message", None)
            if callable(to_msg):
                try:
                    msg = to_msg()
                except Exception:
                    msg = None
        if isinstance(msg, str) and msg:
            ints = re.findall(r"-?\d+", msg)
            for tok in reversed(ints):
                try:
                    v = int(tok)
                except Exception:
                    continue
                if v in (-2, -1, 1, 2):
                    return v
        return mt if isinstance(mt, int) else None

    def _is_single_target_ally_attack(self, order: SingleBattleOrder) -> bool:
        """
        UNDER NO CIRCUMSTANCES for single-target damaging orders:
        they must not target ally slots.
        """
        move = getattr(order, "order", None)
        if not isinstance(move, Move):
            return False
        if (getattr(move, "base_power", 0) or 0) == 0:
            return False
        target_s = str(getattr(move, "target", "")).lower().replace("_", "")
        # Spread/side targets are not single-target checks.
        if "alladjacent" in target_s or "all" == target_s:
            return False
        if "ally" in target_s:
            return True
        return self._order_target_index(order) in (-1, -2)

    def _best_order_for_slot(
        self,
        attacker,
        valid_orders,
        our_active,
        opp_active,
        battle: AbstractBattle,
    ) -> SingleBattleOrder | None:
        """
        Score actual order objects directly, so winner already carries the exact
        SingleBattleOrder to return (no reverse lookup by move+target needed).
        """
        if attacker is None or attacker.fainted:
            return None

        # One cache per slot scoring pass: same Spore + Showdown target is evaluated twice
        # (duplicate valid_orders); reuse score and avoid duplicate subclass side effects/logging.
        self._spore_score_round_cache: dict[int, float] = {}

        def opp_defender(opp_slot: int):
            if opp_slot == 0:
                return opp_active[0] if len(opp_active) > 0 else None
            return opp_active[1] if len(opp_active) > 1 else None

        def single_cell_score(move: Move, defender) -> float:
            if move.base_power == 0:
                return self._score_utility_move(move, attacker, battle)
            if defender is None or getattr(defender, "fainted", False):
                return -1.0
            if self._is_immune(move, defender):
                return -1.0
            d = float(self._estimate_damage(move, attacker, defender, battle))
            return -1.0 if d < 0 else self._score_move(d, getattr(defender, "current_hp_fraction", 1.0))

        def spread_move_score(move: Move) -> float:
            if move.base_power == 0:
                return -1.0
            total = 0.0
            for slot in (0, 1):
                defender = opp_defender(slot)
                if defender is None or getattr(defender, "fainted", False):
                    continue
                if self._is_immune(move, defender):
                    continue
                d = float(self._estimate_damage(move, attacker, defender, battle))
                if d < 0:
                    continue
                total += self._score_move(d, getattr(defender, "current_hp_fraction", 1.0))
            return total

        # Up to 4 distinct move ids in first-seen order.
        move_slot: dict[str, int] = {}
        moves_list: list[Move | None] = [None, None, None, None]
        next_slot = 0
        for o in valid_orders:
            m = getattr(o, "order", None)
            if not isinstance(m, Move):
                continue
            mid = str(getattr(m, "id", None) or str(m))
            if mid in move_slot:
                continue
            if next_slot >= 4:
                break
            move_slot[mid] = next_slot
            moves_list[next_slot] = m
            next_slot += 1

        # Scores and attached order objects.
        grid = [[-1.0, -1.0] for _ in range(4)]  # [move_slot][opp_slot]
        grid_orders: list[list[list[SingleBattleOrder]]] = [[[], []] for _ in range(4)]
        spread_scores = [-1.0, -1.0, -1.0, -1.0]
        spread_orders: list[list[SingleBattleOrder]] = [[], [], [], []]
        utility_scores = [-1.0, -1.0, -1.0, -1.0]
        utility_orders: list[list[SingleBattleOrder]] = [[], [], [], []]

        eps = 1e-12
        for o in valid_orders:
            move = getattr(o, "order", None)
            if not isinstance(move, Move):
                continue
            mid = str(getattr(move, "id", None) or str(move))
            if mid not in move_slot:
                continue
            mi = move_slot[mid]

            target_attr = getattr(move, "target", "")
            target_s = str(target_attr)
            target_s_norm = target_s.lower().replace("_", "")
            if self._is_single_target_ally_attack(o):
                continue
            is_spread = ("alladjacent" in target_s_norm) or ("alladjacentfoes" in target_s_norm)
            if is_spread:
                s = spread_move_score(move)
                if s < 0:
                    continue
                if s > spread_scores[mi] + eps:
                    spread_scores[mi] = s
                    spread_orders[mi] = [o]
                elif abs(s - spread_scores[mi]) <= eps:
                    spread_orders[mi].append(o)
                continue

            # Utility moves (base_power == 0, not spread) — score before target filter
            if move.base_power == 0:
                move_ui = str(getattr(move, "id", "") or "").lower().replace("-", "").replace(" ", "")
                if move_ui == "spore":
                    tidx = self._order_target_index(o)
                    if tidx not in (1, 2):
                        continue
                    foe_slot = tidx - 1
                    if foe_slot not in self._spore_score_round_cache:
                        defender = opp_defender(foe_slot)
                        self._spore_score_round_cache[foe_slot] = (
                            self._score_spore_for_targeted_foe(
                                move, attacker, battle, defender, foe_slot
                            )
                        )
                    s = self._spore_score_round_cache[foe_slot]
                else:
                    s = single_cell_score(move, None)
                if s < 0:
                    continue
                if s > utility_scores[mi] + eps:
                    utility_scores[mi] = s
                    utility_orders[mi] = [o]
                elif abs(s - utility_scores[mi]) <= eps:
                    utility_orders[mi].append(o)
                continue

            # Single-target: use the same numbering as Showdown targets.
            # target 1 -> opp slot 0, target 2 -> opp slot 1.
            tidx = self._order_target_index(o)
            if tidx not in (1, 2):
                continue
            oj = tidx - 1
            s = single_cell_score(move, opp_defender(oj))
            if s < 0:
                continue
            if s > grid[mi][oj] + eps:
                grid[mi][oj] = s
                grid_orders[mi][oj] = [o]
            elif abs(s - grid[mi][oj]) <= eps:
                grid_orders[mi][oj].append(o)

        singles = (
            grid[0][0],
            grid[0][1],
            grid[1][0],
            grid[1][1],
            grid[2][0],
            grid[2][1],
            grid[3][0],
            grid[3][1],
        )
        max_single = max(singles)
        max_spread = max(spread_scores)
        max_utility = max(utility_scores)
        max_any = max(max_single, max_spread, max_utility)

        all_negative = max_any < 0
        if all_negative:
            pool = []
            for o in valid_orders:
                move = getattr(o, "order", None)
                if not isinstance(move, Move):
                    continue
                if getattr(move, "base_power", 0) == 0:
                    continue
                if self._is_single_target_ally_attack(o):
                    continue
                pool.append(o)
            return random.choice(pool) if pool else None

        candidates: list[SingleBattleOrder] = []

        if max_spread >= max_any - eps:
            for mi in range(4):
                if spread_scores[mi] < 0:
                    continue
                if abs(spread_scores[mi] - max_spread) > eps:
                    continue
                candidates.extend(spread_orders[mi])

        if max_single >= max_any - eps:
            best_ij = [
                (0, 0), (0, 1), (1, 0), (1, 1),
                (2, 0), (2, 1), (3, 0), (3, 1),
            ]
            for mi, oj in best_ij:
                v = grid[mi][oj]
                if v < 0 or abs(v - max_single) > eps:
                    continue
                candidates.extend(grid_orders[mi][oj])

        # Never prefer status/support when nothing scores above zero (ties at 0 are noise).
        if max_utility > eps and max_utility >= max_any - eps:
            for mi in range(4):
                if utility_scores[mi] < 0:
                    continue
                if abs(utility_scores[mi] - max_utility) > eps:
                    continue
                candidates.extend(utility_orders[mi])

        candidates = list({id(o): o for o in candidates}.values())
        if candidates:
            return random.choice(candidates)

        pool = []
        for o in valid_orders:
            move = getattr(o, "order", None)
            if not isinstance(move, Move):
                continue
            if getattr(move, "base_power", 0) == 0:
                continue
            if self._is_single_target_ally_attack(o):
                continue
            pool.append(o)
        if pool:
            return random.choice(pool)
        return None

    def _choose_doubles_move(self, battle: AbstractBattle) -> BattleOrder:
        from poke_env.player.battle_order import DoubleBattleOrder, PassBattleOrder

        valid = battle.valid_orders

        # Debug board snapshot logging disabled.
        # try:
        #     snapshot = get_board_snapshot(battle)
        #     print("[BOARD SNAPSHOT]", snapshot)
        # except Exception as e:
        #     print("[BOARD SNAPSHOT ERROR]", repr(e))

        # Forced switch: pick healthiest distinct incoming mons for each forced slot.
        if any(battle.force_switch):
            def _is_switch_order(order: SingleBattleOrder) -> bool:
                incoming = getattr(order, "order", None)
                return (
                    incoming is not None
                    and hasattr(incoming, "current_hp_fraction")
                    and not getattr(incoming, "fainted", True)
                    and not getattr(incoming, "active", False)
                )

            def _switch_key(order: SingleBattleOrder):
                # Primary: showdown "switch <n>" token (matches server error).
                to_msg = getattr(order, "to_showdown_message", None)
                if callable(to_msg):
                    try:
                        msg = to_msg()
                    except Exception:
                        msg = None
                    if isinstance(msg, str) and msg:
                        import re
                        m = re.search(r"\bswitch\s+\d+\b", msg.strip().lower())
                        if m:
                            return m.group(0)

                msg_fn = getattr(order, "message", None)
                if callable(msg_fn):
                    try:
                        msg = msg_fn()
                    except Exception:
                        msg = None
                    if isinstance(msg, str) and msg:
                        import re
                        m = re.search(r"\bswitch\s+\d+\b", msg.strip().lower())
                        if m:
                            return m.group(0)

                # Fallback: incoming numeric positional attributes.
                incoming = getattr(order, "order", None)
                if incoming is None:
                    return id(order)
                for attr in ("pokemon_index", "position", "slot", "index"):
                    v = getattr(incoming, attr, None)
                    if isinstance(v, int):
                        return v
                return id(incoming)

            def _switch_hp(order: SingleBattleOrder) -> float:
                incoming = getattr(order, "order", None)
                if incoming is None:
                    return 0.0
                return float(getattr(incoming, "current_hp_fraction", 0.0) or 0.0)

            cand0 = valid[0]
            cand1 = valid[1]
            if len(battle.force_switch) > 0 and battle.force_switch[0]:
                cand0 = [o for o in cand0 if _is_switch_order(o)]
            if len(battle.force_switch) > 1 and battle.force_switch[1]:
                cand1 = [o for o in cand1 if _is_switch_order(o)]

            best_pair: tuple[SingleBattleOrder, SingleBattleOrder] | None = None
            best_score = -1.0

            for o0 in cand0:
                for o1 in cand1:
                    if _is_switch_order(o0) and _is_switch_order(o1):
                        if _switch_key(o0) == _switch_key(o1):
                            continue
                    score = _switch_hp(o0) + _switch_hp(o1)
                    if score > best_score:
                        best_score = score
                        best_pair = (o0, o1)
                    elif abs(score - best_score) <= 1e-12 and best_pair is not None:
                        if random.random() < 0.5:
                            best_pair = (o0, o1)

            if best_pair is not None:
                return DoubleBattleOrder(best_pair[0], best_pair[1])
            return self.choose_random_doubles_move(battle)

        if not valid or not valid[0] or not valid[1]:
            return Player.choose_default_move()

        our_active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon

        # Manually read current types for all visible Pokemon each turn (tracks tera changes).
        # This keeps our view of typing in sync with the live battle state.
        for side_mon in list(our_active) + list(opp_active):
            if side_mon is None or side_mon.fainted:
                continue
            _ = [str(t).lower() for t in getattr(side_mon, "types", []) if t]

        attacker0 = our_active[0] if len(our_active) > 0 else None
        attacker1 = our_active[1] if len(our_active) > 1 else None

        def _slot_inactive(idx: int) -> bool:
            if idx >= len(our_active):
                return True
            mon = our_active[idx]
            return mon is None or getattr(mon, "fainted", False)

        def _pass_order_for_slot(idx: int) -> SingleBattleOrder:
            """Legal no-op for a fainted / empty doubles slot (server expects two commands)."""
            for o in valid[idx]:
                if isinstance(o, PassBattleOrder):
                    return o
            return PassBattleOrder()

        order0 = self._best_order_for_slot(attacker0, valid[0], our_active, opp_active, battle)
        order1 = self._best_order_for_slot(attacker1, valid[1], our_active, opp_active, battle)


        # Endgame: one survivor vs two foes — still emit a DoubleBattleOrder(move, pass)
        # or (pass, move). Previously we required both orders non-None and fell through
        # to Player.choose_default_move(), ignoring the scored best move for the live mon.
        if _slot_inactive(0):
            order0 = _pass_order_for_slot(0)
        else:
            if order0 is None:
                pool0 = [
                    o
                    for o in valid[0]
                    if isinstance(getattr(o, "order", None), Move)
                    and getattr(o.order, "base_power", 0) > 0
                    and not self._is_single_target_ally_attack(o)
                ]
                if pool0:
                    order0 = random.choice(pool0)

        if _slot_inactive(1):
            order1 = _pass_order_for_slot(1)
        else:
            if order1 is None:
                pool1 = [
                    o
                    for o in valid[1]
                    if isinstance(getattr(o, "order", None), Move)
                    and getattr(o.order, "base_power", 0) > 0
                    and not self._is_single_target_ally_attack(o)
                ]
                if pool1:
                    order1 = random.choice(pool1)

        if order0 and order1:
            return DoubleBattleOrder(order0, order1)
        return Player.choose_default_move()

class ToddlerBot(BaseDoubleBot):
    pass


class RandomBot(Player):
    def teampreview(self, battle: AbstractBattle) -> str:
        # Regulation VGC uses a 6-mon set but only selects 4 for each battle.
        members = random.sample(range(1, 7), 4)
        return "/team " + "".join(str(m) for m in members)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        Random doubles bot that:
        - randomly picks among legal orders (including terastallize options when present)
        - only allows moves that target the opponent side (never ally-targeting moves)
        """
        return self._choose_doubles_move_enemy_targeting_random(battle)

    def _showdown_target_index(self, order: SingleBattleOrder) -> int | None:
        """Best-effort parse of the Showdown target index from this order (e.g. 1, 2, -1, -2)."""
        import re

        msg = getattr(order, "message", None)
        if callable(getattr(order, "message", None)):
            try:
                msg = order.message()
            except Exception:
                msg = None

        if not isinstance(msg, str) or not msg:
            to_msg = getattr(order, "to_showdown_message", None)
            if callable(to_msg):
                try:
                    msg = to_msg()
                except Exception:
                    msg = None

        if not isinstance(msg, str) or not msg:
            return getattr(order, "move_target", None)

        # Robust: target is the final integer token in the message (handles negatives).
        m = re.search(r"(-?\d+)\s*$", msg.strip())
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass

        return getattr(order, "move_target", None)

    def _is_enemy_targeting_move_order(self, order: SingleBattleOrder) -> bool:
        """
        Enforce the "no ally-attacks" rule:
        - allow switching / non-move orders
        - allow spread moves (including `allAdjacent`, which can hit both teams)
        - block *damaging* moves that explicitly target your partner side (ally target types or -1/-2)
        - otherwise allow moves (random bot should keep turn legality)
        """
        # Switch orders (non-Move) are allowed; switching doesn't target a pokemon side.
        if not isinstance(getattr(order, "order", None), Move):
            return True

        move = order.order
        target_type = str(getattr(move, "target", "") or "").lower()

        # Status moves don't "attack your partner pokemon".
        base_power = getattr(move, "base_power", 0) or 0
        if base_power == 0:
            return True

        # Explicit ally-targeting (most of these are support/heal, but keep it strict for "attacks").
        if "ally" in target_type:
            return False

        # If showdown index indicates you targeted your own active(s), block.
        tidx = self._showdown_target_index(order)
        if tidx in (-1, -2):
            return False

        # Allow spread moves (can hit both sides, but not "specifically target" ally by selection).
        if target_type in {"alladjacent", "alladjacentfoes"}:
            return True
        if "alladjacent" in target_type:
            return True

        # Foe targeting (single-target or randomfoe/selectedfoe)
        if "foe" in target_type:
            return True

        # Otherwise, if we have an explicit index, only allow enemy indices (1 or 2).
        if tidx is not None:
            return tidx in (1, 2)

        # Last resort: if we can't tell, allow.
        return True

    def _choose_doubles_move_enemy_targeting_random(self, battle: AbstractBattle) -> BattleOrder:
        from poke_env.player.battle_order import DoubleBattleOrder

        valid = battle.valid_orders
        if not valid or not valid[0] or not valid[1]:
            return self.choose_random_doubles_move(battle)

        def _incoming_key(order) -> str | None:
            """
            Species name is the only stable, unique identifier for a bench mon.
            All message/index parsing removed — it was producing inconsistent keys.
            """
            incoming = getattr(order, "order", None)
            if incoming is None or isinstance(incoming, Move):
                return None
            species = getattr(incoming, "species", None)
            if isinstance(species, str) and species:
                return species.lower()
            # Last resort: object id (only works if poke-env reuses same object)
            return str(id(incoming))

        def _is_switch_order(order) -> bool:
            incoming = getattr(order, "order", None)
            return incoming is not None and not isinstance(incoming, Move)

        def _available_switches(slot_idx: int, claimed_keys: set) -> list:
            result = []
            for o in valid[slot_idx]:
                if not _is_switch_order(o):
                    continue
                incoming = getattr(o, "order", None)
                if incoming is None:
                    continue
                if getattr(incoming, "fainted", True):
                    continue
                if getattr(incoming, "active", False):
                    continue
                key = _incoming_key(o)
                if key is None or key in claimed_keys:
                    continue
                result.append(o)
            return result

        # Pairwise forced-switch selection to avoid duplicate switch-in slot errors.
        force_switch = battle.force_switch

        def _is_safe_switch(order) -> bool:
            if not _is_switch_order(order):
                return False
            incoming = getattr(order, "order", None)
            return (
                incoming is not None
                and not getattr(incoming, "fainted", True)
                and not getattr(incoming, "active", False)
            )

        def _hp(order) -> float:
            incoming = getattr(order, "order", None)
            if incoming is None:
                return 0.0
            return float(getattr(incoming, "current_hp_fraction", 0.0) or 0.0)

        def _switch_key(order):
            # Parse exact "switch <n>" for dedupe.
            to_msg = getattr(order, "to_showdown_message", None)
            if callable(to_msg):
                try:
                    msg = to_msg()
                except Exception:
                    msg = None
                if isinstance(msg, str) and msg:
                    import re
                    m = re.search(r"\bswitch\s+\d+\b", msg.strip().lower())
                    if m:
                        return m.group(0)

            msg_fn = getattr(order, "message", None)
            if callable(msg_fn):
                try:
                    msg = msg_fn()
                except Exception:
                    msg = None
                if isinstance(msg, str) and msg:
                    import re
                    m = re.search(r"\bswitch\s+\d+\b", msg.strip().lower())
                    if m:
                        return m.group(0)

            # Fallback: incoming positional attributes.
            incoming = getattr(order, "order", None)
            if incoming is None:
                return id(order)
            for attr in ("pokemon_index", "position", "slot", "index"):
                v = getattr(incoming, attr, None)
                if isinstance(v, int):
                    return v
            ident = getattr(incoming, "ident", None)
            return str(ident).lower() if ident else id(incoming)

        cand0 = valid[0]
        cand1 = valid[1]
        if len(force_switch) > 0 and force_switch[0]:
            cand0 = [o for o in cand0 if _is_safe_switch(o)]
        if len(force_switch) > 1 and force_switch[1]:
            cand1 = [o for o in cand1 if _is_safe_switch(o)]

        best_pair = None
        best_score = -1.0
        for o0 in cand0:
            for o1 in cand1:
                if _is_safe_switch(o0) and _is_safe_switch(o1):
                    if _switch_key(o0) == _switch_key(o1):
                        continue
                score = _hp(o0) + _hp(o1)
                if score > best_score:
                    best_score = score
                    best_pair = (o0, o1)
                elif abs(score - best_score) <= 1e-12 and best_pair is not None:
                    if random.random() < 0.5:
                        best_pair = (o0, o1)

        if best_pair is not None:
            return DoubleBattleOrder(best_pair[0], best_pair[1])
        return self.choose_random_doubles_move(battle)
