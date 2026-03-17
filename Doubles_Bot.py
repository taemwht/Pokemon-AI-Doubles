import asyncio
import itertools
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move import Move
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.teambuilder import ConstantTeambuilder
from poke_env import ServerConfiguration

# Regulation F (gen9vgc2024regf) team in Showdown paste format
REGULATION_F_TEAM = """
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

# For embed_battle: canonical order for one-hot and stats
STAT_ORDER = ["atk", "def", "spa", "spd", "spe"]
WEATHER_ORDER = list(Weather)
TERRAIN_FIELDS = [f for f in Field if f != Field.UNKNOWN and f.is_terrain]

from poke_env.battle.double_battle import DoubleBattle
if TYPE_CHECKING:
    from poke_env.battle.pokemon import Pokemon

# Value network (must match train_value_model.py)
VGC_VALUE_MODEL_PATH = Path(__file__).resolve().parent / "vgc_value_model.pth"
VGC_VALUE_SCALER_PATH = Path(__file__).resolve().parent / "vgc_value_scaler.pkl"


def _score_from_state(
    our_alive: int,
    opp_alive: int,
    our_hp_pct: float,
    opp_hp_pct: float,
    tailwind_ours: bool,
) -> float:
    """Raw heuristic score from state counts and HP (used by _evaluate_board and simulation)."""
    return (
        100 * our_alive
        - 100 * opp_alive
        + 50 * our_hp_pct
        - 50 * opp_hp_pct
        + (30 if tailwind_ours else 0)
    )


class SmartBot(Player):
    """Bot that uses a heuristic evaluation and simulates board score for move choice."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history: list[tuple[np.ndarray, int]] = []
        self.won: bool | None = None
        self._value_model = None
        self._value_scaler = None
        self._value_model_n_features = None
        self._load_value_model()

    def _load_value_model(self) -> None:
        """Load vgc_value_model.pth and vgc_value_scaler.pkl for NN move scoring if present."""
        if not VGC_VALUE_MODEL_PATH.is_file() or not VGC_VALUE_SCALER_PATH.is_file():
            return
        try:
            import joblib
            import torch

            class _ValueMLP(torch.nn.Module):
                def __init__(self, n: int):
                    super().__init__()
                    self.layers = torch.nn.Sequential(
                        torch.nn.Linear(n, 128),
                        torch.nn.ReLU(),
                        torch.nn.Linear(128, 64),
                        torch.nn.ReLU(),
                        torch.nn.Linear(64, 1),
                        torch.nn.Sigmoid(),
                    )

                def forward(self, x):
                    return self.layers(x).squeeze(-1)

            self._value_scaler = joblib.load(VGC_VALUE_SCALER_PATH)
            self._value_model_n_features = self._value_scaler.n_features_in_
            self._value_model = _ValueMLP(self._value_model_n_features)
            self._value_model.load_state_dict(torch.load(VGC_VALUE_MODEL_PATH, map_location="cpu", weights_only=True))
            self._value_model.eval()
        except Exception:
            self._value_model = None
            self._value_scaler = None
            self._value_model_n_features = None

    def teampreview(self, battle: AbstractBattle) -> str:
        """Pick 4 Pokémon at random and shuffle their order for varied lead scenarios in training."""
        # Slots 1–6; pick 4 at random, then shuffle so leads/back vary every battle
        members = random.sample(range(1, 7), 4)
        random.shuffle(members)

        our_team_list = getattr(battle, "teampreview_team", None) or list(battle.team.values())
        for idx in members:
            i = idx - 1
            if 0 <= i < len(our_team_list):
                our_team_list[i]._selected_in_teampreview = True
        return "/team " + "".join(str(m) for m in members)

    def _evaluate_board(self, battle: AbstractBattle) -> float:
        """Return a numerical score for the current battle state.
        +100 per our Pokémon alive, -100 per opponent alive;
        +50 * (our total HP fraction), -50 * (opponent total HP fraction);
        +30 if Tailwind is active on our side.
        """
        our_alive = sum(1 for p in battle.team.values() if not p.fainted)
        opp_alive = sum(1 for p in battle.opponent_team.values() if not p.fainted)
        our_hp_pct = sum(
            p.current_hp_fraction for p in battle.team.values() if not p.fainted
        )
        opp_hp_pct = sum(
            p.current_hp_fraction
            for p in battle.opponent_team.values()
            if not p.fainted
        )
        tailwind_ours = SideCondition.TAILWIND in battle.side_conditions
        return _score_from_state(our_alive, opp_alive, our_hp_pct, opp_hp_pct, tailwind_ours)

    def embed_battle(
        self, battle: AbstractBattle, override_hp: tuple[float, float, float, float] | None = None
    ) -> np.ndarray:
        """Embed battle state as a single vector for ML. override_hp: optional (our0, our1, opp0, opp1) to use simulated HP."""
        # Normalize to 4 slots: [our0, our1, opp0, opp1]
        our_active = list(battle.active_pokemon) if battle.active_pokemon else [None, None]
        opp_active = list(battle.opponent_active_pokemon) if battle.opponent_active_pokemon else [None, None]
        while len(our_active) < 2:
            our_active.append(None)
        while len(opp_active) < 2:
            opp_active.append(None)
        slots = [our_active[0], our_active[1], opp_active[0], opp_active[1]]

        # Global weather and terrain (one-hot each)
        weather_onehot = np.zeros(len(WEATHER_ORDER), dtype=np.float32)
        if battle.weather:
            for w in battle.weather:
                if w in WEATHER_ORDER:
                    weather_onehot[WEATHER_ORDER.index(w)] = 1.0
                    break
        terrain_onehot = np.zeros(len(TERRAIN_FIELDS), dtype=np.float32)
        if battle.fields:
            for f in battle.fields:
                if f in TERRAIN_FIELDS:
                    terrain_onehot[TERRAIN_FIELDS.index(f)] = 1.0
                    break

        # Explicit global effects for ML (e.g. Koraidon sun, Trick Room)
        sun_active = 1.0 if Weather.SUNNYDAY in battle.weather else 0.0
        trick_room_active = 1.0 if Field.TRICK_ROOM in battle.fields else 0.0
        global_effects = np.array([sun_active, trick_room_active], dtype=np.float32)

        per_slot = []
        for i, mon in enumerate(slots):
            if override_hp is not None and i < 4:
                hp_pct = float(override_hp[i])
            elif mon is None or mon.fainted:
                hp_pct = 0.0
            else:
                hp_pct = float(mon.current_hp_fraction)
            if mon is None or mon.fainted:
                boosts_norm = np.zeros(5, dtype=np.float32)
                tera = 0.0
            else:
                boosts = mon.boosts
                boosts_norm = np.array(
                    [(boosts.get(s, 0) + 6) / 12.0 for s in STAT_ORDER],
                    dtype=np.float32,
                )
                tera = 1.0 if mon.is_terastallized else 0.0
            per_slot.append(np.concatenate([
                np.array([hp_pct], dtype=np.float32),
                boosts_norm,
                weather_onehot,
                terrain_onehot,
                np.array([tera], dtype=np.float32),
                global_effects,
            ]))
        return np.concatenate(per_slot)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        # Data logging: embed current state and turn for ML
        vec = self.embed_battle(battle)
        self.history.append((vec, battle.turn))
        return self._choose_doubles_move(battle)

    def _choose_singles_move(self, battle: AbstractBattle) -> BattleOrder:
        # Treat switches as valid orders alongside moves: score each and pick best
        valid = battle.valid_orders
        if not valid:
            return self.choose_random_move(battle)
        opponent = battle.opponent_active_pokemon
        scores = []
        for order in valid:
            if isinstance(order.order, Move):
                if opponent is None or opponent.fainted:
                    score = 1.0
                else:
                    score = 50.0 * opponent.damage_multiplier(order.order)
            else:
                score = self._evaluate_board(battle)
            scores.append(score)
        best_idx = int(np.argmax(scores))
        return valid[best_idx]

    def _estimate_damage_fraction(
        self, move: Move, target: "Pokemon | None"
    ) -> float:
        """Rough estimate of damage as fraction of target's max HP (0..1)."""
        if target is None or target.fainted:
            return 0.0
        if move.base_power == 0:
            return 0.0
        effectiveness = target.damage_multiplier(move)
        # Heuristic: (base_power/100) * effectiveness * 0.5 caps around 0.5–1.0 for strong hits
        return min(1.0, (move.base_power / 100.0) * effectiveness * 0.5)

    def _estimate_opponent_damage_to_our(
        self, opp_pokemon: "Pokemon | None", our_pokemon: "Pokemon | None"
    ) -> float:
        """Estimate damage fraction (0..1) from an opponent mon to our mon (STAB, generic BP). Used for minimax."""
        if not opp_pokemon or not our_pokemon or opp_pokemon.fainted or our_pokemon.fainted:
            return 0.0
        effectiveness = our_pokemon.damage_multiplier(opp_pokemon.type_1)
        return min(1.0, (90 / 100.0) * effectiveness * 0.5)

    # Support moves that affect the ally's damage or board score (used in doubles simulation)
    _HELPING_HAND_MULTIPLIER = 1.5  # in-game: ally's move damage 1.5x
    # Risk assessment: blend of expected score and worst outcome (0 = risk-neutral, higher = more cautious)
    _RISK_AVERSION = 0.4

    def _is_support_move(self, order: SingleBattleOrder) -> bool:
        """True if this order is a support move (e.g. Helping Hand, Tailwind) that benefits the pair."""
        if not isinstance(order.order, Move):
            return False
        return order.order.id in {"helpinghand", "tailwind"}

    def _simulate_doubles_order(
        self,
        battle: "DoubleBattle",
        first_order: SingleBattleOrder,
        second_order: SingleBattleOrder,
    ) -> float:
        """Estimate board score after playing the given double battle order (full combination).
        Support moves are accounted for: Helping Hand boosts the ally's damage; Tailwind adds +30 to score.
        """
        from poke_env.battle.double_battle import DoubleBattle

        battle = battle  # type: DoubleBattle
        our_alive = sum(1 for p in battle.team.values() if not p.fainted)
        opp_alive = sum(1 for p in battle.opponent_team.values() if not p.fainted)
        our_hp_pct = sum(
            p.current_hp_fraction for p in battle.team.values() if not p.fainted
        )
        opp_active = battle.opponent_active_pokemon
        opp_hp = [
            opp_active[0].current_hp_fraction if opp_active[0] and not opp_active[0].fainted else 0.0,
            opp_active[1].current_hp_fraction if opp_active[1] and not opp_active[1].fainted else 0.0,
        ]
        opp_hp_pct_total = sum(
            p.current_hp_fraction for p in battle.opponent_team.values() if not p.fainted
        )
        tailwind_ours = SideCondition.TAILWIND in battle.side_conditions

        # If the *other* slot used Helping Hand, our damaging move gets 1.5x
        ally_used_helping_hand = [False, False]  # [for slot 0, for slot 1]
        if isinstance(second_order.order, Move) and second_order.order.id == "helpinghand":
            ally_used_helping_hand[0] = True  # slot 1 used HH → boost slot 0's damage
        if isinstance(first_order.order, Move) and first_order.order.id == "helpinghand":
            ally_used_helping_hand[1] = True  # slot 0 used HH → boost slot 1's damage

        def apply_order(order: SingleBattleOrder, slot: int) -> None:
            nonlocal opp_hp, opp_alive, tailwind_ours
            if not isinstance(order.order, Move):
                return
            move = order.order
            if move.id == "tailwind":
                tailwind_ours = True
            if move.base_power == 0:
                return
            tidx = order.move_target
            idx = 0 if tidx == 1 else (1 if tidx == 2 else -1)
            if idx < 0 or idx >= len(opp_active) or not opp_active[idx] or opp_active[idx].fainted:
                return
            target = opp_active[idx]
            dmg = self._estimate_damage_fraction(move, target)
            dmg *= move.accuracy
            if ally_used_helping_hand[slot]:
                dmg *= self._HELPING_HAND_MULTIPLIER
            opp_hp[idx] = max(0.0, opp_hp[idx] - dmg)
            if opp_hp[idx] <= 0:
                opp_alive -= 1

        apply_order(first_order, 0)
        apply_order(second_order, 1)

        # Simulated opponent HP: subtract current active contributions, add simulated
        active_contrib = 0.0
        for i, p in enumerate(opp_active):
            if p is not None and not p.fainted and i < 2:
                active_contrib += p.current_hp_fraction
        opp_hp_pct_sim = opp_hp_pct_total - active_contrib + opp_hp[0] + opp_hp[1]
        opp_hp_pct_sim = max(0.0, opp_hp_pct_sim)

        return _score_from_state(our_alive, opp_alive, our_hp_pct, opp_hp_pct_sim, tailwind_ours)

    def _simulate_full_turn(
        self,
        battle: "DoubleBattle",
        first_order: SingleBattleOrder,
        second_order: SingleBattleOrder,
        opp_target_0: int,
        opp_target_1: int,
        *,
        force_miss_0: bool = False,
        force_miss_1: bool = False,
    ) -> float:
        """Simulate our (o1, o2) then opponent targeting our slots; return board score (our perspective).
        opp_target_0/opp_target_1: which of our slots (0 or 1) each opponent mon targets.
        force_miss_0/force_miss_1: if True, that slot's move is treated as a complete miss (no damage, no effect).
        """
        from poke_env.battle.double_battle import DoubleBattle

        battle = battle  # type: DoubleBattle
        our_active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon
        our_alive = sum(1 for p in battle.team.values() if not p.fainted)
        opp_alive = sum(1 for p in battle.opponent_team.values() if not p.fainted)
        our_hp = [
            our_active[0].current_hp_fraction if our_active[0] and not our_active[0].fainted else 0.0,
            our_active[1].current_hp_fraction if our_active[1] and not our_active[1].fainted else 0.0,
        ]
        our_hp_pct_total = sum(
            p.current_hp_fraction for p in battle.team.values() if not p.fainted
        )
        opp_hp = [
            opp_active[0].current_hp_fraction if opp_active[0] and not opp_active[0].fainted else 0.0,
            opp_active[1].current_hp_fraction if opp_active[1] and not opp_active[1].fainted else 0.0,
        ]
        opp_hp_pct_total = sum(
            p.current_hp_fraction for p in battle.opponent_team.values() if not p.fainted
        )
        tailwind_ours = SideCondition.TAILWIND in battle.side_conditions
        # Helping Hand only counts if the ally's move did not "miss"
        ally_used_helping_hand = [
            isinstance(second_order.order, Move) and second_order.order.id == "helpinghand" and not force_miss_1,
            isinstance(first_order.order, Move) and first_order.order.id == "helpinghand" and not force_miss_0,
        ]

        def apply_our_order(order: SingleBattleOrder, slot: int, force_miss: bool) -> None:
            nonlocal opp_hp, opp_alive, tailwind_ours
            if force_miss:
                return
            if not isinstance(order.order, Move):
                return
            move = order.order
            if move.id == "tailwind":
                tailwind_ours = True
            if move.base_power == 0:
                return
            tidx = order.move_target
            idx = 0 if tidx == 1 else (1 if tidx == 2 else -1)
            if idx < 0 or idx >= len(opp_active) or not opp_active[idx] or opp_active[idx].fainted:
                return
            target = opp_active[idx]
            dmg = self._estimate_damage_fraction(move, target)
            if ally_used_helping_hand[slot]:
                dmg *= self._HELPING_HAND_MULTIPLIER
            opp_hp[idx] = max(0.0, opp_hp[idx] - dmg)
            if opp_hp[idx] <= 0:
                opp_alive -= 1

        apply_our_order(first_order, 0, force_miss_0)
        apply_our_order(second_order, 1, force_miss_1)

        # Opponent turn: each opp mon attacks one of our slots (worst case for us)
        for opp_slot, our_slot in [(0, opp_target_0), (1, opp_target_1)]:
            if opp_slot >= len(opp_active) or our_slot >= len(our_active):
                continue
            opp_mon = opp_active[opp_slot]
            our_mon = our_active[our_slot]
            if not opp_mon or not our_mon or opp_mon.fainted or our_mon.fainted:
                continue
            dmg = self._estimate_opponent_damage_to_our(opp_mon, our_mon)
            our_hp[our_slot] = max(0.0, our_hp[our_slot] - dmg)
            if our_hp[our_slot] <= 0:
                our_alive -= 1

        # Recompute our total HP% (replace active slots with simulated)
        active_contrib = sum(
            our_active[i].current_hp_fraction
            for i in (0, 1)
            if our_active[i] is not None and not our_active[i].fainted
        )
        our_hp_pct_sim = our_hp_pct_total - active_contrib + our_hp[0] + our_hp[1]
        our_hp_pct_sim = max(0.0, our_hp_pct_sim)
        active_contrib_opp = sum(
            opp_active[i].current_hp_fraction
            for i in (0, 1)
            if opp_active[i] is not None and not opp_active[i].fainted
        )
        opp_hp_pct_sim = opp_hp_pct_total - active_contrib_opp + opp_hp[0] + opp_hp[1]
        opp_hp_pct_sim = max(0.0, opp_hp_pct_sim)

        score = _score_from_state(our_alive, opp_alive, our_hp_pct_sim, opp_hp_pct_sim, tailwind_ours)
        return (score, our_hp, opp_hp)

    def _risk_adjusted_score(
        self,
        battle: "DoubleBattle",
        first_order: SingleBattleOrder,
        second_order: SingleBattleOrder,
        opp_target_0: int,
        opp_target_1: int,
    ) -> float:
        """Evaluate hit and miss futures for each of our moves; return a risk-averse blend.
        If missing leads to a much worse score (e.g. our Pokémon fainted), the bot prefers safer moves.
        """
        acc1 = (
            first_order.order.accuracy
            if isinstance(first_order.order, Move)
            else 1.0
        )
        acc2 = (
            second_order.order.accuracy
            if isinstance(second_order.order, Move)
            else 1.0
        )
        r_hit_hit = self._simulate_full_turn(
            battle, first_order, second_order, opp_target_0, opp_target_1,
            force_miss_0=False, force_miss_1=False,
        )
        r_hit_miss = self._simulate_full_turn(
            battle, first_order, second_order, opp_target_0, opp_target_1,
            force_miss_0=False, force_miss_1=True,
        )
        r_miss_hit = self._simulate_full_turn(
            battle, first_order, second_order, opp_target_0, opp_target_1,
            force_miss_0=True, force_miss_1=False,
        )
        r_miss_miss = self._simulate_full_turn(
            battle, first_order, second_order, opp_target_0, opp_target_1,
            force_miss_0=True, force_miss_1=True,
        )
        s_hit_hit, s_hit_miss = r_hit_hit[0], r_hit_miss[0]
        s_miss_hit, s_miss_miss = r_miss_hit[0], r_miss_miss[0]
        expected = (
            acc1 * acc2 * s_hit_hit
            + acc1 * (1 - acc2) * s_hit_miss
            + (1 - acc1) * acc2 * s_miss_hit
            + (1 - acc1) * (1 - acc2) * s_miss_miss
        )
        worst = min(s_hit_hit, s_hit_miss, s_miss_hit, s_miss_miss)
        return (1 - self._RISK_AVERSION) * expected + self._RISK_AVERSION * worst

    def _choose_doubles_move(self, battle: AbstractBattle) -> BattleOrder:
        from poke_env.battle.double_battle import DoubleBattle
        from poke_env.player.battle_order import (
            DoubleBattleOrder,
            PassBattleOrder,
            DefaultBattleOrder,
        )

        battle = battle  # type: DoubleBattle
        if any(battle.force_switch):
            return self.choose_random_doubles_move(battle)

        valid = battle.valid_orders
        if not valid or not valid[0] or not valid[1]:
            return self.choose_random_doubles_move(battle)

        OPP_TARGETS = [(0, 0), (0, 1), (1, 0), (1, 1)]

        if self._value_model is not None and self._value_scaler is not None:
            # NN: pick (o1, o2) with highest worst-case win probability over opponent responses
            import torch
            best_prob = float("-inf")
            best_order = None
            with torch.no_grad():
                for o1, o2 in itertools.product(valid[0], valid[1]):
                    probs = []
                    for t0, t1 in OPP_TARGETS:
                        _, our_hp, opp_hp = self._simulate_full_turn(
                            battle, o1, o2, t0, t1,
                            force_miss_0=False, force_miss_1=False,
                        )
                        emb = self.embed_battle(
                            battle,
                            override_hp=(our_hp[0], our_hp[1], opp_hp[0], opp_hp[1]),
                        )
                        if emb.shape[0] != self._value_model_n_features:
                            probs.append(0.5)
                            continue
                        X = self._value_scaler.transform(emb.reshape(1, -1)).astype("float32")
                        t = torch.tensor(X)
                        p = self._value_model(t).item()
                        probs.append(p)
                    worst_prob = min(probs)
                    if worst_prob > best_prob:
                        best_prob = worst_prob
                        best_order = (o1, o2)
        else:
            # Heuristic minimax + risk: pick order pair whose worst-case (after opponent best-response) is best
            best_worst_score = float("-inf")
            best_order = None
            for o1, o2 in itertools.product(valid[0], valid[1]):
                worst_score = min(
                    self._risk_adjusted_score(battle, o1, o2, t0, t1)
                    for t0, t1 in OPP_TARGETS
                )
                if worst_score > best_worst_score:
                    best_worst_score = worst_score
                    best_order = (o1, o2)

        if best_order is not None:
            return DoubleBattleOrder(best_order[0], best_order[1])
        return self.choose_random_doubles_move(battle)

    def save_data(self, filepath: str = "battle_history.npy") -> None:
        """Write all logged (embedding, turn) vectors from this battle to a .npy or .csv file, then clear history."""
        if not self.history:
            return
        turns = np.array([t for _, t in self.history], dtype=np.int32).reshape(-1, 1)
        embeddings = np.stack([v for v, _ in self.history], axis=0)
        data = np.hstack([turns, embeddings])
        if filepath.endswith(".npy"):
            np.save(filepath, data)
        else:
            if not filepath.endswith(".csv"):
                filepath = filepath + ".csv" if "." not in filepath else filepath
            n_features = embeddings.shape[1]
            header = "turn," + ",".join(f"f{i}" for i in range(n_features))
            np.savetxt(filepath, data, delimiter=",", header=header, comments="")
        self.history = []


async def main():
    # Connect to the stadium running in your other tab
    local_server = ServerConfiguration("ws://localhost:8000/showdown/websocket", "http://localhost:8000")

    # Regulation F format; both bots use the same competitive team (Showdown paste → ConstantTeambuilder)
    vgc_format = "gen9vgc2024regf"
    team = ConstantTeambuilder(REGULATION_F_TEAM)
    bot_1 = SmartBot(battle_format=vgc_format, server_configuration=local_server, team=team)
    bot_2 = RandomPlayer(battle_format=vgc_format, server_configuration=local_server, team=team)

    print("Running 10 battles and logging data...")
    for i in range(10):
        await bot_1.battle_against(bot_2, n_battles=1)
        # Set win/loss for NN training
        for b in bot_1.battles.values():
            if b.finished:
                bot_1.won = b.won
                break
        bot_1.save_data(f"battle_{i}.npy")
        print(f"Battle {i + 1}/10 done, won={bot_1.won}, data saved to battle_{i}.npy")
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())