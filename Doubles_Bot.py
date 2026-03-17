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
# Smogon chaos JSON for meta item/ability priors (run download_smogon_chaos.py to fetch)
CHAOS_JSON_FILENAME = "gen9vgc2026regi-1760.json"
CHAOS_JSON_PATH = Path(__file__).resolve().parent / CHAOS_JSON_FILENAME


class StatInference:
    """Load gen9vgc2026regi-1760 chaos JSON and map species -> top item / ability for meta priors."""

    def __init__(self, json_path: Path | None = None):
        self._data: dict = {}
        path = json_path or CHAOS_JSON_PATH
        if path.is_file():
            import json
            self._data = json.loads(path.read_text(encoding="utf-8")).get("data", {})
        self._species_key_cache: dict[str, str] = {}

    def _species_key(self, species: str | None) -> str | None:
        if not species:
            return None
        if species in self._species_key_cache:
            return self._species_key_cache[species]
        # Chaos JSON keys are Title Case, e.g. "Brute Bonnet", "Chi-Yu", "Flutter Mane"
        for key in self._data:
            if key.lower() == species.lower():
                self._species_key_cache[species] = key
                return key
        # Try with hyphen/space normalized
        normalized = species.replace("-", " ").strip()
        for key in self._data:
            if key.lower() == normalized.lower():
                self._species_key_cache[species] = key
                return key
        self._species_key_cache[species] = None
        return None

    def get_prior_item(self, species: str | None) -> str | None:
        """Most common item for this species in 1760 chaos; None if unknown or not in stats."""
        key = self._species_key(species)
        if not key:
            return None
        items = self._data.get(key, {}).get("Items") or {}
        if not items:
            return None
        best = max(items.items(), key=lambda x: x[1])
        return best[0] if best[0].lower() != "nothing" else None

    def get_prior_ability(self, species: str | None) -> str | None:
        """Most common ability for this species in 1760 chaos; None if unknown or not in stats."""
        key = self._species_key(species)
        if not key:
            return None
        abilities = self._data.get(key, {}).get("Abilities") or {}
        if not abilities:
            return None
        best = max(abilities.items(), key=lambda x: x[1])
        return best[0] or None


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
        # Mapping from item string -> numeric id for embed_battle
        self._item_id_map: dict[str, int] = {}
        self._ability_id_map: dict[str, int] = {}
        self._next_item_id: int = 1
        self._next_ability_id: int = 1
        self._stat_inference = StatInference()
        # Deterministic stat solver tracking (per-opponent mon)
        # key: mon.identifier or species-based key; value: dict with fields like max_hp_locked, last_hp
        self.opponent_data: dict[str, dict[str, int]] = {}
        self._load_value_model()

    def _get_item_id(self, item: str | None) -> float:
        """Map a revealed item string to a stable numeric id (0 if unknown)."""
        if not item:
            return 0.0
        if item not in self._item_id_map:
            self._item_id_map[item] = self._next_item_id
            self._next_item_id += 1
        return float(self._item_id_map[item])

    def _get_ability_id(self, ability: str | None) -> float:
        """Map an ability string to a stable numeric id (0 if unknown)."""
        if not ability:
            return 0.0
        if ability not in self._ability_id_map:
            self._ability_id_map[ability] = self._next_ability_id
            self._next_ability_id += 1
        return float(self._ability_id_map[ability])

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
        """Return a numerical score for the current battle state with emphasis on HP preservation and speed control.
        +100 per our Pokémon alive, -100 per opponent alive;
        +80 * (our total HP fraction), -60 * (opponent total HP fraction);
        +40 if Tailwind is active on our side;
        +40 if Trick Room is active and we are the slower team.
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

        # Simple heuristic for "we are slower" under Trick Room: compare average Spe base stat
        trick_room_active = Field.TRICK_ROOM in battle.fields
        if trick_room_active:
            from poke_env.stats import compute_raw_stats
            try:
                our_spe = [
                    compute_raw_stats(p.base_stats, 50, p._gen, getattr(p, "_nature", None))["spe"]
                    for p in battle.team.values()
                    if not p.fainted
                ]
                opp_spe = [
                    compute_raw_stats(p.base_stats, 50, p._gen, getattr(p, "_nature", None))["spe"]
                    for p in battle.opponent_team.values()
                    if not p.fainted
                ]
                we_are_slower = (sum(our_spe) / max(1, len(our_spe))) < (sum(opp_spe) / max(1, len(opp_spe)))
            except Exception:
                we_are_slower = False
        else:
            we_are_slower = False

        base_score = _score_from_state(our_alive, opp_alive, our_hp_pct, opp_hp_pct, tailwind_ours)
        # Reweight HP: applied via coefficients above, plus Trick Room bonus when in our favour
        if trick_room_active and we_are_slower:
            base_score += 40.0
        return base_score

    def embed_battle(
        self, battle: AbstractBattle, override_hp: tuple[float, float, float, float] | None = None
    ) -> np.ndarray:
        """Embed battle state as a single vector for ML. override_hp: optional (our0, our1, opp0, opp1) to use simulated HP."""
        # Update opponent_data with latest HP observations (foundation for exact-stat HP solver)
        for mon in battle.opponent_team.values():
            if mon is None:
                continue
            key = mon.species or ""
            info = self.opponent_data.setdefault(key, {})
            cur_hp = getattr(mon, "current_hp", None)
            if cur_hp is None:
                continue
            last_hp = info.get("last_hp")
            if last_hp is not None and cur_hp != last_hp:
                delta = cur_hp - last_hp
                if "max_hp_locked" not in info:
                    abs_delta = abs(delta)
                    if abs_delta > 0:
                        hp16 = abs_delta * 16
                        hp10 = abs_delta * 10
                        true_max = getattr(mon, "max_hp", 0)
                        cand = None
                        if true_max and abs(true_max - hp16) < abs(true_max - hp10):
                            cand = hp16
                        elif true_max and abs(true_max - hp10) < abs(true_max - hp16):
                            cand = hp10
                        else:
                            cand = hp16
                        if cand:
                            info["max_hp_locked"] = int(cand)
            info["last_hp"] = cur_hp

        # Normalize to 4 slots: [our0, our1, opp0, opp0]
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

        base_vec = np.concatenate(per_slot)

        # Opponent bench / revealed info: 6 mons * (item flag, ability flag, item id, ability id). When hidden, fill from 1760 meta.
        opp_mons = list(battle.opponent_team.values())
        while len(opp_mons) < 6:
            opp_mons.append(None)
        opp_placeholders = []
        for mon in opp_mons[:6]:
            if mon is None:
                item_flag = 0.0
                ability_flag = 0.0
                item_id = 0.0
                ability_id = 0.0
            else:
                item = getattr(mon, "item", None)
                ability = getattr(mon, "ability", None)
                species = getattr(mon, "species", None)
                item_flag = 1.0 if item else 0.0
                ability_flag = 1.0 if ability else 0.0
                if item:
                    item_id = self._get_item_id(item)
                else:
                    prior_item = self._stat_inference.get_prior_item(species)
                    item_id = self._get_item_id(prior_item) if prior_item else 0.0
                if ability:
                    ability_id = self._get_ability_id(ability)
                else:
                    prior_ability = self._stat_inference.get_prior_ability(species)
                    ability_id = self._get_ability_id(prior_ability) if prior_ability else 0.0
            opp_placeholders.append(
                np.array([item_flag, ability_flag, item_id, ability_id], dtype=np.float32)
            )

        opp_vec = np.concatenate(opp_placeholders) if opp_placeholders else np.zeros(24, dtype=np.float32)
        return np.concatenate([base_vec, opp_vec])

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

    def calculate_damage_range(
        self,
        power: int,
        attack: int,
        defense: int,
        modifier_numer: int = 1,
        modifier_denom: int = 1,
    ) -> list[int]:
        """Integer-precise Gen 9-style damage for level 50: returns damage for all 16 rolls (85..100)."""
        level = 50
        if power <= 0 or attack <= 0 or defense <= 0 or level <= 0:
            return [0] * 16

        # 1. Base damage (floors at each division, as in-game)
        # base = floor( floor( floor( ( (2*L/5 + 2) * A * P / D ) / 50 ) ) + 2 )
        step1 = (2 * level) // 5          # floor(2*L/5)
        step2 = step1 + 2                 # +2
        step3 = step2 * attack * power    # * A * P
        step4 = step3 // defense          # / D
        step5 = step4 // 50               # / 50
        base = step5 + 2                  # +2

        # 2. Combined modifier as a single rational multiplier (numer/denom)
        mod_numer = modifier_numer
        mod_denom = modifier_denom if modifier_denom > 0 else 1

        damages: list[int] = []
        for roll in range(85, 101):  # 85..100 inclusive
            # dmg = floor( floor( base * total_mod ) * roll / 100 )
            tmp = base * mod_numer // mod_denom
            dmg = (tmp * roll) // 100
            damages.append(max(1, int(dmg)))   # damage is at least 1

        return damages

    def get_integer_damage_range(
        self,
        attacker: "Pokemon | None",
        defender: "Pokemon | None",
        move: Move,
        battle: AbstractBattle,
    ) -> list[int]:
        """Integer-perfect damage range for this move (16 rolls 85..100) using A/D, type and simple weather."""
        if attacker is None or defender is None or attacker.fainted or defender.fainted:
            return [0] * 16
        if move.base_power <= 0:
            return [0] * 16
        try:
            from poke_env.stats import compute_raw_stats
        except Exception:
            return [0] * 16

        # Attack / Defense stats from base stats at level 50 (ignore boosts/nature for now if missing)
        try:
            atk_stats = compute_raw_stats(
                attacker.base_stats,
                50,
                attacker._gen,
                getattr(attacker, "_nature", None),
            )
            def_stats = compute_raw_stats(
                defender.base_stats,
                50,
                defender._gen,
                getattr(defender, "_nature", None),
            )
        except Exception:
            return [0] * 16

        atk_stat = atk_stats["atk" if move.category == "physical" else "spa"]
        def_stat = def_stats["def" if move.category == "physical" else "spd"]

        # Type effectiveness (map to rational)
        eff = defender.damage_multiplier(move)
        if eff == 0:
            return [0] * 16
        if eff == 0.25:
            eff_num, eff_den = 1, 4
        elif eff == 0.5:
            eff_num, eff_den = 1, 2
        elif eff == 1:
            eff_num, eff_den = 1, 1
        elif eff == 2:
            eff_num, eff_den = 2, 1
        elif eff == 4:
            eff_num, eff_den = 4, 1
        else:
            eff_num, eff_den = 1, 1

        # Simple weather modifier (only handle Sun for Fire and Rain for Water, as common cases)
        weather_num, weather_den = 1, 1
        try:
            mtype = getattr(move, "type", None)
            if Weather.SUNNYDAY in battle.weather and str(mtype).lower() == "fire":
                weather_num, weather_den = 3, 2
            elif Weather.RAINDANCE in battle.weather and str(mtype).lower() == "water":
                weather_num, weather_den = 3, 2
            elif Weather.RAINDANCE in battle.weather and str(mtype).lower() == "fire":
                weather_num, weather_den = 1, 2
        except Exception:
            pass

        # Combine modifiers
        mod_num = eff_num * weather_num
        mod_den = eff_den * weather_den
        if mod_den <= 0:
            mod_den = 1

        return self.calculate_damage_range(int(move.base_power), int(atk_stat), max(1, int(def_stat)), mod_num, mod_den)

    def _estimate_damage_fraction(
        self, move: Move, attacker: "Pokemon | None", target: "Pokemon | None", battle: AbstractBattle
    ) -> float:
        """Estimate damage fraction (0..1) using integer-perfect damage range with type & simple weather."""
        if target is None or target.fainted or attacker is None or attacker.fainted:
            return 0.0
        if move.base_power == 0:
            return 0.0
        # Immunity check: if target is immune, we treat damage as strictly 0
        try:
            if target.damage_multiplier(move) == 0:
                return 0.0
        except Exception:
            pass
        dmg_values = self.get_integer_damage_range(attacker, target, move, battle)
        max_dmg = max(dmg_values)
        max_hp = getattr(target, "max_hp", None) or 0
        if max_hp <= 0:
            return 0.0
        return float(max_dmg) / float(max_hp)

    def _estimate_opponent_damage_to_our(
        self, opp_pokemon: "Pokemon | None", our_pokemon: "Pokemon | None"
    ) -> float:
        """Estimate damage fraction (0..1) from an opponent mon to our mon using integer-perfect damage engine,
        with a safety margin so we play around slightly higher damage than base calc."""
        if not opp_pokemon or not our_pokemon or opp_pokemon.fainted or our_pokemon.fainted:
            return 0.0
        # Use a generic 90 BP STAB move into our primary defensive stat
        from poke_env.stats import compute_raw_stats
        try:
            atk_stats = compute_raw_stats(
                opp_pokemon.base_stats,
                50,
                opp_pokemon._gen,
                opp_pokemon._nature if hasattr(opp_pokemon, "_nature") else None,
            )
            def_stats = compute_raw_stats(
                our_pokemon.base_stats,
                50,
                our_pokemon._gen,
                our_pokemon._nature if hasattr(our_pokemon, "_nature") else None,
            )
        except Exception:
            return 0.0
        # Very rough: treat it as physical into Defense
        atk_stat = atk_stats["atk"]
        def_stat = def_stats["def"]
        dmg_values = self.calculate_damage_range(90, int(atk_stat), max(1, int(def_stat)), 1, 1)
        max_dmg = max(dmg_values)
        # Safety margin: assume they can do ~10% more than our base calculation (but not more than full HP)
        max_dmg = min(int(max_dmg * 1.1), getattr(our_pokemon, "max_hp", 0) or max_dmg)
        max_hp = getattr(our_pokemon, "max_hp", None) or 0
        if max_hp <= 0:
            return 0.0
        return float(max_dmg) / float(max_hp)

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
        our_active = battle.active_pokemon
        our_alive = sum(1 for p in battle.team.values() if not p.fainted)
        opp_alive = sum(1 for p in battle.opponent_team.values() if not p.fainted)
        initial_opp_alive = opp_alive
        initial_our_alive = our_alive
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

        REDIRECTOR_SPECIES = {"Indeedee-F", "Indeedee", "Amoonguss", "Clefairy", "Togekiss"}

        def _find_redirector_index() -> int:
            """Heuristic: if a known redirector is on the opponent's field, assume it may use Follow Me / Rage Powder."""
            for i, mon in enumerate(opp_active):
                if mon is not None and not mon.fainted and getattr(mon, "species", "") in REDIRECTOR_SPECIES:
                    return i
            return -1

        redirect_idx = _find_redirector_index()

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
            # If a redirector is present and this is a single-target move, force target to redirector
            move_target = getattr(move, "target", "")
            is_spread = move_target in {"allAdjacent", "allAdjacentFoes"}
            if redirect_idx != -1 and not is_spread:
                idx = redirect_idx
            if idx < 0 or idx >= len(opp_active) or not opp_active[idx] or opp_active[idx].fainted:
                return
            target = opp_active[idx]
            attacker = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
            dmg = self._estimate_damage_fraction(move, attacker, target, battle)
            dmg *= move.accuracy
            if ally_used_helping_hand[slot]:
                dmg *= self._HELPING_HAND_MULTIPLIER
            # Spread move damage is reduced to 75% in doubles
            if is_spread:
                dmg *= 0.75
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

        score = _score_from_state(our_alive, opp_alive, our_hp_pct, opp_hp_pct_sim, tailwind_ours)
        # Kill pressure bonus: reward turns where we KO at least one opponent
        if opp_alive < initial_opp_alive:
            score += 80.0
        # Heuristic guard: massive penalty if we lost one of our Pokémon during our own simulated turn
        if our_alive < initial_our_alive:
            score -= 500.0
        return score

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
        initial_opp_alive = opp_alive
        initial_our_alive = our_alive
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

        REDIRECTOR_SPECIES = {"Indeedee-F", "Indeedee", "Amoonguss", "Clefairy", "Togekiss"}

        def _find_redirector_index_opp() -> int:
            for i, mon in enumerate(opp_active):
                if mon is not None and not mon.fainted and getattr(mon, "species", "") in REDIRECTOR_SPECIES:
                    return i
            return -1

        redirect_idx_opp = _find_redirector_index_opp()

        # Simple speed check: who is faster on average this turn?
        try:
            from poke_env.stats import compute_raw_stats
            our_spe = [
                compute_raw_stats(p.base_stats, 50, p._gen, getattr(p, "_nature", None))["spe"]
                for p in our_active
                if p is not None and not p.fainted
            ]
            opp_spe = [
                compute_raw_stats(p.base_stats, 50, p._gen, getattr(p, "_nature", None))["spe"]
                for p in opp_active
                if p is not None and not p.fainted
            ]
            we_go_first = (sum(our_spe) / max(1, len(our_spe))) >= (sum(opp_spe) / max(1, len(opp_spe)))
        except Exception:
            we_go_first = True

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
            move_target = getattr(move, "target", "")
            is_spread = move_target in {"allAdjacent", "allAdjacentFoes"}
            if redirect_idx_opp != -1 and not is_spread:
                idx = redirect_idx_opp
            if idx < 0 or idx >= len(opp_active) or not opp_active[idx] or opp_active[idx].fainted:
                return
            target = opp_active[idx]
            attacker = our_active[slot] if slot < len(our_active) else None
            dmg = self._estimate_damage_fraction(move, attacker, target, battle)
            if ally_used_helping_hand[slot]:
                dmg *= self._HELPING_HAND_MULTIPLIER
            if is_spread:
                dmg *= 0.75
            opp_hp[idx] = max(0.0, opp_hp[idx] - dmg)
            if opp_hp[idx] <= 0:
                opp_alive -= 1

        # Depending on who is faster, have opponent act first or second
        if we_go_first:
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
        else:
            # Opponent hits first
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

            # Then our orders, assuming any survivors still act
            if our_active[0] is not None and not our_active[0].fainted:
                apply_our_order(first_order, 0, force_miss_0)
            if our_active[1] is not None and not our_active[1].fainted:
                apply_our_order(second_order, 1, force_miss_1)

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
        # Kill pressure bonus: +80 if we reduced opponent's number of Pokémon
        if opp_alive < initial_opp_alive:
            score += 80.0
        # Massive penalty if our own Pokémon count decreased during our simulated turn
        if our_alive < initial_our_alive:
            score -= 500.0
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

        def _is_single_target_enemy_move(order: SingleBattleOrder) -> bool:
            """True if this is a single-target attacking move we intend to send at the opponent."""
            if not isinstance(order.order, Move):
                return False
            move = order.order
            # Spread moves (e.g. Earthquake, Heat Wave) are allowed to hit both sides – we don't treat them as single-target.
            if move.target in {"allAdjacent", "allAdjacentFoes"}:
                return False
            # Status or non-damaging moves are not considered for strict enemy-only targeting.
            if move.base_power == 0:
                return False
            return True

        def _targets_own_side(order: SingleBattleOrder) -> bool:
            """Heuristic: treat any single-target move with move_target not 1 or 2 as aiming at our own side."""
            if not _is_single_target_enemy_move(order):
                return False
            tidx = order.move_target
            # 1 -> opponent_active_pokemon[0], 2 -> opponent_active_pokemon[1]; anything else we treat as invalid (own side)
            return tidx not in (1, 2)

        def _targets_immune_foe(order: SingleBattleOrder) -> bool:
            """Return True if this single-target attacking move would hit an immune opponent."""
            if not _is_single_target_enemy_move(order):
                return False
            tidx = order.move_target
            # 1 -> opp_active[0], 2 -> opp_active[1]
            if tidx == 1:
                idx = 0
            elif tidx == 2:
                idx = 1
            else:
                return False
            if idx >= len(battle.opponent_active_pokemon):
                return False
            defender = battle.opponent_active_pokemon[idx]
            if defender is None or defender.fainted:
                return False
            try:
                return defender.damage_multiplier(order.order) == 0
            except Exception:
                return False

        def _joint_orders():
            """Yield only joint (o1, o2) pairs that don't try illegal double-switches, self-targeting, or immune targets."""
            for o1, o2 in itertools.product(valid[0], valid[1]):
                is_move_1 = isinstance(o1.order, Move)
                is_move_2 = isinstance(o2.order, Move)
                # If both orders are non-move (i.e. switches) and target the same thing, skip them.
                if not is_move_1 and not is_move_2 and o1.order == o2.order:
                    continue
                # Strictly forbid single-target moves that point at our own side (bad targeting).
                if _targets_own_side(o1) or _targets_own_side(o2):
                    continue
                # Also forbid single-target attacks whose chosen target is immune.
                if _targets_immune_foe(o1) or _targets_immune_foe(o2):
                    continue
                yield o1, o2

        if self._value_model is not None and self._value_scaler is not None:
            # NN: pick (o1, o2) with highest worst-case win probability over opponent responses
            import torch
            best_prob = float("-inf")
            best_order = None
            with torch.no_grad():
                for o1, o2 in _joint_orders():
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
            for o1, o2 in _joint_orders():
                worst_score = min(
                    self._risk_adjusted_score(battle, o1, o2, t0, t1)
                    for t0, t1 in OPP_TARGETS
                )
                # Switching penalty: discourage switches unless they provide a clearly better board
                if (not isinstance(o1.order, Move)) or (not isinstance(o2.order, Move)):
                    worst_score -= 15.0
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