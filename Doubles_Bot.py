"""
AdolenceBot (upgraded policy / evaluation) + battle runner.

Debug AdolenceBot and match setup here. Shared doubles logic: ``bots.py``
(ToddlerBot, RandomBot, BaseDoubleBot, team paste, snapshots).
"""

from __future__ import annotations

import asyncio
import random
import string
from pathlib import Path

from poke_env import AccountConfiguration, LocalhostServerConfiguration, Player
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status
from poke_env.player.battle_order import (
    BattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.teambuilder import ConstantTeambuilder

from bots import REGULATION_I_TEAM, BaseDoubleBot, ToddlerBot

__all__ = ["AdolenceBot", "main"]


class AdolenceBot(BaseDoubleBot):
    """Base doubles tactics + position evaluation (for future search / upgrades)."""

    @staticmethod
    def _format_chosen_slot(order: SingleBattleOrder) -> str:
        inner = getattr(order, "order", None)
        if isinstance(order, PassBattleOrder):
            return "pass"
        if isinstance(inner, Move):
            mid = str(getattr(inner, "id", "?"))
            t = int(getattr(order, "move_target", 0) or 0)
            return f"{mid} {t}".strip() if t else mid
        if isinstance(inner, Pokemon):
            sp = str(getattr(inner, "species", "") or inner)
            return f"switch {sp}"
        if isinstance(inner, str):
            return inner
        return repr(inner)

    def _log_adolence_chosen_turn(self, battle: AbstractBattle, order: BattleOrder) -> None:
        tag = getattr(battle, "battle_tag", "?")
        lead = [str(getattr(m, "species", "?")) for m in (battle.active_pokemon or []) if m is not None]
        if isinstance(order, DoubleBattleOrder):
            a = self._format_chosen_slot(order.first_order)
            b = self._format_chosen_slot(order.second_order)
            print(f"[ADOLENCE CHOICE] {tag} active={lead} | {a} || {b}")
        else:
            msg = getattr(order, "message", None)
            line = msg if isinstance(msg, str) else str(order)
            print(f"[ADOLENCE CHOICE] {tag} active={lead} | {line}")

    def _log_adolence_switches_if_any(self, battle: AbstractBattle, order: BattleOrder) -> None:
        """One line per slot that is a switch, aligned with [ADOLENCE CHOICE] / [UTILITY] style."""
        if not isinstance(order, DoubleBattleOrder):
            return
        tag = getattr(battle, "battle_tag", "?")
        our = battle.active_pokemon or []
        for slot_idx, so in enumerate((order.first_order, order.second_order)):
            inner = getattr(so, "order", None)
            if not isinstance(inner, Pokemon):
                continue
            out_mon = our[slot_idx] if slot_idx < len(our) else None
            out_sp = str(getattr(out_mon, "species", "?") or "?").lower()
            in_sp = str(getattr(inner, "species", "?") or "?").lower()
            print(f"[ADOLENCE SWITCH] {tag} slot={slot_idx} out={out_sp} in={in_sp}")

    def _adolence_spore_memory_reset_if_new_battle(self, battle: AbstractBattle) -> None:
        tag = battle.battle_tag
        if getattr(self, "_adolence_spore_tag", None) != tag:
            self._adolence_spore_tag = tag
            self._adolence_spored_keys: set[tuple[int, str]] = set()
            self._adolence_prev_active_ids = None

    def _adolence_note_spore_after_choice(self, battle: AbstractBattle, order: BattleOrder) -> None:
        """Remember (foe slot, species) we attacked with Spore — do not re-rank Spore vs that pair."""
        if not isinstance(order, DoubleBattleOrder):
            return
        for so in (order.first_order, order.second_order):
            mov = getattr(so, "order", None)
            if not isinstance(mov, Move):
                continue
            if str(getattr(mov, "id", "")).lower() != "spore":
                continue
            tidx = self._order_target_index(so)
            if tidx not in (1, 2):
                continue
            foe_slot = tidx - 1
            opp_act = battle.opponent_active_pokemon or []
            if foe_slot >= len(opp_act):
                continue
            tgt = opp_act[foe_slot]
            if tgt is None:
                continue
            spec_key = (
                str(getattr(tgt, "species", "")).lower().replace(" ", "").replace("-", "")
            )
            self._adolence_spored_keys.add((foe_slot, spec_key))

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        self._log_opponent_hp_delta(battle)
        self._adolence_spore_memory_reset_if_new_battle(battle)

        # Temporary sanity check — remove after Step 1 is confirmed working
        score = self.evaluate_position(battle)
        print(f"[EVAL] position_score={score:.4f}")

        # Switch-in cooldown: mons whose object id first appeared among actives since last choice.
        prev = getattr(self, "_adolence_prev_active_ids", None)
        cur_ids = {id(m) for m in (battle.active_pokemon or []) if m is not None}
        if prev is not None:
            self._recently_switched_in = cur_ids - prev
        else:
            self._recently_switched_in = set()
        self._adolence_prev_active_ids = cur_ids

        order = self._choose_doubles_move(battle)
        self._adolence_note_spore_after_choice(battle, order)
        self._log_adolence_switches_if_any(battle, order)
        self._log_adolence_chosen_turn(battle, order)
        self._last_opp_active_hp = self._opponent_active_hp_snapshot(battle)
        return order

    def _get_opponent_active(self, battle: AbstractBattle) -> list:
        """
        Get opponent active mons from opponent_team.
        opponent_team tracks the opponent's mons as a separate object from ours.
        """
        result = []
        for mon in (battle.opponent_team.values() or []):
            if mon is None:
                continue
            if getattr(mon, "fainted", False):
                continue
            if not getattr(mon, "active", False):
                continue
            result.append(mon)
        return result

    def evaluate_position(self, battle: AbstractBattle) -> float:
        """
        Score the board from our perspective. Higher = better for us.
        Called both as a standalone heuristic AND by the search tree at leaf nodes.
        """
        score = 0.0

        # --- HP advantage ---
        for mon in (battle.active_pokemon or []):
            if mon and not getattr(mon, "fainted", False):
                score += mon.current_hp_fraction * 1.5

        for mon in battle.team.values():
            if not getattr(mon, "fainted", False) and not getattr(mon, "active", False):
                score += mon.current_hp_fraction * 0.8

        for mon in self._get_opponent_active(battle):
            if not getattr(mon, "fainted", False):
                score -= mon.current_hp_fraction * 1.5

        for mon in (battle.opponent_team.values() or []):
            if not getattr(mon, "fainted", False) and not getattr(mon, "active", False):
                score -= mon.current_hp_fraction * 0.8

        # --- Numbers advantage ---
        our_alive = sum(1 for m in battle.team.values() if not getattr(m, "fainted", False))
        opp_alive = sum(
            1
            for m in (battle.opponent_team.values() or [])
            if not getattr(m, "fainted", False)
        )
        score += (our_alive - opp_alive) * 1.0

        # --- Status advantage (use Status enum; str(status) is not a stable API) ---
        def _status_eval_adj(st) -> float:
            if st == Status.SLP or st == Status.FRZ:
                return 0.6
            if st in (Status.BRN, Status.PSN, Status.TOX, Status.PAR):
                return 0.2
            return 0.0

        for mon in self._get_opponent_active(battle):
            if mon is None:
                continue
            adj = _status_eval_adj(getattr(mon, "status", None))
            if adj:
                score += adj

        for mon in (battle.active_pokemon or []):
            if mon is None:
                continue
            adj = _status_eval_adj(getattr(mon, "status", None))
            if adj:
                score -= adj

        # --- Field conditions ---
        try:
            fields = getattr(battle, "fields", {}) or {}
            tr_active = any("trick" in str(k).lower() for k in fields.keys())
            if tr_active:
                score += 0.3 if self._our_team_wants_trick_room(battle) else -0.3
        except Exception:
            pass

        try:
            our_side = getattr(battle, "side_conditions", {}) or {}
            opp_side = getattr(battle, "opponent_side_conditions", {}) or {}
            if any("tailwind" in str(k).lower() for k in our_side.keys()):
                score += 0.25
            if any("tailwind" in str(k).lower() for k in opp_side.keys()):
                score -= 0.25
        except Exception:
            pass

        return score

    def _our_team_wants_trick_room(self, battle: AbstractBattle) -> bool:
        """
        True if our currently active mons benefit from Trick Room.
        Checks active slot first — if Ursaluna or another slow mon is
        active, TR is worth setting regardless of the full roster.
        """
        slow_threshold = 80  # raised from 60 — catches more mid-speed mons too

        # Check active mons first — if any active mon is slow, TR is worth it
        active_slow = 0
        active_total = 0
        for mon in (battle.active_pokemon or []):
            if mon is None or getattr(mon, "fainted", False):
                continue
            base_spe = (getattr(mon, "base_stats", {}) or {}).get("spe", 999)
            active_total += 1
            if base_spe <= slow_threshold:
                active_slow += 1

        if active_total > 0 and active_slow >= 1:
            return True  # at least one active mon wants TR

        # Fall back to bench check — if slow mons are waiting to come in
        bench_slow = 0
        bench_total = 0
        for mon in battle.team.values():
            if getattr(mon, "fainted", False) or getattr(mon, "active", False):
                continue
            base_spe = (getattr(mon, "base_stats", {}) or {}).get("spe", 999)
            bench_total += 1
            if base_spe <= slow_threshold:
                bench_slow += 1

        return bench_total > 0 and (bench_slow / bench_total) >= 0.5

    def _score_move(self, damage_fraction: float, defender_hp_fraction: float) -> float:
        """
        KO priority scoring — overrides base class raw damage scoring.
        A KO scores ~10x higher than the best non-KO move.
        """
        if damage_fraction <= 0:
            return 0.0

        if damage_fraction >= defender_hp_fraction:
            return 10.0 + min(damage_fraction, 1.0)

        damage_ratio = damage_fraction / defender_hp_fraction
        return damage_ratio * 2.0

    def _score_utility_move(self, move, attacker, battle: AbstractBattle) -> float:
        move_id = str(getattr(move, "id", "") or "").lower().replace("-", "").replace(" ", "")
        attacker_name = str(getattr(attacker, "species", None) or "?").lower()

        if move_id == "protect":
            facing_ko = self._facing_likely_ko(attacker, battle)
            attacker_hp = getattr(attacker, "current_hp_fraction", 1.0)
            if facing_ko:
                print(f"[UTILITY] protect={attacker_name} reason=facing_ko hp={attacker_hp:.2f} score=8.0")
                return 8.0
            if attacker_hp <= 0.4:
                print(f"[UTILITY] protect={attacker_name} reason=low_hp hp={attacker_hp:.2f} score=6.0")
                return 6.0
            if attacker_hp <= 0.6:
                print(f"[UTILITY] protect={attacker_name} reason=mid_hp hp={attacker_hp:.2f} score=2.0")
                return 2.0
            if self._should_stall(battle):
                print(f"[UTILITY] protect={attacker_name} reason=stall hp={attacker_hp:.2f} score=3.0")
                return 3.0
            print(f"[UTILITY] protect={attacker_name} reason=none hp={attacker_hp:.2f} score=-1.0")
            return -1.0

        if move_id in ("ragepowder", "followme"):
            partner = self._get_partner(attacker, battle)
            if partner is None:
                print(f"[UTILITY] ragepowder={attacker_name} no_partner score=-1.0")
                return -1.0
            partner_hp = getattr(partner, "current_hp_fraction", 0.0)
            attacker_hp = getattr(attacker, "current_hp_fraction", 0.0)
            if partner_hp > attacker_hp + 0.3:
                print(
                    f"[UTILITY] ragepowder={attacker_name} partner={getattr(partner, 'species', '?')} "
                    f"partner_hp={partner_hp:.2f} our_hp={attacker_hp:.2f} score=6.0"
                )
                return 6.0
            print(f"[UTILITY] ragepowder={attacker_name} partner_not_worth_protecting score=-1.0")
            return -1.0

        if move_id == "wideguard":
            has_spread = self._opponent_has_spread_moves(battle)
            print(
                f"[UTILITY] wideguard={attacker_name} opp_has_spread={has_spread} "
                f"score={'5.0' if has_spread else '-1.0'}"
            )
            if has_spread:
                return 5.0
            return -1.0

        if move_id == "trickroom":
            tr_active = any(
                "trick" in str(k).lower()
                for k in (getattr(battle, "fields", {}) or {}).keys()
            )
            wants_tr = self._our_team_wants_trick_room(battle)
            if tr_active:
                print(f"[UTILITY] trickroom={attacker_name} already_active score=-1.0")
                return -1.0
            if wants_tr:
                print(f"[UTILITY] trickroom={attacker_name} setting_tr score=8.0")
                return 8.0
            print(f"[UTILITY] trickroom={attacker_name} team_doesnt_want_tr score=-1.0")
            return -1.0

        if move_id == "tailwind":
            our_side = getattr(battle, "side_conditions", {}) or {}
            if any("tailwind" in str(k).lower() for k in our_side.keys()):
                print(f"[UTILITY] tailwind={attacker_name} already_active score=-1.0")
                return -1.0
            print(f"[UTILITY] tailwind={attacker_name} setting score=5.0")
            return 5.0

        if move_id in ("snarl", "icywind"):
            print(f"[UTILITY] debuff={move_id} attacker={attacker_name} score=1.5")
            return 1.5

        print(f"[UTILITY] unknown={move_id} attacker={attacker_name} score=-1.0")
        return -1.0

    def _score_spore_for_targeted_foe(
        self, move, attacker, battle: AbstractBattle, defender, foe_slot: int
    ) -> float:
        """
        Spore is scored per Showdown foe slot so each target gets a stable score.

        Sleep is detected with ``Status.SLP`` (not string hacks on the enum's ``__str__``).
        We also track (foe_slot, species) pairs we already committed Spore against this battle
        so we do not keep re-prioritizing Spore when ``status`` is laggy or inconsistent.
        """
        if defender is None or getattr(defender, "fainted", False):
            return -1.0
        attacker_name = str(getattr(attacker, "species", None) or "?").lower()
        # Never Spore our own active Pokémon (self or ally), even if targeting metadata glitches.
        for m in battle.active_pokemon or []:
            if m is None:
                continue
            if m is defender or id(m) == id(defender):
                print(f"[UTILITY] spore blocked own_side target={getattr(defender, 'species', '?')}")
                return -1.0
        opp_species = str(getattr(defender, "species", "") or "").lower()
        spec_key = opp_species.replace(" ", "").replace("-", "")
        if (foe_slot, spec_key) in getattr(self, "_adolence_spored_keys", set()):
            print(f"[UTILITY] spore blocked prior_attempt slot={foe_slot} species={opp_species}")
            return -1.0
        item = str(getattr(defender, "item", "") or "").lower().replace(" ", "")
        if item in ("lumberry", "safetygoggles"):
            print(f"[UTILITY] spore blocked by item={item} on {opp_species}")
            return -1.0
        st = getattr(defender, "status", None)
        if st == Status.SLP:
            print(f"[UTILITY] spore blocked already_asleep {opp_species}")
            return -1.0
        opp_types = [str(t).lower() for t in (getattr(defender, "types", []) or []) if t]
        if "grass" in opp_types:
            print(f"[UTILITY] spore blocked grass_immune {getattr(defender, 'species', '?')}")
            return -1.0
        print(f"[UTILITY] spore={attacker_name} target={opp_species} score=7.0")
        return 7.0

    def _facing_likely_ko(self, attacker, battle: AbstractBattle) -> bool:
        attacker_hp = getattr(attacker, "current_hp_fraction", 1.0)
        for opp in self._get_opponent_active(battle):
            if opp is None or getattr(opp, "fainted", False):
                continue
            for move in (getattr(opp, "moves", {}) or {}).values():
                if getattr(move, "base_power", 0) == 0:
                    continue
                dmg = self._estimate_damage(move, opp, attacker, battle)
                if dmg >= attacker_hp:
                    return True
        return False

    def _should_stall(self, battle: AbstractBattle) -> bool:
        try:
            fields = getattr(battle, "fields", {}) or {}
            for k, v in fields.items():
                k_str = str(k).lower()
                if "trick" in k_str and isinstance(v, int) and v <= 2:
                    if not self._our_team_wants_trick_room(battle):
                        return True
        except Exception:
            pass
        return False

    def _get_partner(self, attacker, battle: AbstractBattle) -> object | None:
        for mon in (battle.active_pokemon or []):
            if mon is None or mon is attacker:
                continue
            if not getattr(mon, "fainted", False):
                return mon
        return None

    def _opponent_has_spread_moves(self, battle: AbstractBattle) -> bool:
        spread_ids = {
            "earthquake", "rockslide", "discharge", "surf",
            "heatwave", "blizzard", "bulldoze", "glaciallance",
            "hyperdrill", "expandingforce"
        }
        for opp in self._get_opponent_active(battle):
            if opp is None:
                continue
            for move in (getattr(opp, "moves", {}) or {}).values():
                mid = str(getattr(move, "id", "") or "").lower().replace("-", "")
                if mid in spread_ids:
                    return True
        return False

    def _score_switch_in(self, candidate, battle: AbstractBattle) -> float:
        """
        Score a bench mon as a potential switch-in.
        Higher = better candidate to bring in right now.
        Returns negative if switching is not worth it.
        """
        if candidate is None or getattr(candidate, "fainted", False):
            return -999.0

        species = str(getattr(candidate, "species", "") or "").lower()
        hp = getattr(candidate, "current_hp_fraction", 1.0)

        score = 0.0

        # Penalise switching in a low-HP mon
        if hp < 0.3:
            return -999.0
        score += (hp - 0.5) * 2.0  # bonus for healthy mons, penalty below 50%

        opp_active = self._get_opponent_active(battle)

        # Type matchup: reward resisting or being immune to opponent moves,
        # penalise being weak to them
        for opp in opp_active:
            if opp is None:
                continue
            for move in (getattr(opp, "moves", {}) or {}).values():
                if getattr(move, "base_power", 0) == 0:
                    continue
                if self._is_immune(move, candidate):
                    score += 1.5
                    continue
                try:
                    eff = candidate.damage_multiplier(move)
                    if eff >= 2.0:
                        score -= 2.0
                    elif eff >= 1.5:
                        score -= 1.0
                    elif eff <= 0.5:
                        score += 1.0
                    elif eff == 0.0:
                        score += 1.5
                except Exception:
                    pass

        # Role bonuses: reward bringing in the right mon for the situation
        fields = getattr(battle, "fields", {}) or {}
        tr_active = any("trick" in str(k).lower() for k in fields.keys())

        # Lunala: reward if TR not up and we want to set it
        if "lunala" in species:
            if not tr_active and self._our_team_wants_trick_room(battle):
                score += 3.0

        # Brute Bonnet: reward if there's a valid Spore target
        if "brutebonnet" in species:
            for opp in opp_active:
                if opp is None or getattr(opp, "fainted", False):
                    continue
                st = getattr(opp, "status", None)
                if st == Status.SLP:
                    continue
                opp_types = [str(t).lower() for t in (getattr(opp, "types", []) or []) if t]
                if "grass" in opp_types:
                    continue
                item = str(getattr(opp, "item", "") or "").lower().replace(" ", "")
                if item in ("lumberry", "safetygoggles"):
                    continue
                score += 2.5  # valid spore target exists
                break

        # Ursaluna: reward under TR (benefits most from TR)
        if "ursaluna" in species:
            if tr_active:
                score += 2.0

        # Koraidon: reward in sun (Orichalcum Pulse)
        if "koraidon" in species:
            weather = getattr(battle, "weather", {}) or {}
            if any("sun" in str(w).lower() or "harsh" in str(w).lower() for w in weather):
                score += 1.5

        return score

    def _consider_voluntary_switch(
        self,
        attacker,
        slot_idx: int,
        valid_orders,
        battle: AbstractBattle,
    ) -> SingleBattleOrder | None:
        """
        Evaluate whether voluntarily switching `attacker` out is better than
        any move it can make. Returns a switch SingleBattleOrder if switching
        wins, otherwise None.

        Only switches if the best bench candidate scores meaningfully higher
        than staying in — uses a threshold to avoid pointless pivoting.
        """
        SWITCH_THRESHOLD = 7.0  # bench candidate must beat stay-in score by this much

        if attacker is None or getattr(attacker, "fainted", False):
            return None

        recently_switched = getattr(self, "_recently_switched_in", set())
        attacker_id = id(attacker)
        if attacker_id in recently_switched:
            return None

        # Get the best move score for staying in
        best_move_score = 0.0
        opp_active = battle.opponent_active_pokemon or []

        for o in valid_orders:
            move = getattr(o, "order", None)
            if not isinstance(move, Move):
                continue
            if getattr(move, "base_power", 0) == 0:
                continue
            if self._is_single_target_ally_attack(o):
                continue
            target_s = str(getattr(move, "target", "")).lower().replace("_", "")
            is_spread = "alladjacent" in target_s
            if is_spread:
                total = 0.0
                for opp in opp_active:
                    if opp is None or getattr(opp, "fainted", False):
                        continue
                    if self._is_immune(move, opp):
                        continue
                    d = self._estimate_damage(move, attacker, opp, battle)
                    if d >= 0:
                        total += self._score_move(d, getattr(opp, "current_hp_fraction", 1.0))
                best_move_score = max(best_move_score, total)
            else:
                tidx = self._order_target_index(o)
                if tidx not in (1, 2):
                    continue
                opp_slot = tidx - 1
                opp = opp_active[opp_slot] if opp_slot < len(opp_active) else None
                if opp is None or getattr(opp, "fainted", False):
                    continue
                if self._is_immune(move, opp):
                    continue
                d = self._estimate_damage(move, attacker, opp, battle)
                if d >= 0:
                    s = self._score_move(d, getattr(opp, "current_hp_fraction", 1.0))
                    best_move_score = max(best_move_score, s)

        # Don't pivot away from a good attacking position
        if best_move_score > 5.0:
            return None

        # Find the best switch-in candidate among valid switch orders for this slot
        best_switch_order = None
        best_switch_score = -999.0
        already_active_ids = {id(m) for m in (battle.active_pokemon or []) if m is not None}

        for o in valid_orders:
            candidate = getattr(o, "order", None)
            if candidate is None or isinstance(candidate, Move):
                continue
            if getattr(candidate, "fainted", True):
                continue
            if id(candidate) in already_active_ids:
                continue
            s = self._score_switch_in(candidate, battle)
            if s > best_switch_score:
                best_switch_score = s
                best_switch_order = o

        if best_switch_order is None:
            return None

        # Only switch if the candidate is clearly better than staying and attacking
        if best_switch_score > best_move_score + SWITCH_THRESHOLD:
            return best_switch_order

        return None

    def _choose_doubles_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        Override to inject voluntary switching before move scoring.
        Falls back to base class logic for all non-switch decisions.
        """
        if any(battle.force_switch):
            return super()._choose_doubles_move(battle)

        valid = battle.valid_orders
        if not valid or not valid[0] or not valid[1]:
            return Player.choose_default_move()

        our_active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon

        attacker0 = our_active[0] if len(our_active) > 0 else None
        attacker1 = our_active[1] if len(our_active) > 1 else None

        def _slot_inactive(idx: int) -> bool:
            if idx >= len(our_active):
                return True
            mon = our_active[idx]
            return mon is None or getattr(mon, "fainted", False)

        def _pass_order_for_slot(idx: int) -> SingleBattleOrder:
            for o in valid[idx]:
                if isinstance(o, PassBattleOrder):
                    return o
            return PassBattleOrder()

        # Try voluntary switch first, fall back to best move order
        if not _slot_inactive(0):
            order0 = self._consider_voluntary_switch(attacker0, 0, valid[0], battle)
            if order0 is None:
                order0 = self._best_order_for_slot(
                    attacker0, valid[0], our_active, opp_active, battle
                )
        else:
            order0 = _pass_order_for_slot(0)

        if not _slot_inactive(1):
            order1 = self._consider_voluntary_switch(attacker1, 1, valid[1], battle)
            if order1 is None:
                order1 = self._best_order_for_slot(
                    attacker1, valid[1], our_active, opp_active, battle
                )
        else:
            order1 = _pass_order_for_slot(1)

        # Prevent switching in the same mon to both slots
        def _is_switch(o: SingleBattleOrder | None) -> bool:
            if o is None:
                return False
            inner = getattr(o, "order", None)
            return (
                not isinstance(inner, Move)
                and not isinstance(o, PassBattleOrder)
            )

        def _switch_species(o: SingleBattleOrder | None) -> str:
            if o is None:
                return ""
            return str(getattr(getattr(o, "order", None), "species", "") or "").lower()

        if _is_switch(order0) and _is_switch(order1):
            if _switch_species(order0) == _switch_species(order1):
                sp = _switch_species(order0)
                print(
                    f"[ADOLENCE SWITCH] {battle.battle_tag} dedup same_incoming={sp} "
                    f"slot1_fallback_to_moves"
                )
                order1 = self._best_order_for_slot(
                    attacker1, valid[1], our_active, opp_active, battle
                )

        if order0 is None:
            pool = [
                o
                for o in valid[0]
                if isinstance(getattr(o, "order", None), Move)
                and getattr(o.order, "base_power", 0) > 0
                and not self._is_single_target_ally_attack(o)
            ]
            order0 = random.choice(pool) if pool else _pass_order_for_slot(0)

        if order1 is None:
            pool = [
                o
                for o in valid[1]
                if isinstance(getattr(o, "order", None), Move)
                and getattr(o.order, "base_power", 0) > 0
                and not self._is_single_target_ally_attack(o)
            ]
            order1 = random.choice(pool) if pool else _pass_order_for_slot(1)

        return DoubleBattleOrder(order0, order1)


async def main():
    # Regulation I format; same team via ConstantTeambuilder — AdolenceBot vs ToddlerBot.
    vgc_format = "gen9vgc2026regi"
    team = ConstantTeambuilder(REGULATION_I_TEAM)
    suffix1 = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    suffix2 = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))

    bot_1 = AdolenceBot(
        battle_format=vgc_format,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        account_configuration=AccountConfiguration(f"AdolenceBot {suffix1}", ""),
    )
    bot_2 = ToddlerBot(
        battle_format=vgc_format,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        account_configuration=AccountConfiguration(f"ToddlerBot {suffix2}", ""),
    )

    total_battles = 200
    print(f"Running {total_battles} battles (AdolenceBot vs ToddlerBot)...")

    adolence_wins = 0
    toddler_wins = 0
    ties = 0

    results_path = Path(__file__).with_name("battle_results.csv")
    with results_path.open("w", encoding="utf-8") as f:
        f.write("battle_id,winner\n")

    for i in range(total_battles):
        await bot_1.battle_against(bot_2, n_battles=1)

        winner_label = "tie"
        bot_1.won = None
        finished = [b for b in bot_1.battles.values() if b.finished]
        if finished:
            latest = max(finished, key=lambda b: getattr(b, "battle_tag", ""))
            bot_1.won = latest.won
        if bot_1.won is True:
            winner_label = "AdolenceBot"
            adolence_wins += 1
        elif bot_1.won is False:
            winner_label = "ToddlerBot"
            toddler_wins += 1
        else:
            ties += 1

        with results_path.open("a", encoding="utf-8") as f:
            f.write(f"{i + 1},{winner_label}\n")

        print(f"Battle {i + 1}/{total_battles} done, winner={winner_label}")

        bot_1.reset_battles()
        bot_2.reset_battles()

    winrate_pct = 100.0 * adolence_wins / total_battles
    print("Done!")
    print(
        f"AdolenceBot winrate ({total_battles} battles): "
        f"{adolence_wins}W / {toddler_wins}L / {ties}T -> "
        f"{winrate_pct:.2f}% wins (vs ToddlerBot)"
    )


if __name__ == "__main__":
    asyncio.run(main())
